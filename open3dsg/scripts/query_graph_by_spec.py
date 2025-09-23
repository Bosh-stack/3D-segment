# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
"""Query a scene graph for relation triples defined in a JSON spec."""

from __future__ import annotations

import argparse
import json
import pickle
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F


_DEFAULT_BLIP_TEXT_MODEL = "Salesforce/instructblip-flan-t5-xl"


@dataclass
class FeaturePack:
    """Container bundling the loaded and normalised feature tensors."""

    node_vectors: torch.Tensor
    node_ids: list[int]
    node_clip_text_model: str
    edge_vectors: torch.Tensor
    edge_clip_text_model: str | None
    edge_blip_text_model: str | None
    edge_encoder: str


def _find_feature_subdir(root: Path, prefixes: Sequence[str]) -> tuple[Path, str, str]:
    """Return the first sub-directory whose name starts with any of ``prefixes``."""

    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        for prefix in prefixes:
            if entry.name.startswith(prefix):
                suffix = entry.name[len(prefix) :]
                return entry, prefix, suffix
    joined = ", ".join(prefixes)
    raise FileNotFoundError(f"Could not locate feature directory with prefixes: {joined}")


def _load_node_features(
    feature_root: Path, scene_id: str
) -> tuple[torch.Tensor, torch.Tensor, str | None]:
    """Load node embeddings and validity mask for ``scene_id``."""

    node_dir, _, suffix = _find_feature_subdir(feature_root, ("export_obj_clip_emb_clip_",))
    emb_path = node_dir / f"{scene_id}.pt"
    embeddings = torch.load(emb_path, map_location="cpu")

    valid_dir = feature_root / "export_obj_clip_valids"
    valid_path = valid_dir / f"{scene_id}.pt"
    if valid_path.exists():
        valids = torch.load(valid_path, map_location="cpu").bool()
        valid_idx = torch.nonzero(valids, as_tuple=False).squeeze(1)
        embeddings = embeddings[valid_idx]
    else:
        valid_idx = torch.arange(embeddings.shape[0], dtype=torch.long)

    return embeddings.float(), valid_idx.to(torch.long), suffix or None


def _load_edge_features(
    feature_root: Path, scene_id: str
) -> tuple[torch.Tensor, str | None, str]:
    """Load relation embeddings for ``scene_id`` along with metadata."""

    prefixes = (
        "export_rel_clip_emb_clip_",
        "export_rel_blip_emb_",
        "export_rel_blip_",
    )
    edge_dir, prefix, suffix = _find_feature_subdir(feature_root, prefixes)
    emb_path = edge_dir / f"{scene_id}.pt"
    embeddings = torch.load(emb_path, map_location="cpu").float()

    encoder = "clip" if prefix == "export_rel_clip_emb_clip_" else "blip"
    return embeddings, suffix or None, encoder


def _extract_label(value: Any) -> str | None:
    """Extract the first textual label from ``value`` preserving order."""

    if value is None:
        return None

    if isinstance(value, str):
        label = value.strip()
        return label or None

    if isinstance(value, Mapping):
        for key in ("label", "name", "predicate", "relation", "synset", "category", "type"):
            if key in value:
                candidate = _extract_label(value[key])
                if candidate:
                    return candidate
        for candidate in value.values():
            resolved = _extract_label(candidate)
            if resolved:
                return resolved
        return None

    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        for item in value:
            candidate = _extract_label(item)
            if candidate:
                return candidate
        return None

    return _extract_label(str(value))


def _iter_edges(edges: Any) -> list[tuple[int, int]]:
    if edges is None:
        return []

    if hasattr(edges, "tolist"):
        edges = edges.tolist()

    result: list[tuple[int, int]] = []
    for entry in edges:
        if entry is None:
            continue
        if hasattr(entry, "tolist"):
            entry = entry.tolist()
        if isinstance(entry, Mapping):
            src = entry.get("source") or entry.get("from")
            dst = entry.get("target") or entry.get("to") or entry.get("end")
            try:
                result.append((int(src), int(dst)))
            except (TypeError, ValueError):
                continue
            continue
        try:
            src, dst = entry[:2]
            result.append((int(src), int(dst)))
        except (TypeError, ValueError, IndexError):
            continue
    return result


def _clip_model_from_export(name: str | None) -> str:
    if not name:
        return "ViT-L/14@336px"
    lowered = name.lower()
    if lowered == "openseg":
        return "ViT-L/14@336px"
    if "/" in name:
        return name
    parts = name.split("-")
    if len(parts) >= 3 and parts[0].startswith("ViT"):
        prefix = "-".join(parts[:2])
        suffix = "-".join(parts[2:])
        return f"{prefix}/{suffix}"
    if len(parts) == 2 and parts[0].startswith("ViT"):
        return f"{parts[0]}/{parts[1]}"
    return name.replace("-", "/")


def _normalize_matrix(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.numel() == 0:
        return matrix.clone()
    normalised = F.normalize(matrix, dim=-1)
    return torch.nan_to_num(normalised, nan=0.0, posinf=0.0, neginf=0.0)


def _normalize_vector(vector: torch.Tensor) -> torch.Tensor:
    if vector.numel() == 0:
        return vector.clone()
    normalised = F.normalize(vector, dim=-1)
    return torch.nan_to_num(normalised, nan=0.0, posinf=0.0, neginf=0.0)


def _prepare_edge_embeddings(raw: torch.Tensor) -> torch.Tensor:
    raw = raw.to(torch.float32)
    if raw.ndim == 3:
        if raw.shape[0] == 0:
            return raw.new_zeros((0, raw.shape[-1]))
        reduced = torch.nanmean(raw, dim=1)
    elif raw.ndim == 2:
        if raw.shape[0] == 0:
            return raw.new_zeros((0, raw.shape[-1]))
        reduced = raw
    else:
        raise ValueError(f"Unsupported relation embedding shape: {tuple(raw.shape)}")
    reduced = torch.nan_to_num(reduced, nan=0.0, posinf=0.0, neginf=0.0)
    return _normalize_matrix(reduced)


@lru_cache(maxsize=4)
def _load_clip_model(model_name: str) -> tuple[Any, Any]:  # type: ignore[valid-type]
    import clip  # pylint: disable=import-error

    model, _ = clip.load(model_name, device="cpu")
    model.eval()
    return model, clip


def encode_clip_texts(texts: Sequence[str], model_name: str) -> torch.Tensor:
    model, clip_module = _load_clip_model(model_name)
    tokens = clip_module.tokenize(list(texts))
    with torch.no_grad():
        features = model.encode_text(tokens)
    return features.to(torch.float32)


@lru_cache(maxsize=2)
def _load_blip_model(model_name: str) -> tuple[Any, Any]:  # type: ignore[valid-type]
    from transformers import AutoModel, AutoTokenizer  # pylint: disable=import-error

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model = model.to("cpu")
    model.eval()
    return model, tokenizer


def encode_blip_texts(texts: Sequence[str], model_name: str) -> torch.Tensor:
    model, tokenizer = _load_blip_model(model_name)
    inputs = tokenizer(list(texts), return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
    pooled = hidden.mean(dim=1)
    return pooled.to(torch.float32)


def _infer_scene_id(graph_path: Path) -> str | None:
    stem = graph_path.stem
    if stem.startswith("data_dict_"):
        stem = stem[len("data_dict_"):]
    return stem or None


def load_feature_pack(
    feature_root: Path,
    scene_id: str,
    clip_override: str | None,
    relation_clip_override: str | None,
    blip_override: str | None,
) -> FeaturePack:
    feature_root = feature_root.resolve()

    node_embs, valid_idx, node_model = _load_node_features(feature_root, scene_id)
    node_vectors = _normalize_matrix(node_embs.to(torch.float32))
    node_ids = [int(idx) for idx in valid_idx.to(torch.long).tolist()]
    node_clip_model = clip_override or _clip_model_from_export(node_model)

    edge_embs, edge_model, edge_encoder = _load_edge_features(feature_root, scene_id)
    edge_vectors = _prepare_edge_embeddings(edge_embs)

    relation_clip_model = relation_clip_override
    if relation_clip_model is None:
        if edge_encoder == "clip" and edge_model:
            relation_clip_model = _clip_model_from_export(edge_model)
        else:
            relation_clip_model = node_clip_model

    relation_blip_model = blip_override
    if edge_encoder == "blip" and relation_blip_model is None:
        relation_blip_model = _DEFAULT_BLIP_TEXT_MODEL

    return FeaturePack(
        node_vectors=node_vectors,
        node_ids=node_ids,
        node_clip_text_model=node_clip_model,
        edge_vectors=edge_vectors,
        edge_clip_text_model=relation_clip_model,
        edge_blip_text_model=relation_blip_model,
        edge_encoder=edge_encoder,
    )


def _graph_pattern_from_spec(spec: Any) -> Sequence[Mapping[str, Any]]:
    if isinstance(spec, Mapping):
        pattern = spec.get("graph_pattern", [])
    else:
        pattern = spec
    if isinstance(pattern, Sequence) and not isinstance(pattern, (str, bytes)):
        return pattern
    return []


def match_triple(
    query: Mapping[str, Any],
    features: FeaturePack,
    edges: Sequence[tuple[int, int]],
    node_index: Mapping[int, int],
) -> dict[str, Any]:
    src_label = _extract_label(query.get("source"))
    rel_label = _extract_label(query.get("relation"))
    dst_label = _extract_label(query.get("end"))

    result = {
        "query": {"source": src_label, "relation": rel_label, "end": dst_label},
        "match": None,
    }

    if not src_label or not rel_label or not dst_label:
        return result
    if features.node_vectors.numel() == 0 or features.edge_vectors.numel() == 0:
        return result

    src_vec = encode_clip_texts([src_label], features.node_clip_text_model)[0]
    dst_vec = encode_clip_texts([dst_label], features.node_clip_text_model)[0]
    src_vec = _normalize_vector(src_vec).to(features.node_vectors.dtype)
    dst_vec = _normalize_vector(dst_vec).to(features.node_vectors.dtype)

    intended_encoder = features.edge_encoder
    used_encoder = intended_encoder
    relation_vec: torch.Tensor
    fallback = False
    if intended_encoder == "blip":
        model_name = features.edge_blip_text_model
        try:
            relation_vec = encode_blip_texts([rel_label], model_name or _DEFAULT_BLIP_TEXT_MODEL)[0]
        except (ModuleNotFoundError, ImportError, OSError, RuntimeError, ValueError) as exc:
            warnings.warn(
                f"Failed to encode relation text with BLIP ({exc}). Falling back to CLIP.",
                RuntimeWarning,
                stacklevel=1,
            )
            relation_vec = encode_clip_texts([rel_label], features.edge_clip_text_model or features.node_clip_text_model)[0]
            used_encoder = "clip"
            fallback = True
    else:
        relation_vec = encode_clip_texts([rel_label], features.edge_clip_text_model or features.node_clip_text_model)[0]

    relation_vec = _normalize_vector(relation_vec).to(features.edge_vectors.dtype)

    node_scores_src = torch.matmul(features.node_vectors, src_vec)
    node_scores_dst = torch.matmul(features.node_vectors, dst_vec)

    edge_count = min(len(edges), features.edge_vectors.shape[0])
    if edge_count == 0:
        return result
    edge_scores = torch.matmul(features.edge_vectors[:edge_count], relation_vec)

    best: dict[str, Any] | None = None
    for edge_idx in range(edge_count):
        src_id, dst_id = edges[edge_idx]
        src_idx = node_index.get(int(src_id))
        dst_idx = node_index.get(int(dst_id))
        if src_idx is None or dst_idx is None:
            continue
        src_score = float(node_scores_src[src_idx])
        dst_score = float(node_scores_dst[dst_idx])
        rel_score = float(edge_scores[edge_idx])
        combined = src_score + dst_score + rel_score
        if best is None or combined > best["combined_score"]:
            best = {
                "source_id": int(src_id),
                "end_id": int(dst_id),
                "edge_index": edge_idx,
                "source_score": src_score,
                "end_score": dst_score,
                "relation_score": rel_score,
                "combined_score": combined,
            }

    if best is None:
        return result

    result["match"] = {
        "source": {"id": best["source_id"], "score": best["source_score"]},
        "end": {"id": best["end_id"], "score": best["end_score"]},
        "relation": {
            "edge_index": best["edge_index"],
            "pair": [best["source_id"], best["end_id"]],
            "score": best["relation_score"],
            "encoder": used_encoder,
            "intended_encoder": intended_encoder,
            "used_fallback": fallback,
        },
        "combined_score": best["combined_score"],
    }
    return result


def _load_graph(path: Path) -> Any:
    if path.suffix.lower() in {".json", ".jsn"}:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    with path.open("rb") as handle:
        return pickle.load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graph", type=Path, help="Path to the serialized data_dict")
    parser.add_argument("spec", type=Path, help="JSON file describing the query")
    parser.add_argument("--features", type=Path, required=True, help="Directory with precomputed features")
    parser.add_argument("--scene", help="Scene identifier (defaults to the graph stem)")
    parser.add_argument("--clip-text-model", help="Override CLIP text encoder for nodes")
    parser.add_argument(
        "--relation-clip-text-model",
        help="Override CLIP text encoder used for relation embeddings",
    )
    parser.add_argument("--blip-text-model", help="Override BLIP text encoder for relations")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene_id = args.scene or _infer_scene_id(args.graph)
    if not scene_id:
        raise ValueError("Could not infer scene identifier; please provide --scene")

    data_dict = _load_graph(args.graph)
    with args.spec.open("r", encoding="utf-8") as handle:
        spec = json.load(handle)

    pattern = _graph_pattern_from_spec(spec)
    edges = _iter_edges(data_dict.get("edges")) if isinstance(data_dict, Mapping) else []

    features = load_feature_pack(
        args.features,
        scene_id,
        args.clip_text_model,
        args.relation_clip_text_model,
        args.blip_text_model,
    )
    node_index = {int(nid): idx for idx, nid in enumerate(features.node_ids)}

    results: list[dict[str, Any]] = []
    for entry in pattern:
        if isinstance(entry, Mapping):
            results.append(match_triple(entry, features, edges, node_index))

    print(json.dumps(results))


if __name__ == "__main__":
    main()
