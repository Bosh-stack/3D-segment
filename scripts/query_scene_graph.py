#!/usr/bin/env python3
"""Command-line tool to query an Open3DSG-style scene graph using CLIP text embeddings.

This script implements the "CLIP-Only Scene-Graph Query" described in the
build specification. Given a TSV file of scene-graph edges and a query
described either via a JSON file/string or individual command-line flags,
it finds all source nodes that satisfy a simple relational pattern. The
pattern is defined by three strings: a source node label, a relation and
an end node label. The matching is performed in two stages using cosine
similarity of CLIP text embeddings only. No string normalisation,
heuristics or synonym expansions are applied.

Usage examples::

    # Query via JSON file
    python query_scene_graph.py --tsv edges.tsv --query-file query.json

    # Query via JSON string
    python query_scene_graph.py --tsv edges.tsv --query-json '{"query": ..., "graph_pattern": {"relations": [{"source": "computer", "end": "desk", "relation": "on"}]}}'

    # Query via individual fields
    python query_scene_graph.py --tsv edges.tsv --source "computer" --end "desk" --relation "on"

The default similarity thresholds are tuned for CLIP ViT-B/32 on CPU but can
be overridden via ``--tau-label`` and ``--tau-relation``. The script prints a
JSON object to stdout with three fields:

* ``nodes``: sorted list of unique source node indices that match the pattern.
* ``matches``: list of detailed edge matches including similarity scores.
* ``count``: number of unique source nodes matched (redundant with
  ``len(nodes)`` for convenience).

This tool depends on PyTorch and either ``open_clip`` or ``clip``. It
automatically falls back to the ``clip`` package if ``open_clip`` is not
installed. The text encoder runs on CPU by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _load_clip_model() -> Tuple[Any, Any, str, bool]:
    """Load a CLIP text model from open_clip or clip."""
    try:
        import torch  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency check
        raise ImportError(
            "torch is required to run this script. Please install PyTorch"
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        import open_clip  # type: ignore

        model_name = "ViT-B-32"
        model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained="openai"
        )
        tokenizer = open_clip.get_tokenizer(model_name)
        model = model.to(device)
        model.eval()
        return model, tokenizer, device, True
    except ImportError:
        pass

    try:
        import clip  # type: ignore

        model, _ = clip.load("ViT-B/32", device=device)
        tokenizer = clip.tokenize
        model.eval()
        return model, tokenizer, device, False
    except ImportError as exc:  # pragma: no cover - dependency check
        raise ImportError(
            "Neither open_clip nor clip could be imported. Please install one of them."
        ) from exc


def _encode_texts(
    model: Any,
    tokenizer: Any,
    device: str,
    texts: Iterable[str],
    using_open_clip: bool,
):
    """Encode a list of texts into normalised CLIP embeddings."""
    import torch  # type: ignore

    with torch.no_grad():
        tokens = tokenizer(list(texts))
        tokens = tokens.to(device)
        embeddings = model.encode_text(tokens)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings.cpu()


def _cosine_similarity(v1: Any, v2: Any) -> float:
    """Compute cosine similarity between two 1D torch tensors."""
    import torch  # type: ignore

    v1 = v1.to(dtype=torch.float32)
    v2 = v2.to(dtype=torch.float32)
    return float((v1 @ v2) / (v1.norm() * v2.norm() + 1e-6))


def parse_query(args: argparse.Namespace) -> Tuple[str, str, str]:
    """Extract the (source, end, relation) triple from CLI arguments."""
    spec_provided = sum(
        [
            1 if args.query_file else 0,
            1 if args.query_json else 0,
            1 if args.source and args.end and args.relation else 0,
        ]
    )
    if spec_provided != 1:
        raise ValueError(
            "Exactly one of --query-file, --query-json or the trio --source/--end/--relation must be provided"
        )

    if args.query_file:
        data = json.loads(Path(args.query_file).read_text(encoding="utf-8"))
    elif args.query_json:
        data = json.loads(args.query_json)
    else:
        return args.source, args.end, args.relation

    try:
        relation_spec = data["graph_pattern"]["relations"][0]
        return (
            relation_spec["source"],
            relation_spec["end"],
            relation_spec["relation"],
        )
    except Exception as exc:  # pragma: no cover - defensive parsing
        raise ValueError(
            "Invalid query JSON format. Expected graph_pattern.relations[0] with keys 'source', 'end' and 'relation'."
        ) from exc


def read_edges(tsv_path: str) -> List[Dict[str, Any]]:
    """Load edges from a TSV file with a header."""
    edges: List[Dict[str, Any]] = []
    with open(tsv_path, newline="", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj, delimiter="\t")
        required_fields = {"src_idx", "dst_idx", "src_name", "dst_name", "relation"}
        missing = required_fields - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "TSV file is missing required fields: " + ", ".join(sorted(missing))
            )
        for row in reader:
            for key in ("src_idx", "dst_idx"):
                try:
                    row[key] = int(row[key])
                except Exception:
                    pass
            edges.append(row)
    return edges


def _get_embedding(
    cache: Dict[str, Any],
    text: str,
    *,
    model: Any,
    tokenizer: Any,
    device: str,
    using_open_clip: bool,
):
    """Return a cached embedding for *text*."""
    if text not in cache:
        cache[text] = _encode_texts(model, tokenizer, device, [text], using_open_clip)[0]
    return cache[text]


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Query an Open3DSG-style scene graph using CLIP text embeddings."
    )
    parser.add_argument(
        "--tsv",
        required=True,
        help="Path to the TSV file containing graph edges (with a header)",
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--query-file",
        help="Path to a JSON file describing the query. Only graph_pattern.relations[0] is used.",
    )
    group.add_argument(
        "--query-json",
        help="String containing a JSON query. Only graph_pattern.relations[0] is used.",
    )
    parser.add_argument("--source", help="Source node label for the pattern")
    parser.add_argument("--end", help="End node label for the pattern")
    parser.add_argument("--relation", help="Relation label for the pattern")
    parser.add_argument(
        "--tau-label",
        type=float,
        default=0.27,
        help="Similarity threshold for node labels (default: 0.27)",
    )
    parser.add_argument(
        "--tau-relation",
        type=float,
        default=0.22,
        help="Similarity threshold for relation labels (default: 0.22)",
    )

    args = parser.parse_args(argv)

    source_label, end_label, relation_label = parse_query(args)
    edges = read_edges(args.tsv)

    model, tokenizer, device, using_open_clip = _load_clip_model()

    query_embeddings = _encode_texts(
        model,
        tokenizer,
        device,
        [source_label, end_label, relation_label],
        using_open_clip,
    )
    src_query_vec, end_query_vec, rel_query_vec = query_embeddings

    label_cache: Dict[str, Any] = {}
    relation_cache: Dict[str, Any] = {}

    candidate_sources: Dict[str, float] = {}
    for row in edges:
        src_label = row["src_name"]
        if src_label in candidate_sources:
            continue
        src_emb = _get_embedding(
            label_cache,
            src_label,
            model=model,
            tokenizer=tokenizer,
            device=device,
            using_open_clip=using_open_clip,
        )
        sim = _cosine_similarity(src_emb, src_query_vec)
        if sim >= args.tau_label:
            candidate_sources[src_label] = sim

    matched_edges: List[Dict[str, Any]] = []
    unique_src_ids: set[Any] = set()

    for row in edges:
        src_label = row["src_name"]
        if src_label not in candidate_sources:
            continue

        rel_emb = _get_embedding(
            relation_cache,
            row["relation"],
            model=model,
            tokenizer=tokenizer,
            device=device,
            using_open_clip=using_open_clip,
        )
        rel_sim = _cosine_similarity(rel_emb, rel_query_vec)
        if rel_sim < args.tau_relation:
            continue

        end_emb = _get_embedding(
            label_cache,
            row["dst_name"],
            model=model,
            tokenizer=tokenizer,
            device=device,
            using_open_clip=using_open_clip,
        )
        end_sim = _cosine_similarity(end_emb, end_query_vec)
        if end_sim < args.tau_label:
            continue

        unique_src_ids.add(row["src_idx"])
        match: Dict[str, Any] = {
            "src_idx": row["src_idx"],
            "dst_idx": row["dst_idx"],
            "src_name": row["src_name"],
            "dst_name": row["dst_name"],
            "relation": row["relation"],
            "sim_source_node": round(candidate_sources[src_label], 4),
            "sim_relation": round(rel_sim, 4),
            "sim_end_node": round(end_sim, 4),
        }
        if row.get("caption"):
            match["caption"] = row["caption"]
        matched_edges.append(match)

    output = {
        "nodes": sorted(unique_src_ids),
        "matches": matched_edges,
        "count": len(unique_src_ids),
    }

    json.dump(output, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
