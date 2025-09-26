#!/usr/bin/env python3
# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
# -*- coding: utf-8 -*-

"""
Script: extract_open3dsg_style_labels.py

Methodology (very short):
- Nodes: zero-shot object naming by CLIP similarity to a predefined vocabulary.
- Edges: open-set relation text via InstructBLIP (Q-Former + LLM) prompted with the two object names,
         operating on precomputed relation image embeddings.

This mirrors the Open3DSG idea of CLIP-like models for objects but BLIP(+LLM) for relationships,
rather than CLIP for relationships, which the paper shows is ill-suited. See paper/project: arXiv
and project page. Weights for InstructBLIP are on Hugging Face. 
Refs: Open3DSG paper & supp., project page, InstructBLIP model card.
"""

from __future__ import annotations
import argparse
import os
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import clip  # openai-clip

# We rely on the Open3DSG custom InstructBLIP wrapper which accepts precomputed image embeddings.
# If you don't have Open3DSG in your PYTHONPATH, install/clone it and `pip install -e .`.
try:
    from transformers import InstructBlipProcessor
    from open3dsg.models.custom_instruct_blip import InstructBlipForConditionalGeneration
except Exception as e:
    raise RuntimeError(
        "Could not import Open3DSG custom InstructBLIP. Make sure the Open3DSG repo is installed "
        "and accessible (it provides a model that accepts precomputed relation image embeddings)."
    ) from e


# ----------------------------- feature loading ------------------------------ #

def _find_first_dir_with_prefix(root: str, prefixes: List[str]) -> Optional[str]:
    for d in sorted(os.listdir(root)):
        full = os.path.join(root, d)
        if not os.path.isdir(full):
            continue
        for p in prefixes:
            if d.startswith(p):
                return d
    return None


def load_node_embeddings(feature_dir: str, scene_id: str) -> Tuple[torch.Tensor, torch.Tensor, str]:
    """
    Returns:
        node_embs: [N_valid x D] float32
        valid_idx: [N_valid] long (indices into original graph nodes)
        node_model_name: str like "ViT-L/14@336px" or "OpenSeg"
    """
    obj_dir = _find_first_dir_with_prefix(feature_dir, ["export_obj_clip_emb_clip_"])
    if obj_dir is None:
        raise FileNotFoundError("Missing node embeddings dir (export_obj_clip_emb_clip_*) in features.")
    valid_dir = os.path.join(feature_dir, "export_obj_clip_valids")
    emb_path = os.path.join(feature_dir, obj_dir, f"{scene_id}.pt")
    valid_path = os.path.join(valid_dir, f"{scene_id}.pt")

    node_embs = torch.load(emb_path, map_location="cpu").float()
    valids = torch.load(valid_path, map_location="cpu").bool()
    valid_idx = torch.nonzero(valids, as_tuple=False).squeeze(1)
    node_embs = node_embs[valid_idx].contiguous()
    # Clip model name is suffix of the directory
    node_model_name = obj_dir.split("export_obj_clip_emb_clip_")[1]
    return node_embs, valid_idx, node_model_name


def load_edge_embeddings(feature_dir: str, scene_id: str) -> torch.Tensor:
    """
    Prefer BLIP relation embeddings (export_rel_blip_emb_*). Fallback to export_rel_clip_emb_clip_*
    Returns tensor of shape [E x T x D] (preferred) or [E x D] depending on dump.
    """
    rel_dir = _find_first_dir_with_prefix(
        feature_dir,
        ["export_rel_blip_emb_", "export_rel_blip_", "export_rel_clip_emb_clip_"]
    )
    if rel_dir is None:
        raise FileNotFoundError(
            "Missing relation embeddings. Expected one of: "
            "export_rel_blip_emb_*, export_rel_blip_*, export_rel_clip_emb_clip_*"
        )
    path = os.path.join(feature_dir, rel_dir, f"{scene_id}.pt")
    rel_embs = torch.load(path, map_location="cpu").float()
    return rel_embs


def load_graph(graph_pkl: str) -> Dict:
    with open(graph_pkl, "rb") as f:
        return pickle.load(f)


# -------------------------- node label prediction --------------------------- #

def _clip_model_for(node_model_name: str) -> str:
    # Map OpenSeg node features to a CLIP text space used for querying
    if node_model_name.lower() == "openseg":
        return "ViT-L/14@336px"
    return node_model_name.replace("-", "/")


@torch.no_grad()
def encode_text_clip(texts: List[str], clip_model_name: str, device: torch.device) -> torch.Tensor:
    model, _ = clip.load(clip_model_name, device=device)
    toks = clip.tokenize(texts).to(device)
    embs = model.encode_text(toks)
    return (embs / embs.norm(dim=-1, keepdim=True)).float()


def read_candidate_list(path: Optional[str]) -> List[str]:
    if path is None:
        # A small default list; pass your own with --candidate_classes for your dataset.
        return [
            "wall","floor","cabinet","bed","chair","sofa","table","door","window","bookshelf","picture",
            "counter","desk","curtain","refrigerator","toilet","sink","bathtub","lamp","otherfurniture",
            "tv","microwave","oven","mirror","pillow","blanket","dresser","closet","shelf"
        ]
    with open(path, "r") as f:
        return [ln.strip() for ln in f if ln.strip()]


@torch.no_grad()
def predict_object_names(
    node_embs: torch.Tensor,
    valid_idx: torch.Tensor,
    node_model_name: str,
    candidate_classes: List[str],
    device: torch.device,
    add_scene_context: bool = False
) -> Dict[int, str]:
    """
    Returns {original_node_idx: predicted_name}
    """
    node_embs = (node_embs / node_embs.norm(dim=-1, keepdim=True)).to(device)
    clip_name = _clip_model_for(node_model_name)
    texts = [f"A {c} in a scene" if add_scene_context else c for c in candidate_classes]
    text_embs = encode_text_clip(texts, clip_name, device)

    sim = node_embs @ text_embs.t()  # [N_valid x C]
    top = sim.argmax(dim=1).tolist()
    names = [candidate_classes[i] for i in top]

    mapping = {}
    for original, name in zip(valid_idx.tolist(), names):
        mapping[int(original)] = name
    return mapping


# --------------------------- relation generation ---------------------------- #

def make_blip_prompts(pairs: List[Tuple[str, str]]) -> List[str]:
    """Construct BLIP prompts for generating caption and relation strings."""
    out = []
    for a, b in pairs:
        other = "other " if a == b else ""
        out.append(
            f"Describe the relationship between the {a} and the {other}{b}.\n"
            "Respond in two lines:\n"
            f"Caption: Start with \"the {a}\" and describe the scene.\n"
            "Relation: Provide only the relation phrase between them (no object names or extra details)."
        )
    return out


@torch.no_grad()
def blip_generate_relations(
    edge_embs: torch.Tensor,                   # [E x T x D] preferred; [E x D] ok (will add token dim)
    name_pairs: List[Tuple[str, str]],
    processor: InstructBlipProcessor,
    blip: InstructBlipForConditionalGeneration,
    device: torch.device,
    beams: int = 5,
    batch_size: int = 8
) -> Tuple[List[str], List[str]]:
    assert len(edge_embs) == len(name_pairs), "edge_embs and name_pairs must have same length"
    if edge_embs.ndim == 2:
        edge_embs = edge_embs.unsqueeze(1)  # [E x 1 x D]

    E = edge_embs.shape[0]
    # if an embedding row is fully NaN, mark it
    mask_nan = torch.isnan(edge_embs).all(dim=-1).all(dim=-1)  # [E]
    preds, caps = [], []

    model_dtype: Optional[torch.dtype] = None
    try:
        model_dtype = next(blip.parameters()).dtype
    except StopIteration:
        model_dtype = None

    for s in range(0, E, batch_size):
        e = min(E, s + batch_size)
        img_emb = edge_embs[s:e].to(device)
        if model_dtype is not None and img_emb.is_floating_point():
            if device.type == "cuda" and img_emb.dtype != model_dtype:
                img_emb = img_emb.to(model_dtype)
        pairs = name_pairs[s:e]
        qs = make_blip_prompts(pairs)

        inputs = processor(images=None, text=qs, return_tensors="pt", padding=True).to(device)
        out = blip.generate_caption(
            img_emb,
            **inputs,
            do_sample=False,
            num_beams=beams,
            max_length=20,
            min_length=15,
            repetition_penalty=1.5,
            length_penalty=0.7,
            temperature=1.0,
        )
        out[out == processor.tokenizer.pad_token_id * 0 - 1] = processor.tokenizer.pad_token_id  # safety; keep like repo
        texts = processor.batch_decode(out, skip_special_tokens=True)

        m = mask_nan[s:e].tolist()
        for i, text in enumerate(texts):
            cap, rel = "none", "none"
            if "Relation:" in text:
                cap_part, rel_part = text.split("Relation:", 1)
            else:
                cap_part, rel_part = text, ""
            if "Caption:" in cap_part:
                cap = cap_part.split("Caption:", 1)[1].strip()
            else:
                cap = cap_part.strip()
            rel = rel_part.strip().split("\n")[0].strip().rstrip(".")

            a, b = pairs[i]
            rel_l = rel.lower()
            if a.lower() in rel_l or b.lower() in rel_l:
                print(f"[warn] relation contains object names: '{rel}' for pair ({a}, {b})")
                rel = "none"

            if m[i]:
                cap, rel = "none", "none"

            caps.append(cap if cap else "none")
            preds.append(rel if rel else "none")

    return preds, caps


# ----------------------------------- CLI ------------------------------------ #

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Open-vocabulary node & relation labeling (CLIP nodes, InstructBLIP edges)")
    ap.add_argument("--features", required=True, help="Directory with exported features")
    ap.add_argument("--graph", required=True, help="Pickled graph (data_dict_*.pkl)")
    ap.add_argument("--scene", required=True, help="Scene id (filename stem for feature tensors)")
    ap.add_argument("--out", required=True, help="Output TSV path")
    ap.add_argument("--candidate_classes", default=None, help="Optional text file: one object class per line")
    ap.add_argument("--context_words", action="store_true", help='Prefix candidates with "A <cls> in a scene"')
    ap.add_argument("--beams", type=int, default=5, help="Beam search for BLIP")
    ap.add_argument("--batch_size", type=int, default=8, help="Batch size for BLIP generation")
    ap.add_argument("--device", default=None, help="cpu/cuda")
    ap.add_argument("--hf_model", default="Salesforce/instructblip-flan-t5-xl",
                    help="Hugging Face model id for InstructBLIP (e.g., Salesforce/instructblip-flan-t5-xl)")
    ap.add_argument(
        "--torch_dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help=(
            "Torch dtype for loading InstructBLIP. Half/bfloat16 reduce memory and speed up GPU inference,"
            " but require CUDA support."
        ),
    )
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_lookup = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype_arg = args.torch_dtype.lower()
    if torch_dtype_arg != "auto" and torch_dtype_arg not in dtype_lookup:
        raise ValueError(f"Unsupported torch dtype: {args.torch_dtype}")
    model_dtype = dtype_lookup.get(torch_dtype_arg)
    if device.type == "cpu" and model_dtype in {torch.float16, torch.bfloat16}:
        raise ValueError("Half-precision dtypes require CUDA. Choose float32 or auto on CPU.")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # Load graph/edges and features
    graph = load_graph(args.graph)
    edges = np.asarray(graph["edges"], dtype=np.int64)  # [E x 2]

    node_embs, valid_idx, node_model_name = load_node_embeddings(args.features, args.scene)
    edge_embs = load_edge_embeddings(args.features, args.scene)

    # Node labels via CLIP similarity over a predefined vocabulary
    candidates = read_candidate_list(args.candidate_classes)
    name_map = predict_object_names(
        node_embs=node_embs,
        valid_idx=valid_idx,
        node_model_name=node_model_name,
        candidate_classes=candidates,
        device=device,
        add_scene_context=args.context_words
    )

    # Build edge-level (name_a, name_b) pairs aligned with relation embeddings order
    pairs: List[Tuple[str, str]] = []
    for s, t in edges.tolist():
        a = name_map.get(int(s), "object")
        b = name_map.get(int(t), "object")
        pairs.append((a, b))

    # Length alignment (in practice should match already)
    if len(pairs) != int(edge_embs.shape[0]):
        n = min(len(pairs), int(edge_embs.shape[0]))
        print(f"[warn] Mismatch: {len(pairs)} edges in graph vs {int(edge_embs.shape[0])} relation embeddings. Truncating to {n}.")
        pairs = pairs[:n]
        edge_embs = edge_embs[:n]

    # Load InstructBLIP (HF weights)
    dtype_msg = torch_dtype_arg if torch_dtype_arg == "auto" else str(model_dtype)
    print(f"[info] Loading InstructBLIP: {args.hf_model} on {device} (dtype={dtype_msg}) ...")
    processor = InstructBlipProcessor.from_pretrained(args.hf_model)
    from_pretrained_kwargs = {"torch_dtype": torch_dtype_arg if torch_dtype_arg == "auto" else model_dtype}
    blip = InstructBlipForConditionalGeneration.from_pretrained(args.hf_model, **from_pretrained_kwargs)
    blip.eval().to(device)

    # Edge relations via BLIP+LLM over relation image embeddings
    print(f"[info] Generating relations for {len(pairs)} edges ...")
    rels, caps = blip_generate_relations(
        edge_embs=edge_embs, name_pairs=pairs,
        processor=processor, blip=blip, device=device,
        beams=args.beams, batch_size=args.batch_size
    )

    # Write TSV
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("src_idx\tdst_idx\tsrc_name\tdst_name\trelation\tcaption\n")
        for (s, t), (a, b), r, c in zip(edges[:len(pairs)].tolist(), pairs, rels, caps):
            f.write(f"{s}\t{t}\t{a}\t{b}\t{r}\t{c}\n")

    print(f"[done] Wrote {args.out}")


if __name__ == "__main__":
    main()
