#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Precompute 2D features (OpenSeg/CLIP + BLIP) with **batched** forward passes
to keep peak VRAM low for very large graphs (1000s of nodes/edges).

Usage (examples):
  # OpenSeg/CLIP only, batched
  python open3dsg/scripts/precompute_2d_features.py \
    --dataset myset --clip_model OpenSeg --top_k_frames 2 \
    --max_nodes 500 --max_edges 1000 \
    --out_dir ./node_500_edges_2 --gpus 3 \
    --clip_batch 64 --amp

  # BLIP only
  python open3dsg/scripts/precompute_2d_features.py \
    --dataset myset --clip_model none --blip \
    --out_dir ./blip_feats --blip_batch 16 --amp

Notes:
- This script assumes your project’s data loaders already yield the
  tensors/images the previous version expected. Only the encoding step
  is changed to iterate in small chunks.
- If your environment has mixed precision enabled, `--amp` is the easiest win.
"""

import argparse
import os
import math
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch import nn
from torch.cuda.amp import autocast

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def _device(rank: int) -> torch.device:
    return torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

def _to_half_if(t: torch.Tensor, use_fp16: bool) -> torch.Tensor:
    return t.half() if use_fp16 else t

@torch.no_grad()
def batched_encode_images(
    model: nn.Module,
    imgs: torch.Tensor,
    batch_size: int,
    device: torch.device,
    use_amp: bool = False,
    save_fp32: bool = False,
    forward_fn: str = "encode_image",
) -> torch.Tensor:
    """
    Encode image tensor in smaller chunks to cap VRAM.
    imgs: (N, C, H, W)
    Returns: (N, D)
    """
    assert imgs.dim() == 4, f"Expected (N,C,H,W), got {imgs.shape}"
    n = imgs.shape[0]
    outs = []
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        batch = imgs[s:e].to(device, non_blocking=True)
        with autocast(enabled=use_amp):
            if forward_fn == "encode_image":
                out = model.encode_image(batch)  # CLIP/OpenSeg-like API
            elif forward_fn == "forward":
                out = model(batch)
            else:
                # Fallback to attribute if model exposes custom method
                out = getattr(model, forward_fn)(batch)
        # Keep RAM small: move to CPU as we go.
        outs.append(out.float().cpu() if save_fp32 else out.half().cpu())
        del batch, out
        torch.cuda.empty_cache()
    return torch.cat(outs, dim=0)

def reshape_back(t: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
    return t.reshape(*shape)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def maybe_load_openseg(clip_model_name: str, device: torch.device):
    """
    Load your OpenSeg/CLIP model by name. If you already have a project-specific
    loader, call it here. Return None if clip_model_name == 'none'.
    """
    if clip_model_name.lower() in ["none", "no", "off"]:
        return None
    # Example stub: adapt to your project
    # from open3dsg.models.clip import get_clip_like_model
    # model = get_clip_like_model(clip_model_name)
    # model.eval().to(device)
    # return model
    raise NotImplementedError(
        "Hook up your OpenSeg/CLIP loader here (clip_model_name=%s)." % clip_model_name
    )

def maybe_load_blip(use_blip: bool, device: torch.device):
    if not use_blip:
        return None
    # Example stub: adapt to your project
    # from open3dsg.models.blip import get_blip_image_encoder
    # blip = get_blip_image_encoder()
    # blip.eval().to(device)
    # return blip
    raise NotImplementedError("Hook up your BLIP image encoder loader here.")

# --------------------------------------------------------------------------------------
# Core dumping logic (replace the monolithic single-pass encoding with batched version)
# --------------------------------------------------------------------------------------

@torch.no_grad()
def dump_features_for_scene(
    scene_id: str,
    *,
    dataset,
    out_dir: Path,
    clip_model: Optional[nn.Module],
    blip_model: Optional[nn.Module],
    clip_batch: int,
    blip_batch: int,
    amp: bool,
    save_fp32: bool,
    device: torch.device,
    scales: int,
    top_k_frames: int,
):
    """
    Expected dataset contract (same as your previous script):
      dataset.get_scene_images(scene_id) -> dict with keys:
        - "node_imgs": Tensor (N_nodes * top_k_frames * scales, 3, H, W)
        - "edge_imgs": Tensor (N_edges * top_k_frames * scales, 3, H, W)  [optional]
        - "node_shape": tuple (N_nodes, top_k_frames, scales)
        - "edge_shape": tuple (N_edges, top_k_frames, scales)            [optional]
      The shapes let us reshape the flat encodings back to [N, top_k, S, D].
    """
    pack = dataset.get_scene_images(scene_id, top_k_frames=top_k_frames, scales=scales)

    # ------------------------ OpenSeg / CLIP node features -------------------------
    if clip_model is not None and "node_imgs" in pack and pack["node_imgs"] is not None:
        node_imgs = pack["node_imgs"]  # (N_all, 3, H, W)
        node_shape = pack["node_shape"]  # (N_nodes, top_k_frames, scales)
        n_nodes, tk, sc = node_shape
        # Encode in chunks
        node_feats = batched_encode_images(
            clip_model, node_imgs, clip_batch, device, use_amp=amp,
            save_fp32=save_fp32, forward_fn="encode_image"
        )  # (N_all, D)
        # Reshape back to (N_nodes, top_k_frames, scales, D)
        d = node_feats.shape[-1]
        node_feats = reshape_back(node_feats, (n_nodes, tk, sc, d))
        # Save
        out_nodes = out_dir / scene_id / "clip_nodes.pt"
        ensure_dir(out_nodes.parent)
        torch.save(node_feats, out_nodes)

    # ------------------------ OpenSeg / CLIP edge features -------------------------
    if clip_model is not None and "edge_imgs" in pack and pack["edge_imgs"] is not None:
        edge_imgs = pack["edge_imgs"]
        edge_shape = pack["edge_shape"]  # (N_edges, top_k_frames, scales)
        n_edges, tk, sc = edge_shape
        edge_feats = batched_encode_images(
            clip_model, edge_imgs, clip_batch, device, use_amp=amp,
            save_fp32=save_fp32, forward_fn="encode_image"
        )
        d = edge_feats.shape[-1]
        edge_feats = reshape_back(edge_feats, (n_edges, tk, sc, d))
        out_edges = out_dir / scene_id / "clip_edges.pt"
        ensure_dir(out_edges.parent)
        torch.save(edge_feats, out_edges)

    # ------------------------------ BLIP features ----------------------------------
    if blip_model is not None and "node_imgs" in pack and pack["node_imgs"] is not None:
        node_imgs = pack["node_imgs"]
        node_shape = pack["node_shape"]
        n_nodes, tk, sc = node_shape
        node_feats = batched_encode_images(
            blip_model, node_imgs, blip_batch, device, use_amp=amp,
            save_fp32=save_fp32, forward_fn="forward"  # adapt if your BLIP uses encode_image
        )
        d = node_feats.shape[-1]
        node_feats = reshape_back(node_feats, (n_nodes, tk, sc, d))
        out_nodes_blip = out_dir / scene_id / "blip_nodes.pt"
        ensure_dir(out_nodes_blip.parent)
        torch.save(node_feats, out_nodes_blip)

    if blip_model is not None and "edge_imgs" in pack and pack["edge_imgs"] is not None:
        edge_imgs = pack["edge_imgs"]
        edge_shape = pack["edge_shape"]
        n_edges, tk, sc = edge_shape
        edge_feats = batched_encode_images(
            blip_model, edge_imgs, blip_batch, device, use_amp=amp,
            save_fp32=save_fp32, forward_fn="forward"
        )
        d = edge_feats.shape[-1]
        edge_feats = reshape_back(edge_feats, (n_edges, tk, sc, d))
        out_edges_blip = out_dir / scene_id / "blip_edges.pt"
        ensure_dir(out_edges_blip.parent)
        torch.save(edge_feats, out_edges_blip)


# --------------------------------------------------------------------------------------
# CLI / Orchestration
# --------------------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("Precompute 2D features (batched)")
    # existing/common flags
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--clip_model", type=str, default="OpenSeg",
                   help="OpenSeg / CLIP model name. Use 'none' to disable CLIP.")
    p.add_argument("--blip", action="store_true", help="Enable BLIP export.")
    p.add_argument("--no_blip", action="store_true", help="Disable BLIP export.")
    p.add_argument("--scales", type=int, default=3)
    p.add_argument("--top_k_frames", type=int, default=5)
    p.add_argument("--max_nodes", type=int, default=0)
    p.add_argument("--max_edges", type=int, default=0)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--gpus", type=int, default=1)

    # NEW: batching & precision
    p.add_argument("--clip_batch", type=int, default=64, help="Chunk size for CLIP/OpenSeg")
    p.add_argument("--blip_batch", type=int, default=32, help="Chunk size for BLIP")
    p.add_argument("--amp", action="store_true", help="Use AMP (mixed precision)")
    p.add_argument("--save_fp32", action="store_true",
                   help="Save features as fp32 (default saves fp16 to cut disk/RAM)")

    return p.parse_args()

def main():
    args = parse_args()

    # Choose a GPU (simple round-robin by rank 0)
    device = _device(0)

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    # ------------------------------------------------------------------
    # Load dataset – plug in your project-specific dataset accessor here.
    # It just needs to provide:
    #   - dataset.scene_ids (iterable of scene ids)
    #   - dataset.get_scene_images(scene_id, top_k_frames, scales)
    # You can keep using your current dataset implementation.
    # ------------------------------------------------------------------
    # Example stub; replace with your loader:
    # from open3dsg.data import get_feature_dump_dataset
    # dataset = get_feature_dump_dataset(
    #     name=args.dataset,
    #     max_nodes=args.max_nodes,
    #     max_edges=args.max_edges,
    # )
    raise NotImplementedError(
        "Hook up your dataset loader here (use the same one you used before).\n"
        "It must expose `scene_ids` and `get_scene_images(scene_id, top_k_frames, scales)`."
    )

    # Load models
    clip_model = maybe_load_openseg(args.clip_model, device)
    blip_model = maybe_load_blip(use_blip=(args.blip and not args.no_blip), device=device)

    if clip_model is None and blip_model is None:
        print("[WARN] Both CLIP/OpenSeg and BLIP are disabled. Nothing to do.")
        return

    # Iterate scenes and dump features
    for sid in dataset.scene_ids:
        dump_features_for_scene(
            sid,
            dataset=dataset,
            out_dir=out_dir,
            clip_model=clip_model,
            blip_model=blip_model,
            clip_batch=max(1, int(args.clip_batch)),
            blip_batch=max(1, int(args.blip_batch)),
            amp=bool(args.amp),
            save_fp32=bool(args.save_fp32),
            device=device,
            scales=args.scales,
            top_k_frames=args.top_k_frames,
        )

if __name__ == "__main__":
    main()
