"""Query node features and export colourised point cloud.

This utility loads precomputed node embeddings for a scene, searches for the
nodes most similar to a given text query and exports their point clouds with
distinct colours. It assumes that node embeddings were produced by
``precompute_2d_features``.

Example
-------
```bash
python open3dsg/scripts/query_node_features.py \
    --features path/to/features \
    --graph path/to/data_dict.pkl \
    --scene scene_id \
    --word "chair" \
    --topk 5 \
    --out_ply queried.ply \
    --log queried.txt
```
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import torch
import clip
import open3d as o3d

# Simple 20-colour palette for visualisation
_COLOR_PALETTE = np.array([
    [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200],
    [245, 130, 48], [145, 30, 180], [70, 240, 240], [240, 50, 230],
    [210, 245, 60], [250, 190, 190], [0, 128, 128], [230, 190, 255],
    [170, 110, 40], [255, 250, 200], [128, 0, 0], [170, 255, 195],
    [128, 128, 0], [255, 215, 180], [0, 0, 128], [128, 128, 128],
], dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query node features and export colourised point clouds",
    )
    parser.add_argument(
        "--features",
        required=True,
        help="Directory containing export_obj_clip_emb_clip_* and export_obj_clip_valids",
    )
    parser.add_argument(
        "--graph",
        required=True,
        help="Pickled data_dict_*.pkl for the scene",
    )
    parser.add_argument("--scene", required=True, help="Scene ID used for the feature files")
    parser.add_argument("--word", required=True, help="Text query")
    parser.add_argument("--topk", type=int, default=5, help="Number of nodes to export")
    parser.add_argument("--out_ply", required=True, help="Output PLY file")
    parser.add_argument("--log", required=True, help="Text log file path")
    return parser.parse_args()


def _find_feature_dirs(root: str) -> tuple[str, str]:
    obj_dir = None
    for d in os.listdir(root):
        if d.startswith("export_obj_clip_emb_clip_"):
            obj_dir = d
            break
    if obj_dir is None:
        raise FileNotFoundError("Could not locate object embedding directory")
    valid_dir = os.path.join(root, "export_obj_clip_valids")
    return obj_dir, valid_dir


def load_embeddings(feature_dir: str, scene_id: str) -> tuple[torch.Tensor, torch.Tensor, str]:
    obj_dir, valid_dir = _find_feature_dirs(feature_dir)
    emb_path = os.path.join(feature_dir, obj_dir, f"{scene_id}.pt")
    valid_path = os.path.join(valid_dir, f"{scene_id}.pt")
    embeddings = torch.load(emb_path, map_location="cpu")
    valids = torch.load(valid_path, map_location="cpu").bool()
    valid_idx = torch.nonzero(valids, as_tuple=False).squeeze(1)
    embeddings = embeddings[valid_idx].float()
    model_name = obj_dir.split("export_obj_clip_emb_clip_")[1]
    return embeddings, valid_idx, model_name


def encode_text(query: str, model_name: str) -> torch.Tensor:
    if model_name.lower() == "openseg":
        model_name = "ViT-L/14@336px"
    else:
        model_name = model_name.replace("-", "/")
    model, _ = clip.load(model_name, device="cpu")
    with torch.no_grad():
        tokens = clip.tokenize([query])
        emb = model.encode_text(tokens)
    return emb


def load_object_clouds(graph: dict) -> list[np.ndarray]:
    if "objects_pcl_glob" in graph and len(graph["objects_pcl_glob"]) > 0:
        obj_pcls = np.asarray(graph["objects_pcl_glob"], dtype=np.float32)
        return [p[:, :3] for p in obj_pcls]
    obj_pcls = np.asarray(graph["objects_pcl"], dtype=np.float32)
    centers = np.asarray(graph["objects_center"], dtype=np.float32)
    scales = graph.get("objects_scale")
    if scales is not None:
        scales = np.asarray(scales, dtype=np.float32)
    clouds = []
    for idx, obj in enumerate(obj_pcls):
        pts = obj[:, :3]
        if scales is not None:
            s = np.asarray(scales[idx]).reshape(1, -1)
            pts = pts * s
        pts = pts + centers[idx]
        clouds.append(pts)
    return clouds


def assign_colors(n: int) -> np.ndarray:
    colors = np.zeros((n, 3), dtype=np.uint8)
    for i in range(n):
        colors[i] = _COLOR_PALETTE[i % len(_COLOR_PALETTE)]
    return colors


def main() -> None:
    args = parse_args()
    embeddings, valid_idx, model_name = load_embeddings(args.features, args.scene)
    text_emb = encode_text(args.word, model_name).to(embeddings.dtype)
    embeddings = torch.nn.functional.normalize(embeddings, dim=1)
    text_emb = torch.nn.functional.normalize(text_emb, dim=1)
    sims = (embeddings @ text_emb.t()).squeeze(1)
    topk = min(args.topk, sims.numel())
    vals, idx = torch.topk(sims, k=topk)
    selected = valid_idx[idx].tolist()
    scores = vals.tolist()

    with open(args.graph, "rb") as f:
        graph = pickle.load(f)
    clouds = load_object_clouds(graph)

    colors = assign_colors(topk)
    all_pts = []
    all_cols = []
    log_lines = []
    for obj_idx, score, col in zip(selected, scores, colors):
        pts = np.asarray(clouds[obj_idx], dtype=np.float32)
        all_pts.append(pts)
        col_arr = np.tile(col, (pts.shape[0], 1))
        all_cols.append(col_arr)
        log_lines.append(f"{obj_idx}\t{col[0]} {col[1]} {col[2]}\t{score:.4f}\n")

    if all_pts:
        points = np.concatenate(all_pts, axis=0)
        cols = np.concatenate(all_cols, axis=0).astype(np.float32) / 255.0
    else:
        points = np.zeros((0, 3), dtype=np.float32)
        cols = np.zeros((0, 3), dtype=np.float32)

    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(points)
    pcl.colors = o3d.utility.Vector3dVector(cols)
    Path(args.out_ply).parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(args.out_ply, pcl)

    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    with open(args.log, "w") as f:
        f.writelines(log_lines)


if __name__ == "__main__":
    main()
