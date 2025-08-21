"""Query relation features and export colourised point cloud.

This utility loads precomputed node and edge embeddings for a scene. It first
selects the nodes most similar to ``word_a`` and, for each, finds the most
compatible neighbour with respect to ``word_b``. The similarity between the
connecting edge and ``relation`` is then computed. Point clouds of the selected
pairs are exported with identical colours. It assumes that the embeddings were
produced by ``precompute_2d_features``.

Example
-------
```bash
python open3dsg/scripts/query_relation_features.py \
    --features path/to/features \
    --graph path/to/data_dict.pkl \
    --scene scene_id \
    --word_a "chair" \
    --word_b "table" \
    --relation "next to" \
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
        description="Query relation features and export colourised point clouds",
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
    parser.add_argument("--word_a", required=True, help="First text query")
    parser.add_argument("--word_b", required=True, help="Neighbour text query")
    parser.add_argument("--relation", required=True, help="Edge relation text query")
    parser.add_argument("--topk", type=int, default=5, help="Number of node pairs to export")
    parser.add_argument("--out_ply", required=True, help="Output PLY file")
    parser.add_argument("--log", required=True, help="Text log file path")
    return parser.parse_args()


def _find_node_feature_dirs(root: str) -> tuple[str, str]:
    obj_dir = None
    for d in os.listdir(root):
        if d.startswith("export_obj_clip_emb_clip_"):
            obj_dir = d
            break
    if obj_dir is None:
        raise FileNotFoundError("Could not locate object embedding directory")
    valid_dir = os.path.join(root, "export_obj_clip_valids")
    return obj_dir, valid_dir


def load_node_embeddings(feature_dir: str, scene_id: str) -> tuple[torch.Tensor, torch.Tensor, str]:
    obj_dir, valid_dir = _find_node_feature_dirs(feature_dir)
    emb_path = os.path.join(feature_dir, obj_dir, f"{scene_id}.pt")
    valid_path = os.path.join(valid_dir, f"{scene_id}.pt")
    embeddings = torch.load(emb_path, map_location="cpu")
    valids = torch.load(valid_path, map_location="cpu").bool()
    valid_idx = torch.nonzero(valids, as_tuple=False).squeeze(1)
    embeddings = embeddings[valid_idx].float()
    model_name = obj_dir.split("export_obj_clip_emb_clip_")[1]
    return embeddings, valid_idx, model_name


def load_edge_embeddings(feature_dir: str, scene_id: str) -> torch.Tensor:
    rel_dir = None
    for d in os.listdir(feature_dir):
        if d.startswith("export_rel_clip_emb_clip_"):
            rel_dir = d
            break
    if rel_dir is None:
        raise FileNotFoundError("Could not locate relationship embedding directory")
    emb_path = os.path.join(feature_dir, rel_dir, f"{scene_id}.pt")
    embeddings = torch.load(emb_path, map_location="cpu").float()
    return embeddings


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
    node_embs, valid_idx, model_name = load_node_embeddings(args.features, args.scene)
    edge_embs = load_edge_embeddings(args.features, args.scene)

    text_a = encode_text(args.word_a, model_name).to(node_embs.dtype)
    text_b = encode_text(args.word_b, model_name).to(node_embs.dtype)
    text_r = encode_text(args.relation, model_name).to(node_embs.dtype)

    node_embs = torch.nn.functional.normalize(node_embs, dim=1)
    edge_embs = torch.nn.functional.normalize(edge_embs, dim=1)
    text_a = torch.nn.functional.normalize(text_a, dim=1)
    text_b = torch.nn.functional.normalize(text_b, dim=1)
    text_r = torch.nn.functional.normalize(text_r, dim=1)

    sims_a = (node_embs @ text_a.t()).squeeze(1)
    topk = min(args.topk, sims_a.numel())
    vals_a, idx_a = torch.topk(sims_a, k=topk)
    selected = valid_idx[idx_a].tolist()
    scores_a = vals_a.tolist()

    with open(args.graph, "rb") as f:
        graph = pickle.load(f)
    clouds = load_object_clouds(graph)
    edges = np.asarray(graph["edges"], dtype=np.int64)

    adjacency: dict[int, list[tuple[int, int]]] = {}
    for e_idx, (s, t) in enumerate(edges):
        adjacency.setdefault(int(s), []).append((int(t), e_idx))
        adjacency.setdefault(int(t), []).append((int(s), e_idx))

    emb_idx_by_obj = {int(o): i for i, o in enumerate(valid_idx.tolist())}

    colors = assign_colors(topk)
    all_pts: list[np.ndarray] = []
    all_cols: list[np.ndarray] = []
    log_lines: list[str] = []

    for obj_a, sim_a, col in zip(selected, scores_a, colors):
        best_b = None
        best_b_sim = -1.0
        best_edge = None
        for nb, e_idx in adjacency.get(int(obj_a), []):
            nb_emb_idx = emb_idx_by_obj.get(nb)
            if nb_emb_idx is None:
                continue
            sim_b = float((node_embs[nb_emb_idx] @ text_b.t()).item())
            if sim_b > best_b_sim:
                best_b_sim = sim_b
                best_b = nb
                best_edge = e_idx
        if best_b is None or best_edge is None:
            continue
        sim_rel = float((edge_embs[best_edge] @ text_r.t()).item())
        log_lines.append(
            f"{obj_a}\t{best_b}\t{sim_a:.4f}\t{best_b_sim:.4f}\t{sim_rel:.4f}\n"
        )

        pts_a = np.asarray(clouds[obj_a], dtype=np.float32)
        pts_b = np.asarray(clouds[best_b], dtype=np.float32)
        col_arr_a = np.tile(col, (pts_a.shape[0], 1))
        col_arr_b = np.tile(col, (pts_b.shape[0], 1))
        all_pts.extend([pts_a, pts_b])
        all_cols.extend([col_arr_a, col_arr_b])

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
