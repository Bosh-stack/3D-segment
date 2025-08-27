"""Merge graph nodes based on embedding similarity.

This utility loads a pickled scene graph and a tensor of node embeddings.
Existing edges are used to compute cosine similarities between connected
nodes.  Nodes whose similarity exceeds a user defined threshold are merged
using a union--find data structure.  During merging the point clouds are
aggregated in global coordinates and re-normalised.  Relationship lists are
rebuilt without duplicates and the resulting graph is written back in the
original pickle based format.  Optionally the merged instance point clouds
and the concatenated full point cloud can be exported to ``.ply``/``.npz``
files.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:  # optional dependency used for loading embeddings
    import torch
except Exception:  # pragma: no cover - torch may be unavailable
    torch = None  # type: ignore


class UnionFind:
    """Disjoint set union structure."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> None:
        xr, yr = self.find(x), self.find(y)
        if xr == yr:
            return
        if self.rank[xr] < self.rank[yr]:
            self.parent[xr] = yr
        elif self.rank[xr] > self.rank[yr]:
            self.parent[yr] = xr
        else:
            self.parent[yr] = xr
            self.rank[xr] += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge graph nodes whose embeddings are similar"
    )
    parser.add_argument("--graph", required=True, help="Path to pickled graph data")
    parser.add_argument(
        "--embeddings", required=True, help="Path to node embedding tensor"
    )
    parser.add_argument(
        "--threshold", type=float, required=True, help="Cosine similarity threshold"
    )
    parser.add_argument("--out", required=True, help="Output path for merged graph")
    parser.add_argument(
        "--instances-out",
        required=True,
        help="Directory to write merged instance .ply files",
    )
    parser.add_argument(
        "--pointcloud-out",
        required=True,
        help="Output file for merged full point cloud (.ply or .npz)",
    )
    args = parser.parse_args()

    input_paths = {Path(args.graph).resolve(), Path(args.embeddings).resolve()}
    output_paths = [
        Path(args.out).resolve(),
        Path(args.instances_out).resolve(),
        Path(args.pointcloud_out).resolve(),
    ]
    if len(set(output_paths)) != len(output_paths) or any(p in input_paths for p in output_paths):
        parser.error("Output paths must be distinct from inputs and from each other")
    return args


def load_graph(path: str) -> Dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def load_embeddings(path: str) -> np.ndarray:
    p = Path(path)
    if p.suffix in {".pt", ".pth"}:
        if torch is None:
            raise RuntimeError("torch is required to load .pt embeddings")
        emb = torch.load(p, map_location="cpu")
        if isinstance(emb, torch.Tensor):
            emb = emb.cpu().numpy()
        else:
            emb = np.asarray(emb)
    elif p.suffix in {".npy", ".npz"}:
        emb = np.load(p)
        if isinstance(emb, np.lib.npyio.NpzFile):
            emb = emb[emb.files[0]]
    else:
        raise ValueError(f"Unsupported embedding format: {p.suffix}")
    emb = np.asarray(emb, dtype=np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
    return emb / norms


def merge_nodes(graph: Dict, embeddings: np.ndarray, thr: float) -> Dict:
    edges = np.asarray(graph.get("edges", []), dtype=int)
    if edges.size == 0:
        return graph
    n = embeddings.shape[0]
    uf = UnionFind(n)
    for s, o in edges:
        if s >= n or o >= n:
            continue
        sim = float(np.dot(embeddings[s], embeddings[o]))
        if sim >= thr:
            uf.union(s, o)

    groups: Dict[int, List[int]] = {}
    for idx in range(n):
        root = uf.find(idx)
        groups.setdefault(root, []).append(idx)

    obj_pcls = graph["objects_pcl"]
    centers = graph["objects_center"]
    scales = graph.get("objects_scale")
    obj_ids = graph.get("objects_id", list(range(n)))
    obj_cats = graph.get("objects_cat")
    obj_nums = graph.get("objects_num")
    glob_pcls = graph.get("objects_pcl_glob")

    new_pcl: List[np.ndarray] = []
    new_center: List[List[float]] = []
    new_scale: List[float] = []
    new_id: List[int] = []
    new_cat: List[int] = [] if obj_cats is not None else []
    new_num: List[int] = [] if obj_nums is not None else []
    new_glob: List[np.ndarray] = [] if glob_pcls is not None else []

    idx_map: Dict[int, int] = {}
    for new_idx, members in enumerate(groups.values()):
        pts_global: List[np.ndarray] = []
        feats: List[np.ndarray] = []
        glob_parts: List[np.ndarray] = [] if glob_pcls is not None else []
        for m in members:
            pcl = np.asarray(obj_pcls[m])
            c = np.asarray(centers[m])
            s = np.asarray(scales[m]) if scales is not None else 1.0
            coords = pcl[:, :3]
            coords = coords * s + c
            pts_global.append(coords)
            if pcl.shape[1] > 3:
                feats.append(pcl[:, 3:])
            if glob_pcls is not None:
                glob_parts.append(np.asarray(glob_pcls[m]))
        pts = np.concatenate(pts_global, axis=0)
        if feats:
            extra = np.concatenate(feats, axis=0)
            combined = np.concatenate([pts, extra], axis=1)
        else:
            combined = pts
        centroid = np.mean(combined[:, :3], axis=0)
        offsets = combined[:, :3] - centroid
        scale = float(np.max(np.linalg.norm(offsets, axis=1)))
        norm_pts = combined.copy()
        if scale > 0:
            norm_pts[:, :3] = offsets / scale
        else:
            norm_pts[:, :3] = offsets
        new_pcl.append(norm_pts)
        new_center.append(centroid.tolist())
        new_scale.append(scale)
        new_id.append(new_idx)
        if obj_cats is not None:
            new_cat.append(obj_cats[members[0]])
        if obj_nums is not None:
            new_num.append(len(norm_pts))
        if glob_pcls is not None:
            glob = np.concatenate(glob_parts, axis=0)
            new_glob.append(glob)
        for m in members:
            idx_map[m] = new_idx

    # Rebuild relationships
    id_to_idx = {oid: i for i, oid in enumerate(obj_ids)}
    pairs = graph.get("pairs", [])
    triples = graph.get("triples", [])
    pred_cat = graph.get("predicate_cat", [])
    pred_num = graph.get("predicate_num", [])
    pred_pcl = graph.get("predicate_pcl_flag", [])
    pred_dist = graph.get("predicate_dist", [])
    pred_min = graph.get("predicate_min_dist")
    rel2frame = graph.get("rel2frame")

    new_pairs: List[List[int]] = []
    new_edges: List[List[int]] = []
    new_pred_cat: List = []
    new_pred_num: List = []
    new_pred_pcl: List = []
    new_pred_dist: List = []
    new_pred_min: List = [] if pred_min is not None else []
    new_rel2frame: Dict[Tuple[int, int], List] = {} if rel2frame is not None else {}
    seen_pairs: set[Tuple[int, int]] = set()

    for i, (a_id, b_id) in enumerate(pairs):
        ia = idx_map[id_to_idx[a_id]]
        ib = idx_map[id_to_idx[b_id]]
        if ia == ib:
            continue
        key = (ia, ib)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        new_pairs.append([new_id[ia], new_id[ib]])
        new_edges.append([ia, ib])
        if pred_cat:
            new_pred_cat.append(pred_cat[i])
        if pred_num:
            new_pred_num.append(pred_num[i])
        if pred_pcl:
            new_pred_pcl.append(pred_pcl[i])
        if pred_dist:
            new_pred_dist.append(pred_dist[i])
        if pred_min:
            new_pred_min.append(pred_min[i])
        if rel2frame is not None:
            orig_key = (a_id, b_id)
            frames = rel2frame.get(orig_key, [])
            new_rel2frame[(new_id[ia], new_id[ib])] = frames

    new_triples: List[List[int]] = []
    seen_triples: set[Tuple[int, int, int]] = set()
    for s_id, o_id, pred in triples:
        if s_id not in id_to_idx or o_id not in id_to_idx:
            continue
        ns = idx_map[id_to_idx[s_id]]
        no = idx_map[id_to_idx[o_id]]
        if ns == no:
            continue
        tkey = (ns, no, pred)
        if tkey in seen_triples:
            continue
        seen_triples.add(tkey)
        new_triples.append([new_id[ns], new_id[no], pred])

    graph["objects_pcl"] = [p.tolist() for p in new_pcl]
    graph["objects_center"] = new_center
    graph["objects_scale"] = new_scale
    graph["objects_id"] = new_id
    if obj_cats is not None:
        graph["objects_cat"] = new_cat
    if obj_nums is not None:
        graph["objects_num"] = new_num
    if new_glob:
        graph["objects_pcl_glob"] = [g.tolist() for g in new_glob]
    graph["objects_count"] = len(new_id)

    graph["pairs"] = new_pairs
    graph["edges"] = new_edges
    graph["triples"] = new_triples
    if pred_cat:
        graph["predicate_cat"] = new_pred_cat
    if pred_num:
        graph["predicate_num"] = new_pred_num
    if pred_pcl:
        graph["predicate_pcl_flag"] = new_pred_pcl
    if pred_dist:
        graph["predicate_dist"] = new_pred_dist
    if pred_min:
        graph["predicate_min_dist"] = new_pred_min
    graph["predicate_count"] = len(new_pairs)
    if rel2frame is not None:
        graph["rel2frame"] = new_rel2frame

    obj2frame = graph.get("object2frame")
    if obj2frame is not None:
        new_obj2frame: Dict[int, List] = {}
        for old_idx, frames in obj2frame.items():
            new_idx = idx_map[id_to_idx[int(old_idx)]]
            new_obj2frame.setdefault(new_id[new_idx], []).extend(frames)
        graph["object2frame"] = new_obj2frame

    return graph


def main() -> None:
    args = parse_args()
    graph = load_graph(args.graph)
    embeddings = load_embeddings(args.embeddings)
    merged = merge_nodes(graph, embeddings, args.threshold)

    import open3d as o3d

    inst_dir = Path(args.instances_out)
    inst_dir.mkdir(parents=True, exist_ok=True)
    palette = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.5, 0.5, 0.5],
            [1.0, 0.5, 0.0],
            [0.5, 0.0, 1.0],
            [0.0, 0.5, 1.0],
        ],
        dtype=np.float32,
    )

    points_all: List[np.ndarray] = []
    if "objects_pcl_glob" in merged and len(merged["objects_pcl_glob"]) > 0:
        obj_pcls = np.asarray(merged["objects_pcl_glob"], dtype=np.float32)
        for idx, obj in enumerate(obj_pcls):
            pts = np.asarray(obj, dtype=np.float32)
            color = palette[idx % len(palette)]
            rgb = np.tile(color, (pts.shape[0], 1))
            points_all.append(np.concatenate([pts[:, :3], rgb], axis=1))
            pcl = o3d.geometry.PointCloud()
            pcl.points = o3d.utility.Vector3dVector(pts[:, :3])
            if pts.shape[1] >= 6:
                pcl.colors = o3d.utility.Vector3dVector(pts[:, 3:6])
            else:
                pcl.colors = o3d.utility.Vector3dVector(rgb)
            o3d.io.write_point_cloud(str(inst_dir / f"instance_{idx:04d}.ply"), pcl)
    else:
        obj_pcls = np.asarray(merged["objects_pcl"], dtype=np.float32)
        centers = np.asarray(merged["objects_center"], dtype=np.float32)
        scales = merged.get("objects_scale")
        if scales is not None:
            scales = np.asarray(scales, dtype=np.float32)
        for idx, obj in enumerate(obj_pcls):
            pts = obj.copy()
            coords = pts[:, :3]
            if scales is not None:
                coords = coords * np.asarray(scales[idx]).reshape(1, -1)
            coords = coords + centers[idx]
            pts[:, :3] = coords
            color = palette[idx % len(palette)]
            rgb = np.tile(color, (coords.shape[0], 1))
            points_all.append(np.hstack([coords, rgb]))
            pcl = o3d.geometry.PointCloud()
            pcl.points = o3d.utility.Vector3dVector(pts[:, :3])
            if pts.shape[1] >= 6:
                pcl.colors = o3d.utility.Vector3dVector(pts[:, 3:6])
            else:
                pcl.colors = o3d.utility.Vector3dVector(rgb)
            o3d.io.write_point_cloud(str(inst_dir / f"instance_{idx:04d}.ply"), pcl)

    full_points = np.concatenate(points_all, axis=0)
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(full_points[:, :3])
    pcl.colors = o3d.utility.Vector3dVector(full_points[:, 3:6])
    o3d.io.write_point_cloud(str(args.pointcloud_out), pcl)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(merged, f)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
