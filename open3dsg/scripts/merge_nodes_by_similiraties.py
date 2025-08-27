""""
Merge graph nodes based on embedding similarity.

This utility loads a pickled scene graph and a tensor of node embeddings.
Existing edges are used to compute cosine similarities between connected
nodes. Nodes whose similarity exceeds a user defined threshold are merged
using a union--find data structure. During merging the point clouds are
aggregated in global coordinates and re-normalised. Relationship lists are
rebuilt by coalescing parallel edges (same ordered node pair), aggregating
side attributes, and averaging edge embeddings on the unit sphere.
The resulting graph is written back in the original pickle format. Optionally
the merged instance point clouds and the concatenated full point cloud can be
exported to ``.ply``/``.npz`` files.
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


def spherical_mean(vecs: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Unweighted spherical mean (normalize -> average -> re-normalize).
    Returns the unit vector mean; caller may compute resultant length if needed.
    """
    V = np.asarray(vecs, dtype=np.float32)
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + eps)
    M = V.sum(axis=0)
    n = np.linalg.norm(M) + eps
    return M / n


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge graph nodes whose embeddings are similar"
    )
    parser.add_argument("--graph", required=True, help="Path to pickled graph data")
    parser.add_argument(
        "--embedding-dir",
        required=True,
        help="Directory containing node and relation embedding tensors",
    )
    parser.add_argument(
        "--threshold", type=float, required=True, help="Cosine similarity threshold"
    )
    parser.add_argument("--out", required=True, help="Output path for merged graph")
    parser.add_argument(
        "--embedding-out",
        required=True,
        help="Directory to write merged embedding tensors",
    )
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

    input_paths = {
        Path(args.graph).resolve(),
        Path(args.embedding_dir).resolve(),
    }
    output_paths = [
        Path(args.out).resolve(),
        Path(args.instances_out).resolve(),
        Path(args.pointcloud_out).resolve(),
        Path(args.embedding_out).resolve(),
    ]
    if len(set(output_paths)) != len(output_paths) or any(p in input_paths for p in output_paths):
        parser.error("Output paths must be distinct from inputs and from each other")
    return args


def load_graph(path: str) -> Dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def load_embeddings(path: str) -> np.ndarray:
    """
    Loads embeddings from .pt/.pth/.npy/.npz.
    Handles Torch tensors saved in bfloat16 by upcasting to float32 before
    converting to NumPy (NumPy lacks native bfloat16 support).
    """
    p = Path(path)
    if p.suffix in {".pt", ".pth"}:
        if torch is None:
            raise RuntimeError("torch is required to load .pt embeddings")
        emb_t = torch.load(p, map_location="cpu")
        if isinstance(emb_t, torch.Tensor):
            if emb_t.dtype == torch.bfloat16:
                emb_t = emb_t.to(torch.float32)
            emb = emb_t.detach().cpu().numpy()
        else:
            emb = np.asarray(emb_t)
    elif p.suffix in {".npy", ".npz"}:
        emb = np.load(p)
        if isinstance(emb, np.lib.npyio.NpzFile):
            emb = emb[emb.files[0]]
    else:
        raise ValueError(f"Unsupported embedding format: {p.suffix}")
    emb = np.asarray(emb, dtype=np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
    return emb / norms


def merge_nodes(
    graph: Dict, embeddings: np.ndarray, thr: float
) -> Tuple[Dict, Dict[int, int]]:
    # Use ONLY existing edges as merge candidates (per your choice).
    edges = np.asarray(graph.get("edges", []), dtype=int)
    n = embeddings.shape[0]
    if edges.size == 0:
        return graph, {i: i for i in range(n)}
    uf = UnionFind(n)
    for s, o in edges:
        if s >= n or o >= n:
            continue
        sim = float(np.dot(embeddings[s], embeddings[o]))
        if sim >= thr:
            uf.union(s, o)

    # Build groups
    groups: Dict[int, List[int]] = {}
    for idx in range(n):
        root = uf.find(idx)
        groups.setdefault(root, []).append(idx)

    # Aggregate objects
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
    # Deterministic order over groups for reproducibility
    for new_idx, (root, members) in enumerate(sorted(groups.items(), key=lambda kv: kv[0])):
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

    # --- Rebuild relationships via BUCKETING and aggregation ---
    from collections import defaultdict

    id_to_idx = {oid: i for i, oid in enumerate(obj_ids)}
    pairs = graph.get("pairs", [])
    triples = graph.get("triples", [])
    pred_cat = graph.get("predicate_cat")
    pred_num = graph.get("predicate_num")
    pred_pcl = graph.get("predicate_pcl_flag")
    pred_dist = graph.get("predicate_dist")
    pred_min = graph.get("predicate_min_dist")
    rel2frame = graph.get("rel2frame")

    # Bucket duplicate ordered edges (ia, ib) -> list of original pair indices
    edge_buckets: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for i, (a_id, b_id) in enumerate(pairs):
        if a_id not in id_to_idx or b_id not in id_to_idx:
            continue
        ia = idx_map[id_to_idx[a_id]]
        ib = idx_map[id_to_idx[b_id]]
        if ia == ib:
            continue
        edge_buckets[(ia, ib)].append(i)

    # Aggregated containers
    new_pairs: List[List[int]] = []
    new_edges: List[List[int]] = []
    new_pred_cat: List = [] if pred_cat is not None else None
    new_pred_num: List = [] if pred_num is not None else None
    new_pred_pcl: List = [] if pred_pcl is not None else None
    new_pred_dist: List = [] if pred_dist is not None else None
    new_pred_min: List = [] if pred_min is not None else None
    new_rel2frame: Dict[Tuple[int, int], List] = {} if rel2frame is not None else None

    # Helper reducers (equal-weight)
    def majority_vote(int_list: List[int]) -> int:
        if not int_list:
            return 0
        arr = np.asarray(int_list, dtype=np.int64)
        return int(np.argmax(np.bincount(arr)))

    for (ia, ib), idxs in edge_buckets.items():
        # record edge once
        new_pairs.append([ia, ib])
        new_edges.append([ia, ib])

        if pred_cat is not None:
            cats = [pred_cat[i] for i in idxs]
            new_pred_cat.append(majority_vote(cats))

        if pred_num is not None:
            vals = [pred_num[i] for i in idxs]
            new_pred_num.append(int(np.round(np.mean(vals))))

        if pred_pcl is not None:
            vals = [pred_pcl[i] for i in idxs]
            new_pred_pcl.append(int(bool(np.any(vals))))

        if pred_dist is not None:
            vals = [pred_dist[i] for i in idxs]
            new_pred_dist.append(float(np.mean(vals)))

        if pred_min is not None:
            vals = [pred_min[i] for i in idxs]
            new_pred_min.append(float(np.min(vals)))

        if rel2frame is not None:
            frames = []
            for i_edge in idxs:
                a_id, b_id = pairs[i_edge]
                frames.extend(rel2frame.get((a_id, b_id), []))
            if new_rel2frame is not None:
                new_rel2frame[(ia, ib)] = frames

    # Triples: keep unique (subject, object, predicate) after remapping
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
        new_triples.append([ns, no, pred])

    # Write back aggregated graph
    graph["objects_pcl"] = [p.tolist() for p in new_pcl]
    graph["objects_center"] = new_center
    graph["objects_scale"] = new_scale
    graph["objects_id"] = list(range(len(new_id)))  # stable 0..N-1
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

    if new_pred_cat is not None:
        graph["predicate_cat"] = new_pred_cat
    if new_pred_num is not None:
        graph["predicate_num"] = new_pred_num
    if new_pred_pcl is not None:
        graph["predicate_pcl_flag"] = new_pred_pcl
    if new_pred_dist is not None:
        graph["predicate_dist"] = new_pred_dist
    if new_pred_min is not None:
        graph["predicate_min_dist"] = new_pred_min

    graph["predicate_count"] = len(new_pairs)

    if new_rel2frame is not None:
        graph["rel2frame"] = new_rel2frame  # keys in new-id space (ia, ib)

    obj2frame = graph.get("object2frame")
    if obj2frame is not None:
        new_obj2frame: Dict[int, List] = {}
        for old_idx, frames in obj2frame.items():
            new_idx = idx_map[id_to_idx[int(old_idx)]]
            new_obj2frame.setdefault(new_idx, []).extend(frames)
        graph["object2frame"] = new_obj2frame

    return graph, idx_map


def main() -> None:
    args = parse_args()

    # Option B: derive scan_id from the PARENT folder name (e.g., "scan0")
    scan_id = Path(args.graph).parent.name

    obj_path = Path(
        args.embedding_dir, "export_obj_clip_emb_clip_OpenSeg", f"{scan_id}.pt"
    )
    valid_path = Path(
        args.embedding_dir, "export_obj_clip_valids", f"{scan_id}.pt"
    )
    rel_path = Path(
        args.embedding_dir, "export_rel_clip_emb_clip_BLIP", f"{scan_id}.pt"
    )

    graph = load_graph(args.graph)

    # ---- Load node embeddings (handles bf16 in loader) ----
    obj_emb = load_embeddings(str(obj_path))
    if torch is None:
        raise RuntimeError("torch is required to load .pt embeddings")

    # ---- Load valids / relation embeddings with bf16-safe casts ----
    valids_t = torch.load(valid_path, map_location="cpu")
    rel_emb_t = torch.load(rel_path, map_location="cpu")

    if isinstance(valids_t, torch.Tensor):
        # Typically bool/uint8; just move to CPU NumPy
        valids = valids_t.detach().cpu().numpy()
    else:
        valids = np.asarray(valids_t)

    if isinstance(rel_emb_t, torch.Tensor):
        # Upcast from bfloat16 if needed before NumPy conversion
        rel_emb = rel_emb_t.to(torch.float32).cpu().numpy()
    else:
        rel_emb = np.asarray(rel_emb_t, dtype=np.float32)

    pairs_orig = list(graph.get("pairs", []))
    obj_ids_orig = graph.get("objects_id", list(range(obj_emb.shape[0])))

    # Merge nodes using existing edges + CLIP threshold
    merged, idx_map = merge_nodes(graph, obj_emb, args.threshold)

    from collections import defaultdict

    # ---- Node embeddings: spherical mean over members (unweighted) ----
    emb_groups: Dict[int, List[np.ndarray]] = defaultdict(list)
    valid_groups: Dict[int, List[np.ndarray]] = defaultdict(list)
    for old_idx, new_idx in idx_map.items():
        emb_groups[new_idx].append(obj_emb[old_idx])
        valid_groups[new_idx].append(valids[old_idx])

    if emb_groups:
        new_obj_emb = np.stack(
            [spherical_mean(np.stack(emb_groups[i], axis=0)) for i in range(len(emb_groups))]
        )
        new_valids = np.array(
            [np.any(valid_groups[i]) for i in range(len(valid_groups))]
        )
    else:
        new_obj_emb = obj_emb
        new_valids = valids

    # ---- Relation embeddings: bucket duplicates and spherical-mean per (ia, ib) ----
    rel_agg: Dict[Tuple[int, int], List[np.ndarray]] = defaultdict(list)
    id_to_idx = {oid: i for i, oid in enumerate(obj_ids_orig)}
    for i, (a_id, b_id) in enumerate(pairs_orig):
        ia = idx_map[id_to_idx[a_id]]
        ib = idx_map[id_to_idx[b_id]]
        if ia == ib:
            continue
        rel_agg[(ia, ib)].append(rel_emb[i])

    if merged.get("edges"):
        new_rel_emb = []
        for ia, ib in merged["edges"]:
            vecs = np.stack(rel_agg[(ia, ib)], axis=0)
            new_rel_emb.append(spherical_mean(vecs))
        new_rel_emb = np.stack(new_rel_emb, axis=0)
    else:
        new_rel_emb = np.zeros((0,) + rel_emb.shape[1:], dtype=rel_emb.dtype)

    # ---- Write merged embeddings ----
    obj_out_dir = Path(args.embedding_out, "export_obj_clip_emb_clip_OpenSeg")
    obj_out_dir.mkdir(parents=True, exist_ok=True)
    valid_out_dir = Path(args.embedding_out, "export_obj_clip_valids")
    valid_out_dir.mkdir(parents=True, exist_ok=True)
    rel_out_dir = Path(args.embedding_out, "export_rel_clip_emb_clip_BLIP")
    rel_out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(torch.as_tensor(new_obj_emb), obj_out_dir / f"{scan_id}.pt")
    torch.save(torch.as_tensor(new_valids), valid_out_dir / f"{scan_id}.pt")
    torch.save(torch.as_tensor(new_rel_emb), rel_out_dir / f"{scan_id}.pt")

    # ---- Visualization/exports ----
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
