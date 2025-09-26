import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


def load_ply_points(path: Path):
    pc = o3d.io.read_point_cloud(str(path))
    return np.asarray(pc.points)


def compute_aabb(points: np.ndarray):
    mn = points.min(axis=0)
    mx = points.max(axis=0)
    return [mn.tolist(), mx.tolist()]


def build_graph_for_scan(
    scan_dir: Path,
    inst_dir_name: str = "mask/vis_instances",
    max_edges_per_node: int = 10,
    kd_voxel_size: float = 0.0,
):
    inst_files = sorted(Path(scan_dir, inst_dir_name).glob("inst_*.ply"))
    nodes = []
    kd_point_sets = []
    kdtrees = []
    for i, f in enumerate(inst_files):
        pts_full = load_ply_points(f)
        aabb = compute_aabb(pts_full)
        centroid = pts_full.mean(axis=0).tolist()
        nodes.append({
            "inst_id": i,
            "file": str(f.relative_to(scan_dir)),
            "centroid": centroid,
            "aabb": aabb,
            "n_points": int(pts_full.shape[0])
        })
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pts_full)
        if kd_voxel_size > 0:
            pc = pc.voxel_down_sample(voxel_size=kd_voxel_size)
        kd_pts = np.asarray(pc.points)
        if kd_pts.size == 0:
            kd_pts = pts_full
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(kd_pts)
        kd_point_sets.append(kd_pts)
        kdtrees.append(o3d.geometry.KDTreeFlann(pc))

    edges = []
    if len(nodes) > 1:
        def min_distance(idx_a: int, idx_b: int):
            pts_a = kd_point_sets[idx_a]
            pts_b = kd_point_sets[idx_b]
            if pts_a.size == 0 or pts_b.size == 0:
                return float("inf")
            if pts_a.shape[0] <= pts_b.shape[0]:
                base_idx, query_pts = idx_a, pts_b
            else:
                base_idx, query_pts = idx_b, pts_a
            kd_tree = kdtrees[base_idx]
            min_sq = float("inf")
            for pt in query_pts:
                _, _, sq_dists = kd_tree.search_knn_vector_3d(pt, 1)
                if sq_dists:
                    sq = sq_dists[0]
                    if sq < min_sq:
                        min_sq = sq
            return float(np.sqrt(min_sq)) if np.isfinite(min_sq) else float("inf")

        n_nodes = len(nodes)
        centroids = np.asarray([node["centroid"] for node in nodes], dtype=np.float64)
        distance_cache = {}
        added_edges = set()
        for i in range(n_nodes):
            centroid_dists = np.linalg.norm(centroids - centroids[i], axis=1)
            centroid_dists[i] = np.inf
            oversample = max(max_edges_per_node * 4, max_edges_per_node)
            candidate_count = min(n_nodes - 1, oversample)
            if candidate_count <= 0:
                continue
            if candidate_count < centroid_dists.size:
                partition_idx = candidate_count - 1
                candidate_indices = np.argpartition(centroid_dists, partition_idx)[:candidate_count]
            else:
                candidate_indices = np.argsort(centroid_dists)

            neighbor_dists = []
            for j in candidate_indices:
                if i == j:
                    continue
                key = (min(i, j), max(i, j))
                if key not in distance_cache:
                    distance_cache[key] = min_distance(*key)
                dist = distance_cache[key]
                neighbor_dists.append((dist, j))

            neighbor_dists.sort(key=lambda x: x[0])
            for dist, j in neighbor_dists[:max_edges_per_node]:
                if not np.isfinite(dist):
                    continue
                key = (min(i, j), max(i, j))
                if key in added_edges:
                    continue
                edges.append([key[0], key[1], "spatial"])
                added_edges.add(key)

    return {"nodes": nodes, "edges": edges}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--max_edges_per_node", type=int, default=10)
    ap.add_argument(
        "--kd_voxel_size",
        type=float,
        default=0.0,
        help=(
            "Voxel size used to downsample instance point clouds before building the "
            "KD-tree (0 disables downsampling)."
        ),
    )
    args = ap.parse_args()

    root = Path(args.root)
    scans = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("scan")])

    graphs = []
    for scan in scans:
        scan_id = scan.name
        graphs.append({
            "scan": scan_id,
            "split": 0,
            "graph": build_graph_for_scan(
                scan,
                max_edges_per_node=args.max_edges_per_node,
                kd_voxel_size=args.kd_voxel_size,
            )
        })

    out_dir = Path(args.out) / "graphs"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.split}.json").write_text(json.dumps({"scans": graphs}))
    print(f"Wrote {len(graphs)} graphs to {out_dir}")


if __name__ == "__main__":
    main()
