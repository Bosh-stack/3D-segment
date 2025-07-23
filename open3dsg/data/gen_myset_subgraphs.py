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


def build_graph_for_scan(scan_dir: Path, inst_dir_name: str = "mask/vis_instances"):
    inst_files = sorted(Path(scan_dir, inst_dir_name).glob("inst_*.ply"))
    nodes = []
    centers = []
    for i, f in enumerate(inst_files):
        pts = load_ply_points(f)
        aabb = compute_aabb(pts)
        centroid = pts.mean(axis=0).tolist()
        nodes.append({
            "inst_id": i,
            "file": str(f.relative_to(scan_dir)),
            "centroid": centroid,
            "aabb": aabb,
            "n_points": int(pts.shape[0])
        })
        centers.append(np.array(centroid))

    centers = np.stack(centers) if len(centers) else np.zeros((0, 3))
    edges = []
    if len(nodes) > 1:
        diags = [np.linalg.norm(np.array(n["aabb"][1]) - np.array(n["aabb"][0])) for n in nodes]
        mean_diag = float(np.mean(diags)) if diags else 1.0
        thresh = 1.5 * mean_diag

        pairs = []
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                d = np.linalg.norm(centers[i] - centers[j])
                if d < thresh:
                    pairs.append((d, i, j))

        pairs.sort(key=lambda x: x[0])
        deg = {i: 0 for i in range(len(nodes))}
        for _, i, j in pairs:
            if deg[i] < 10 and deg[j] < 10:
                edges.append([i, j, "spatial"])
                deg[i] += 1
                deg[j] += 1

    return {"nodes": nodes, "edges": edges}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    root = Path(args.root)
    scans = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("scan")])

    graphs = []
    for scan in scans:
        scan_id = scan.name
        graphs.append({
            "scan": scan_id,
            "split": 0,
            "graph": build_graph_for_scan(scan)
        })

    out_dir = Path(args.out) / "graphs"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.split}.json").write_text(json.dumps({"scans": graphs}))
    print(f"Wrote {len(graphs)} graphs to {out_dir}")


if __name__ == "__main__":
    main()
