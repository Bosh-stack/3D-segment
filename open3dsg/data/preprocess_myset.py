import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from tqdm import tqdm


def o3d_load(p: Path):
    pc = o3d.io.read_point_cloud(str(p))
    pts = np.asarray(pc.points)
    if pc.has_colors():
        cols = np.asarray(pc.colors)
        arr = np.concatenate([pts, cols], axis=1)
    else:
        arr = pts
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--graphs", required=True, help="graphs/train.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.root)
    graphs = json.loads(Path(args.graphs).read_text())

    out_cache = Path(args.out) / "cache"
    out_cache.mkdir(parents=True, exist_ok=True)

    for scan_id, g in tqdm(graphs.items()):
        scan_dir = root / scan_id
        raw_candidates = list(scan_dir.glob(f"{scan_id}.*"))
        if not raw_candidates:
            raise FileNotFoundError(f"No raw cloud for {scan_id}")
        raw_cloud = raw_candidates[0]
        arr = o3d_load(raw_cloud)
        torch.save(torch.from_numpy(arr).float(), out_cache / f"{scan_id}_pc.pt")

        inst_meta = []
        for node in g["nodes"]:
            inst_meta.append({
                "inst_id": node["inst_id"],
                "file": node["file"],
                "centroid": node["centroid"],
                "aabb": node["aabb"],
                "n_points": node["n_points"],
            })
        with open(out_cache / f"{scan_id}_instances.pkl", "wb") as fw:
            pickle.dump(inst_meta, fw)

    print("Preprocessing finished.")


if __name__ == "__main__":
    main()
