"""Reconstruct a full point cloud from a preprocessed scene graph.

This utility merges per-object point sets stored in the pickled graph
dictionaries created by :mod:`open3dsg.data.preprocess_scannet` or
:mod:`open3dsg.data.preprocess_3rscan` into a single point cloud.

Example
-------
```bash
python open3dsg/scripts/reconstruct_pointcloud.py \
    --graph path/to/data_dict.pkl \
    --out scene.ply
```
The output format is inferred from the file extension and can be ``.ply``
or ``.npz``.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct a point cloud from a pickled graph dictionary",
    )
    parser.add_argument(
        "--graph",
        required=True,
        help="Path to pickled graph data produced by preprocessing",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output file (.ply or .npz) for the reconstructed point cloud",
    )
    return parser.parse_args()


def load_graph(graph_path: str) -> dict:
    with open(graph_path, "rb") as f:
        return pickle.load(f)


def reconstruct_points(graph: dict) -> np.ndarray:
    """Return concatenated point coordinates from ``graph``.

    If ``objects_pcl_glob`` is present, those coordinates are used directly.
    Otherwise ``objects_pcl`` are denormalised using ``objects_center`` and
    optionally ``objects_scale``.
    """

    if "objects_pcl_glob" in graph and len(graph["objects_pcl_glob"]) > 0:
        obj_pcls = np.asarray(graph["objects_pcl_glob"], dtype=np.float32)
        points = [p[:, :3] for p in obj_pcls]
    else:
        obj_pcls = np.asarray(graph["objects_pcl"], dtype=np.float32)
        centers = np.asarray(graph["objects_center"], dtype=np.float32)
        scales = graph.get("objects_scale")
        if scales is not None:
            scales = np.asarray(scales, dtype=np.float32)
        points = []
        for idx, obj in enumerate(obj_pcls):
            pts = obj[:, :3]
            if scales is not None:
                s = np.asarray(scales[idx]).reshape(1, -1)
                pts = pts * s
            pts = pts + centers[idx]
            points.append(pts)
    return np.concatenate(points, axis=0)


def save_points(points: np.ndarray, out_file: str) -> None:
    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".ply":
        import open3d as o3d

        pcl = o3d.geometry.PointCloud()
        pcl.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(str(out_path), pcl)
    else:
        np.savez_compressed(out_path, points=points)


def main() -> None:
    args = parse_args()
    graph = load_graph(args.graph)
    points = reconstruct_points(graph)
    save_points(points, args.out)


if __name__ == "__main__":
    main()
