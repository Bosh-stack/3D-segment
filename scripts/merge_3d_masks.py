#!/usr/bin/env python3
"""Merge two 3D masks based on 2D overlap with score comparison.

This utility expects the dataset to follow the ``Open3DSG_trainset`` layout.
Two usage modes are supported:

* **Instance crops**: provide the file names of two ``mask/vis_instances``
  ``.ply`` files.  The numeric suffix of the file name is interpreted as a
  confidence score (e.g. ``inst_001_2.03.ply`` has a score of ``2.03``).
* **Full scene mask**: provide a scene-wide ``vis_pred.ply`` via ``--scene``,
  and specify two integer instance ids via ``--inst-a`` and ``--inst-b``.  In
  this mode every point with matching instance id is extracted from the scene
  mask.  No confidence scores are available so both masks are treated equally.

For every pair of overlapping pixels in an RGB frame the point from the mask
with the *lower* score is dropped.  If ``--image-idx`` is not supplied the
script picks the frame where the two masks have the highest combined visibility.
The remaining points are written to the output ``.ply`` file.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import open3d as o3d

from open3dsg.data.get_object_frame_myset import load_cam


LOG_PATH = Path(__file__).resolve().parent / "visibility_log.json"


def _load_log():
    if LOG_PATH.exists():
        return json.loads(LOG_PATH.read_text())
    return {}


def _save_log(cache):
    LOG_PATH.write_text(json.dumps(cache, indent=2))


def _project(points: np.ndarray, K: np.ndarray, T: np.ndarray):
    """Project ``points`` (Nx3) using intrinsics ``K`` and extrinsics ``T``."""
    pts_h = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    cam = (np.linalg.inv(T) @ pts_h.T).T[:, :3]
    in_front = cam[:, 2] > 0
    cam = cam[in_front]
    pix = (K @ cam.T).T
    pix = pix[:, :2] / cam[:, 2:3]
    return pix, in_front


def _parse_score(path: Path) -> float:
    match = re.search(r"_(\d+\.?\d*)\.ply$", path.name)
    return float(match.group(1)) if match else 0.0


def _load_instance_from_scene(scene: o3d.geometry.PointCloud, inst_id: int):
    """Extract points matching ``inst_id`` from ``vis_pred.ply``."""
    colors = (np.asarray(scene.colors) * 255).astype(np.int32)
    ids = colors[:, 0] * 65536 + colors[:, 1] * 256 + colors[:, 2]
    mask = ids == inst_id
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(np.asarray(scene.points)[mask])
    pc.colors = o3d.utility.Vector3dVector(np.asarray(scene.colors)[mask])
    return pc


def merge_pcs(pc_a, pc_b, score_a, score_b, K, T, width, height):
    """Merge two point clouds based on 2D pixel overlap."""
    uv_a, mask_a = _project(np.asarray(pc_a.points), K, T)
    uv_b, mask_b = _project(np.asarray(pc_b.points), K, T)

    keep_a = np.ones(mask_a.sum(), dtype=bool)
    keep_b = np.ones(mask_b.sum(), dtype=bool)

    coords_a = {tuple(p.astype(int)): i for i, p in enumerate(uv_a)}
    for j, p in enumerate(uv_b):
        key = tuple(p.astype(int))
        if 0 <= key[0] < width and 0 <= key[1] < height and key in coords_a:
            if score_a >= score_b:
                keep_b[j] = False
            else:
                keep_a[coords_a[key]] = False

    pts = np.vstack(
        [
            np.asarray(pc_a.points)[mask_a][keep_a],
            np.asarray(pc_b.points)[mask_b][keep_b],
        ]
    )
    cols = None
    if pc_a.has_colors() or pc_b.has_colors():
        cols = np.vstack(
            [
                np.asarray(pc_a.colors)[mask_a][keep_a]
                if pc_a.has_colors()
                else np.zeros((keep_a.sum(), 3)),
                np.asarray(pc_b.colors)[mask_b][keep_b]
                if pc_b.has_colors()
                else np.zeros((keep_b.sum(), 3)),
            ]
        )

    merged = o3d.geometry.PointCloud()
    merged.points = o3d.utility.Vector3dVector(pts)
    if cols is not None:
        merged.colors = o3d.utility.Vector3dVector(cols)
    return merged


def _visible_count(pc: o3d.geometry.PointCloud, K, T, w, h):
    uv, mask = _project(np.asarray(pc.points), K, T)
    if uv.size == 0:
        return 0
    x_valid = (uv[:, 0] >= 0) & (uv[:, 0] < w)
    y_valid = (uv[:, 1] >= 0) & (uv[:, 1] < h)
    return int((x_valid & y_valid).sum())


def _cache_key(args) -> str:
    if args.inst:
        a, b = args.inst
        return f"{args.scan_dir.resolve()}|{a}|{b}"
    return f"{args.scan_dir.resolve()}|{args.scene}|{args.inst_a}|{args.inst_b}"


def main():
    p = argparse.ArgumentParser(description="Merge two 3D masks")
    p.add_argument("scan_dir", type=Path, help="path to scan directory")
    p.add_argument("output", type=Path, help="merged point cloud output")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--inst", nargs=2, metavar=("PLY_A", "PLY_B"), help="instance crop files")
    src.add_argument("--scene", type=Path, help="path to scene-wide vis_pred.ply")
    p.add_argument("--inst-a", type=int, help="instance id A when using --scene")
    p.add_argument("--inst-b", type=int, help="instance id B when using --scene")
    p.add_argument(
        "--image-idx",
        type=int,
        help="RGB frame to use; if omitted the best visible frame is selected",
    )
    p.add_argument(
        "--max-dist",
        type=float,
        default=1.0,
        help="maximum centroid-to-centroid distance allowed for merging",
    )
    args = p.parse_args()

    if args.inst:
        mask_dir = args.scan_dir / "mask" / "vis_instances"
        pc_a = o3d.io.read_point_cloud(mask_dir / args.inst[0])
        pc_b = o3d.io.read_point_cloud(mask_dir / args.inst[1])
        score_a = _parse_score(Path(args.inst[0]))
        score_b = _parse_score(Path(args.inst[1]))
    else:
        if args.inst_a is None or args.inst_b is None:
            raise ValueError("--inst-a and --inst-b are required with --scene")
        scene_pc = o3d.io.read_point_cloud(args.scan_dir / args.scene)
        pc_a = _load_instance_from_scene(scene_pc, args.inst_a)
        pc_b = _load_instance_from_scene(scene_pc, args.inst_b)
        score_a = score_b = 0.0

    center_a = np.asarray(pc_a.get_center())
    center_b = np.asarray(pc_b.get_center())
    dist = np.linalg.norm(center_a - center_b)
    if dist > args.max_dist:
        print(
            f"Skipping merge; pair distance {dist:.2f} exceeds threshold {args.max_dist:.2f}"
        )
        return

    if args.image_idx is not None:
        meta = args.scan_dir / f"im_metadata_{args.image_idx}.json"
        K, T, w, h = load_cam(meta)
    else:
        cache = _load_log()
        key = _cache_key(args)
        if key in cache:
            meta = Path(cache[key]["meta"])
            best_score = cache[key]["score"]
            print(
                f"Loaded frame {meta} with {best_score} visible points from log"
            )
            K, T, w, h = load_cam(meta)
        else:
            best_meta = None
            best_score = -1
            for meta in sorted(args.scan_dir.glob("im_metadata_*.json")):
                K, T, w, h = load_cam(meta)
                score = _visible_count(pc_a, K, T, w, h) + _visible_count(
                    pc_b, K, T, w, h
                )
                if score > best_score:
                    best_meta = meta
                    best_score = score
                    best_cam = (K, T, w, h)
            if best_meta is None:
                raise RuntimeError("No camera metadata files found")
            print(
                f"Selected frame {best_meta} with {best_score} visible points"
            )
            K, T, w, h = best_cam
            cache[key] = {"meta": str(best_meta), "score": best_score}
            _save_log(cache)

    merged = merge_pcs(pc_a, pc_b, score_a, score_b, K, T, w, h)
    o3d.io.write_point_cloud(str(args.output), merged)
    print(f"Merged point cloud written to {args.output}")


if __name__ == "__main__":
    main()
