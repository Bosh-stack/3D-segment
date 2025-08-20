#!/usr/bin/env python3
"""Organize ScanNet frames into the ScanNet 2D layout.

The script expects a dataset structure where every ``scan*/`` directory
contains ``images*/`` subfolders with ``image_<idx>.jpg``, optional
``depth_<idx>.png`` and ``im_metadata_<idx>.json`` files.  The metadata
must provide camera ``translation`` and quaternion ``rotation`` as well as
``focal``, ``pixel`` and ``principal`` blocks describing the camera
intrinsics.  For every scan the script creates a ``scene<id>`` folder under
the output root with ``color``, ``depth``, ``pose`` and ``intrinsic``
subdirectories.  Frames with missing data are skipped with a warning.  The
RGB images and depth maps are copied or symlinked, poses are written as
4×4 matrices and intrinsics are stored either once as
``intrinsic_color.txt`` or per-frame ``<idx>.txt`` files when they vary.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
import math


def _quat_to_rot(qw: float, qx: float, qy: float, qz: float):
    """Convert a quaternion to a 3×3 rotation matrix."""
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm == 0:
        raise ValueError("Zero-length quaternion")
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
    return [
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ]


def _write_matrix(path: Path, mat) -> None:
    """Write a 2D list or tuple to ``path`` as space-separated rows."""
    with path.open("w") as f:
        for row in mat:
            f.write(" ".join(map(str, row)) + "\n")


def _link_or_copy(src: Path, dst: Path, symlink: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if symlink:
        os.symlink(os.path.relpath(src, dst.parent), dst)
    else:
        shutil.copy2(src, dst)


def _process_scan(scan: Path, out_root: Path, symlink: bool) -> None:
    scan_id = scan.name.replace("scan", "")
    scene_dir = out_root / f"scene{scan_id}"
    for sub in ["color", "depth", "pose", "intrinsic"]:
        (scene_dir / sub).mkdir(parents=True, exist_ok=True)

    intrinsics = {}
    for meta_file in sorted(scan.glob("images*/im_metadata_*.json")):
        idx = meta_file.stem.split("_")[-1]
        try:
            meta = json.loads(meta_file.read_text())
        except json.JSONDecodeError as e:
            print(f"[WARN] {meta_file}: {e}; skipping")
            continue

        try:
            t = meta["translation"]
            r = meta["rotation"]
            tx, ty, tz = t["x"], t["y"], t["z"]
            qw, qx, qy, qz = (
                r["w"],
                r["x"],
                r["y"],
                r["z"],
            )
            f = meta["focal"]["focalLength"]
            pw = meta["pixel"]["pixelWidth"]
            ph = meta["pixel"]["pixelHeight"]
            cx = meta["principal"]["principalPointX"]
            cy = meta["principal"]["principalPointY"]
        except KeyError as e:
            print(f"[WARN] {meta_file}: missing {e.args[0]}; skipping frame")
            continue

        img = meta_file.parent / f"image_{idx}.jpg"
        if not img.exists():
            print(f"[WARN] {meta_file}: {img.name} not found; skipping frame")
            continue
        sub_dir = meta_file.parent.name  # e.g. "images10"

        # pose
        try:
            rot = _quat_to_rot(qw, qx, qy, qz)
        except Exception as e:
            print(f"[WARN] {meta_file}: {e}; skipping frame")
            continue
        pose = [
            [rot[0][0], rot[0][1], rot[0][2], tx],
            [rot[1][0], rot[1][1], rot[1][2], ty],
            [rot[2][0], rot[2][1], rot[2][2], tz],
            [0.0, 0.0, 0.0, 1.0],
        ]

        # intrinsics
        fx = float(f) / float(pw)
        fy = float(f) / float(ph)
        K = [[fx, 0.0, float(cx)], [0.0, fy, float(cy)], [0.0, 0.0, 1.0]]

        _link_or_copy(img, scene_dir / "color" / sub_dir / img.name, symlink)

        depth = meta_file.parent / f"depth_{idx}.png"
        if depth.exists():
            _link_or_copy(depth, scene_dir / "depth" / sub_dir / depth.name, symlink)

        pose_path = scene_dir / "pose" / sub_dir / f"{idx}.txt"
        pose_path.parent.mkdir(parents=True, exist_ok=True)
        _write_matrix(pose_path, pose)
        intrinsics[idx] = (K, sub_dir)

    if intrinsics:
        first_mat, _ = next(iter(intrinsics.values()))
        if all(intr[0] == first_mat for intr in intrinsics.values()):
            _write_matrix(scene_dir / "intrinsic" / "intrinsic_color.txt", first_mat)
        else:
            for idx, (intr, sub_dir) in intrinsics.items():
                intr_path = scene_dir / "intrinsic" / sub_dir / f"{idx}.txt"
                intr_path.parent.mkdir(parents=True, exist_ok=True)
                _write_matrix(intr_path, intr)


def main() -> None:
    ap = argparse.ArgumentParser(description="Organize ScanNet 2D dataset")
    ap.add_argument(
        "dataset_root", type=Path, help="root directory containing scan* folders"
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("SCANNET/scannet_2d"),
        help="output directory for scene folders",
    )
    ap.add_argument(
        "--symlink",
        action="store_true",
        help="use symbolic links instead of copying files",
    )
    args = ap.parse_args()

    for scan in sorted(args.dataset_root.glob("scan*")):
        _process_scan(scan, args.out, args.symlink)


if __name__ == "__main__":
    main()
