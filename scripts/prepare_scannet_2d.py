#!/usr/bin/env python3
"""Organize ScanNet frames into the ScanNet 2D layout.

For every ``scan*/`` directory the script creates a ``scene<id>`` folder
under the output root with ``color``, ``depth``, ``pose`` and ``intrinsic``
subdirectories. Color images, depth maps and metadata are copied or
symlinked into their respective locations. Camera poses are written as
text matrices while intrinsics are stored either as a single
``intrinsic_color.txt`` or per-frame ``<idx>.txt`` files when they vary.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


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
        meta = json.loads(meta_file.read_text())

        img = meta_file.parent / f"image_{idx}.jpg"
        if img.exists():
            _link_or_copy(img, scene_dir / "color" / f"{idx}.jpg", symlink)

        depth = meta_file.parent / f"depth_{idx}.png"
        if depth.exists():
            _link_or_copy(depth, scene_dir / "depth" / f"{idx}.png", symlink)

        pose = meta.get("camera_pose")
        if pose is not None:
            _write_matrix(scene_dir / "pose" / f"{idx}.txt", pose)

        intr = meta.get("camera_intrinsics")
        if intr is not None:
            intrinsics[idx] = intr

    if intrinsics:
        first = next(iter(intrinsics.values()))
        if all(intr == first for intr in intrinsics.values()):
            _write_matrix(scene_dir / "intrinsic" / "intrinsic_color.txt", first)
        else:
            for idx, intr in intrinsics.items():
                _write_matrix(scene_dir / "intrinsic" / f"{idx}.txt", intr)


def main() -> None:
    ap = argparse.ArgumentParser(description="Organize ScanNet 2D dataset")
    ap.add_argument("dataset", type=Path, help="root directory containing scan* folders")
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

    for scan in sorted(args.dataset.glob("scan*")):
        _process_scan(scan, args.out, args.symlink)


if __name__ == "__main__":
    main()
