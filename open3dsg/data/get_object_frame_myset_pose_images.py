#!/usr/bin/env python3
"""MySet pose-image frame association utility.

This script mirrors :mod:`get_object_frame_myset` but reads images from a
``pose_images`` directory found in each scan.  For every instance point cloud,
we determine the ``top_k`` frames where it is most visible and store the
results in ``<scan_id>_object2frame.pkl`` with the same structure as the
original script.  Each entry of the pickle file is
``{idx: [(frame_id, pixels, ratio, bbox, pixel_ids)], "names": {idx: inst_name}}`` where ``frame_id`` refers to the pose-image file name.

Example
-------
Run the script from the repository root::

    python open3dsg/data/get_object_frame_myset_pose_images.py \
        --root /data/Open3DSG_trainset \
        --out  output/preprocessed/myset/pose_frames \
        --top_k 5
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import List

import numpy as np
import open3d as o3d

from .get_object_frame_myset import load_cam, projection_details


def gather_pose_images(scan: Path) -> List[Path]:
    """Collect pose-image files for a scan.

    The function first searches for a ``pose_images`` subdirectory.  If it
    doesn't exist, it falls back to files in the scan directory whose name
    starts with ``pose_image``.  Both ``.png`` and ``.jpg`` extensions are
    supported.
    """

    pose_dir = scan / "pose_images"
    patterns = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG"]
    img_files: List[Path] = []
    if pose_dir.is_dir():
        for ptn in patterns:
            img_files.extend(sorted(pose_dir.glob(ptn)))
    else:
        for ptn in patterns:
            img_files.extend(sorted(scan.glob(f"pose_image_{ptn}")))
    return sorted(img_files)


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dataset root")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--top_k", type=int, default=5)
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    scans = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("scan")])

    for scan in scans:
        scan_id = scan.name
        inst_paths = sorted((scan / "mask/vis_instances").glob("inst_*.ply"))
        inst_pts = [np.asarray(o3d.io.read_point_cloud(str(p)).points) for p in inst_paths]

        img_files = gather_pose_images(scan)
        object2frame = {}
        name_map = {}

        for inst_idx, pts in enumerate(inst_pts):
            scores = []
            details = {}
            for img_path in img_files:
                try:
                    idx = int(img_path.stem.split("_")[-1])
                except Exception:
                    idx = img_files.index(img_path)
                meta_candidates = [
                    img_path.with_name(f"im_metadata_{idx}.json"),
                    scan / f"im_metadata_{idx}.json",
                    img_path.with_suffix(".json"),
                ]
                meta = next((m for m in meta_candidates if m.exists()), None)
                if meta is None:
                    continue
                try:
                    K, T, w, h = load_cam(meta)
                except Exception as e:
                    print(f"[WARN] {scan_id} frame {idx}: {e}")
                    continue
                vis, pix_cnt, bbox, pix_ids = projection_details(pts, K, T, w, h)
                scale_x = 320.0 / float(w)
                scale_y = 240.0 / float(h)
                bbox = (
                    int(np.clip(round(bbox[0] * scale_x), 0, 319)),
                    int(np.clip(round(bbox[1] * scale_y), 0, 239)),
                    int(np.clip(round(bbox[2] * scale_x), 0, 319)),
                    int(np.clip(round(bbox[3] * scale_y), 0, 239)),
                )
                if pix_ids.size:
                    pix_ids = np.stack(
                        (
                            np.clip(np.round(pix_ids[:, 1] * scale_y), 0, 239),
                            np.clip(np.round(pix_ids[:, 0] * scale_x), 0, 319),
                        ),
                        axis=1,
                    ).astype(np.uint16)
                scores.append((idx, vis))
                details[idx] = (img_path.name, pix_cnt, vis, bbox, pix_ids)
            top = [i for i, _ in sorted(scores, key=lambda x: -x[1])[: args.top_k]]
            object2frame[int(inst_idx)] = [details[i] for i in top if i in details]
            name_map[int(inst_idx)] = inst_paths[inst_idx].stem

        data = {"names": name_map, **object2frame}
        with open(out_dir / f"{scan_id}_object2frame.pkl", "wb") as fw:
            pickle.dump(data, fw)
        print(f"{scan_id}: {len(inst_paths)} instances processed")


if __name__ == "__main__":
    main()
