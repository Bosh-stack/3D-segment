#!/usr/bin/env python3
"""Annotate frame visibility for a single instance.

Given a scan directory and an instance ``.ply`` file, this script computes the
visibility ratio, pixel count and 2D bounding box for the instance in every
RGB frame.  Each frame is saved with a drawn bounding box and a ``frames.json``
file summarises the results ordered by visibility.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict

import cv2
import numpy as np
import open3d as o3d

from open3dsg.data.get_object_frame_myset import load_cam, projection_details

IMG_PATTERNS = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]


def gather_images(scan: Path) -> List[Path]:
    """Collect RGB images from ``images*`` subdirectories."""
    img_files: List[Path] = []
    for img_dir in scan.glob("images*"):
        if img_dir.is_dir():
            for ptn in IMG_PATTERNS:
                img_files.extend(sorted(img_dir.glob(ptn)))
    return sorted(img_files)


def find_metadata(scan: Path, img_path: Path, idx: int) -> Path | None:
    """Return the metadata file associated with ``img_path`` if present."""
    candidates = [
        img_path.with_name(f"im_metadata_{idx}.json"),
        scan / f"im_metadata_{idx}.json",
        img_path.with_suffix(".json"),
    ]
    for m in candidates:
        if m.exists():
            return m
    return None


def load_instance(ply_path: Path) -> np.ndarray:
    pc = o3d.io.read_point_cloud(str(ply_path))
    if len(pc.points) == 0:
        mesh = o3d.io.read_triangle_mesh(str(ply_path))
        pc = mesh.sample_points_uniformly(number_of_points=len(mesh.vertices))
    pts = np.asarray(pc.points)
    if pts.size == 0:
        raise RuntimeError("Instance contains no points")
    return pts


def annotate(img_path: Path, out_path: Path, bbox: tuple[int, int, int, int], vis: float, pix_cnt: int) -> None:
    img = cv2.imread(str(img_path))
    if img is None:
        return
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        img,
        f"{vis:.3f} ({pix_cnt})",
        (x1, max(0, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(out_path), img)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan_dir", required=True, help="Scan directory containing images and metadata")
    ap.add_argument("--instance", required=True, help="Instance point cloud (.ply)")
    ap.add_argument("--out_dir", required=True, help="Output directory for annotated images and frames.json")
    args = ap.parse_args()

    scan = Path(args.scan_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pts = load_instance(Path(args.instance))
    img_files = gather_images(scan)
    if not img_files:
        raise RuntimeError(f"No images found under {scan}")

    records: List[Dict[str, object]] = []

    for i, img_path in enumerate(img_files):
        try:
            idx = int(img_path.stem.split("_")[-1])
        except Exception:
            idx = i
        meta = find_metadata(scan, img_path, idx)
        if meta is None:
            continue
        try:
            K, T, w, h = load_cam(meta)
        except Exception as e:
            print(f"[WARN] {meta}: {e}")
            continue
        vis, pix_cnt, bbox, _ = projection_details(pts, K, T, w, h)
        records.append({
            "frame": img_path.name,
            "visibility": float(vis),
            "pixels": int(pix_cnt),
            "bbox": list(map(int, bbox)),
        })
        annotate(img_path, out_dir / img_path.name, bbox, vis, pix_cnt)

    records.sort(key=lambda r: r["visibility"], reverse=True)
    with open(out_dir / "frames.json", "w") as fw:
        json.dump(records, fw, indent=2)


if __name__ == "__main__":
    main()
