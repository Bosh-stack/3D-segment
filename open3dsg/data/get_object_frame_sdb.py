#!/usr/bin/env python3
"""Synthetic depth buffer frame association utility.

This script associates each 3D instance with the ``top_k`` RGB frames where the
object is the most visible.  It mirrors
:mod:`open3dsg.data.get_object_frame_myset` but generates a synthetic depth
buffer from the full scene point cloud for each frame.  The buffer is used with
``compute_mapping`` from :mod:`open3dsg.data.get_object_frame` to reason about
occlusion.

Example::

    python open3dsg/data/get_object_frame_sdb.py \
        --root /data/Open3DSG_trainset \
        --out  open3dsg/output/preprocessed/myset/frames \
        --top_k 5
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R


# -----------------------------
# Intrinsics / Extrinsics utils
# -----------------------------


def _intrinsics_from_focal_pixel_block(m: dict) -> Tuple[np.ndarray, int, int]:
    F = m.get("focal", {}).get("focalLength")
    px = m.get("pixel", {}).get("pixelWidth")
    py = m.get("pixel", {}).get("pixelHeight")
    w = (m.get("image", {}) or {}).get("imageWidth") or m.get("width") or m.get("W") or m.get("w")
    h = (m.get("image", {}) or {}).get("imageHeight") or m.get("height") or m.get("H") or m.get("h")
    if F is None or px is None or py is None:
        return None, None, None
    fx = float(F) / float(px)
    fy = float(F) / float(py)
    cx = m.get("principal", {}).get("principalPointX")
    cy = m.get("principal", {}).get("principalPointY")
    if cx is None or cy is None:
        if w is None or h is None:
            raise KeyError("Principal point missing and image size unknown to infer center")
        cx, cy = (float(w) - 1) / 2.0, (float(h) - 1) / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    return K, int(w), int(h)


def _intrinsics_from_fov(m: dict) -> Tuple[np.ndarray, int, int]:
    # Accept fovx/fovy in degrees. Need width/height to convert: fx = (w/2) / tan(fovx/2)
    fovx = m.get("fovx") or (m.get("fov", {}) or {}).get("x")
    fovy = m.get("fovy") or (m.get("fov", {}) or {}).get("y")
    w = (m.get("image", {}) or {}).get("imageWidth") or m.get("width") or m.get("W") or m.get("w")
    h = (m.get("image", {}) or {}).get("imageHeight") or m.get("height") or m.get("H") or m.get("h")
    cx = m.get("principal", {}).get("principalPointX") if "principal" in m else None
    cy = m.get("principal", {}).get("principalPointY") if "principal" in m else None
    if (fovx is None and fovy is None) or w is None or h is None:
        return None, None, None
    import math
    if fovx is not None:
        fx = (float(w) / 2.0) / math.tan(float(fovx) * math.pi / 360.0)
    else:
        fx = None
    if fovy is not None:
        fy = (float(h) / 2.0) / math.tan(float(fovy) * math.pi / 360.0)
    else:
        fy = None
    # If only one is provided, reuse it (assuming square pixels)
    if fx is None and fy is not None:
        fx = fy
    if fy is None and fx is not None:
        fy = fx
    if cx is None:
        cx = (float(w) - 1) / 2.0
    if cy is None:
        cy = (float(h) - 1) / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    return K, int(w), int(h)


def load_cam(meta_file: Path):
    """Load intrinsics and extrinsics from flexible JSON schemas.

    The extrinsics stored in ``meta_file`` are assumed to transform points from
    the camera coordinate system to the world coordinate system
    (``camera→world``).  This helper converts them into a ``world→camera``
    transform before returning the result.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, int, int]
        ``K`` (3x3), ``T_world_cam`` (4x4), image width and height.
    """
    m = json.loads(meta_file.read_text())

    # -------- intrinsics --------
    K = None
    w = h = None

    # Direct full matrix
    for key in ("K", "intrinsic", "intrinsic_matrix"):
        if key in m:
            K = np.array(m[key], dtype=np.float32).reshape(3, 3)
            break

    # fx/fy/cx/cy flat keys
    if K is None and all(k in m for k in ("fx", "fy", "cx", "cy")):
        fx, fy, cx, cy = map(float, (m["fx"], m["fy"], m["cx"], m["cy"]))
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        w = m.get("width") or m.get("W") or m.get("w") or m.get("img_width")
        h = m.get("height") or m.get("H") or m.get("h") or m.get("img_height")

    # focal/pixel/principal block
    if K is None:
        K, w, h = _intrinsics_from_focal_pixel_block(m)

    # FOV-based
    if K is None:
        K, w, h = _intrinsics_from_fov(m)

    if K is None:
        raise KeyError("Camera intrinsics not found in metadata")

    # image size if still missing
    if w is None or h is None:
        w = (m.get("image", {}) or {}).get("imageWidth") or m.get("width") or m.get("W") or m.get("w") or m.get("img_width")
        h = (m.get("image", {}) or {}).get("imageHeight") or m.get("height") or m.get("H") or m.get("h") or m.get("img_height")
        if w is None or h is None:
            raise KeyError("Image width/height not found in metadata")

    # -------- extrinsics --------
    T = None

    # flat keys (qw,qx,qy,qz,tx,ty,tz)
    if all(k in m for k in ("qw", "qx", "qy", "qz", "tx", "ty", "tz")):
        quat = [m["qw"], m["qx"], m["qy"], m["qz"]]
        rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
        trans = np.array([m["tx"], m["ty"], m["tz"]], dtype=np.float32)
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = rot
        T[:3, 3] = trans

    # nested dicts (rotation{x,y,z,w}, translation{x,y,z})
    if T is None and "rotation" in m and "translation" in m:
        r = m["rotation"]
        t = m["translation"]
        quat = [r.get("w"), r.get("x"), r.get("y"), r.get("z")]
        if None not in quat:
            rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
            trans = np.array([t.get("x"), t.get("y"), t.get("z")], dtype=np.float32)
            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = rot
            T[:3, 3] = trans

    # full matrix
    for key in ("pose", "extrinsic", "transform"):
        if T is None and key in m:
            T = np.array(m[key], dtype=np.float32).reshape(4, 4)

    if T is None:
        raise KeyError("Camera extrinsics not found in metadata")

    # Metadata typically provides a camera→world matrix; invert it so that the
    # returned transform maps world coordinates into the camera frame.
    T_world_cam = np.linalg.inv(T)

    return K, T_world_cam, int(w), int(h)


# -----------------------------
# Frame gathering
# -----------------------------


def gather_images(scan: Path) -> List[Path]:
    """Collect RGB frames from every ``images*`` directory in ``scan``."""
    patterns = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]
    img_files: List[Path] = []
    for img_dir in scan.glob("images*"):
        if img_dir.is_dir():
            for ptn in patterns:
                img_files.extend(sorted(img_dir.glob(ptn)))
    return sorted(img_files)


# -----------------------------
# Synthetic depth buffer helpers
# -----------------------------


def build_depth_buffer(world_to_camera: np.ndarray, points: np.ndarray, intrinsics: np.ndarray, image_dim: np.ndarray, cut_bound: int) -> np.ndarray:
    """Project ``points`` and keep the nearest depth per pixel."""
    if points.size == 0:
        return np.full((image_dim[1], image_dim[0]), np.inf, dtype=np.float32)
    coords = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1).T
    cam = world_to_camera @ coords
    x = (cam[0] * intrinsics[0, 0]) / (-cam[2]) + intrinsics[0, 2]
    y = (-cam[1] * intrinsics[1, 1]) / (-cam[2]) + intrinsics[1, 2]
    px = np.round(x).astype(int)
    py = np.round(y).astype(int)
    z = -cam[2]
    w, h = image_dim
    valid = (cam[2] < 0) & (px >= cut_bound) & (py >= cut_bound) & (px < w - cut_bound) & (py < h - cut_bound)
    depth = np.full((h, w), np.inf, dtype=np.float32)
    if np.any(valid):
        np.minimum.at(depth, (py[valid], px[valid]), z[valid])
    return depth


def compute_mapping(world_to_camera, coords, depth, intrinsic, cut_bound, vis_thres, image_dim):
    mapping = np.zeros((3, coords.shape[0]), dtype=int)
    coords_new = np.concatenate([coords, np.ones([coords.shape[0], 1])], axis=1).T
    assert coords_new.shape[0] == 4, "[!] Shape error"

    p = np.matmul(world_to_camera, coords_new)
    # Camera coordinates follow a y-up convention while image rows grow
    # downwards.  Negate the y component before applying the intrinsics so
    # that positive camera ``y`` maps to smaller row indices.
    p[0] = (p[0] * intrinsic[0][0]) / (-p[2]) + intrinsic[0][2]
    p[1] = (-p[1] * intrinsic[1][1]) / (-p[2]) + intrinsic[1][2]
    pi = np.round(p).astype(int)  # simply round the projected coordinates
    inside_mask = (
        (p[2] < 0)
        & (pi[0] >= cut_bound)
        & (pi[1] >= cut_bound)
        & (pi[0] < image_dim[0] - cut_bound)
        & (pi[1] < image_dim[1] - cut_bound)
    )
    depth_cur = depth[pi[1][inside_mask], pi[0][inside_mask]]
    occlusion_mask = np.zeros_like(inside_mask)
    occlusion_mask[inside_mask] = (
        np.abs(depth_cur - (-p[2][inside_mask])) <= vis_thres * depth_cur
    )

    inside_mask &= occlusion_mask
    mapping[0][inside_mask] = pi[1][inside_mask]
    mapping[1][inside_mask] = pi[0][inside_mask]
    mapping[2][inside_mask] = 1

    return mapping


def image_3d_mapping(scan: Path, inst_paths: List[Path], inst_pts: List[np.ndarray], img_files: List[Path], top_k: int, boarder_pixels: int = 0, vis_tresh: float = 0.05):
    """Associate instances with frames using a synthetic depth buffer."""
    point_cloud = np.concatenate(inst_pts, axis=0) if inst_pts else np.empty((0, 3), dtype=np.float32)
    scores = {idx: [] for idx in range(len(inst_pts))}
    details = {idx: {} for idx in range(len(inst_pts))}

    for img_path in img_files:
        try:
            idx = int(img_path.stem.split("_")[-1])
        except Exception:
            idx = img_files.index(img_path)
        meta = img_path.parent / f"im_metadata_{idx}.json"
        if not meta.exists():
            continue
        try:
            K, T_world_cam, w, h = load_cam(meta)
        except Exception as e:  # pragma: no cover - logging only
            print(f"[WARN] {scan.name} frame {idx}: {e}")
            continue
        image_dim = np.array([w, h])
        depth = build_depth_buffer(T_world_cam, point_cloud, K, image_dim, boarder_pixels)
        scale_x = 320.0 / float(w)
        scale_y = 240.0 / float(h)

        for inst_idx, pts in enumerate(inst_pts):
            mapping = compute_mapping(T_world_cam, pts, depth, K, boarder_pixels, vis_tresh, image_dim).T
            if mapping.shape[0] == 0:
                vis_ratio = 0.0
                pix_cnt = 0
                bbox_scaled = (0, 0, 0, 0)
                pix_ids = np.zeros((0, 2), dtype=np.uint16)
            else:
                vis_mask = mapping[:, 2] == 1
                vis_ratio = float(vis_mask.sum()) / float(mapping.shape[0])
                vis_pixels = mapping[vis_mask]
                if vis_pixels.size == 0:
                    pix_cnt = 0
                    bbox_scaled = (0, 0, 0, 0)
                    pix_ids = np.zeros((0, 2), dtype=np.uint16)
                else:
                    rows = vis_pixels[:, 0]
                    cols = vis_pixels[:, 1]
                    bbox = (
                        int(np.clip(np.floor(cols.min()), 0, w - 1)),
                        int(np.clip(np.floor(rows.min()), 0, h - 1)),
                        int(np.clip(np.ceil(cols.max()), 0, w - 1)),
                        int(np.clip(np.ceil(rows.max()), 0, h - 1)),
                    )
                    bbox_scaled = (
                        int(np.clip(round(bbox[0] * scale_x), 0, 319)),
                        int(np.clip(round(bbox[1] * scale_y), 0, 239)),
                        int(np.clip(round(bbox[2] * scale_x), 0, 319)),
                        int(np.clip(round(bbox[3] * scale_y), 0, 239)),
                    )
                    unique = np.unique(vis_pixels[:, :2], axis=0)
                    pix_cnt = unique.shape[0]
                    pix_ids = np.stack(
                        (
                            np.clip(np.round(unique[:, 0] * scale_y), 0, 239),
                            np.clip(np.round(unique[:, 1] * scale_x), 0, 319),
                        ),
                        axis=1,
                    ).astype(np.uint16)

            scores[inst_idx].append((idx, vis_ratio))
            details[inst_idx][idx] = (
                str(img_path.relative_to(scan)),
                pix_cnt,
                vis_ratio,
                bbox_scaled,
                pix_ids,
            )

    object2frame = {}
    name_map = {}
    for inst_idx, inst_path in enumerate(inst_paths):
        sorted_scores = sorted(scores[inst_idx], key=lambda x: -x[1])
        pos_scores = [i for i, v in sorted_scores if v > 0]
        if pos_scores:
            top = pos_scores[:top_k]
            while len(top) < top_k:
                top.append(pos_scores[0])
        else:
            top = [i for i, _ in sorted_scores[:top_k]]
        object2frame[inst_idx] = [details[inst_idx][i] for i in top if i in details[inst_idx]]
        name_map[inst_idx] = inst_path.stem

    return object2frame, name_map


# -----------------------------
# Main
# -----------------------------


def main():
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
        inst_paths = sorted((scan / "mask/vis_instances").glob("inst_*.ply"))
        inst_pts = [np.asarray(o3d.io.read_point_cloud(str(p)).points) for p in inst_paths]
        img_files = gather_images(scan)

        object2frame, name_map = image_3d_mapping(scan, inst_paths, inst_pts, img_files, args.top_k)
        data = {"names": name_map, **object2frame}
        with open(out_dir / f"{scan.name}_object2frame.pkl", "wb") as fw:
            pickle.dump(data, fw)
        print(f"{scan.name}: {len(inst_paths)} instances processed")


if __name__ == "__main__":
    main()

