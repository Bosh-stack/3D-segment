#!/usr/bin/env python3
"""MySet frame association utility.

This script associates each 3D instance with the ``top_k`` RGB frames where the
object is the most visible.  It is modelled after :mod:`get_object_frame.py` but
supports the more relaxed camera metadata found in the custom *myset* dataset.

The output for every scan is a ``pickle`` file named ``<scan_id>_object2frame.pkl``
containing a dictionary ``{inst_id: [(frame_id, pixels, ratio, bbox, pixel_ids)]}``.
Each tuple mimics the structure expected by the legacy preprocessing pipeline so
that downstream tools (e.g. ``preprocess_3rscan.py``) can operate on the data.

Camera metadata can be provided in various schemas:

* direct ``fx/fy/cx/cy`` keys
* a 3x3 intrinsic matrix under ``"K"``, ``"intrinsic"`` or ``"intrinsic_matrix"``
* ``focal``/``pixel``/``principal`` blocks
* horizontal/vertical field of view values
* extrinsics either as quaternion+translation or a full 4x4 matrix

Example::

    python open3dsg/data/get_object_frame_myset.py \
        --root /data/Open3DSG_trainset \
        --out  open3dsg/output/preprocessed/myset/frames \
        --top_k 5
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Tuple

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
    """Load intrinsics/extrinsics from flexible JSON schemas."""
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

    # focal/pixel/principal block (user example)
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

    return K, T, int(w), int(h)


# -----------------------------
# Visibility computation
# -----------------------------

def project_points(pts_cam: np.ndarray, K: np.ndarray):
    zs = pts_cam[:, 2]
    uvs = (K @ pts_cam[:, :3].T).T
    uvs = uvs[:, :2] / zs[:, None]
    return uvs, zs


def visible_ratio(points_world: np.ndarray, K: np.ndarray, T_world_cam: np.ndarray, w: int, h: int) -> float:
    pts_world_h = np.concatenate([points_world, np.ones((points_world.shape[0], 1))], axis=1)
    pts_cam = (np.linalg.inv(T_world_cam) @ pts_world_h.T).T[:, :3]
    infront = pts_cam[:, 2] > 0
    if not np.any(infront):
        return 0.0
    uvs, _ = project_points(pts_cam[infront], K)
    inside = (uvs[:, 0] >= 0) & (uvs[:, 0] < w) & (uvs[:, 1] >= 0) & (uvs[:, 1] < h)
    return float(inside.sum()) / float(points_world.shape[0])


def projection_details(points_world: np.ndarray, K: np.ndarray, T_world_cam: np.ndarray, w: int, h: int):
    """Project ``points_world`` into the image and compute visibility stats."""
    pts_world_h = np.concatenate([points_world, np.ones((points_world.shape[0], 1))], axis=1)
    pts_cam = (np.linalg.inv(T_world_cam) @ pts_world_h.T).T[:, :3]
    infront = pts_cam[:, 2] > 0
    if not np.any(infront):
        return 0.0, 0, (0, 0, 0, 0), np.zeros((0, 2), dtype=np.uint16)

    uvs, zs = project_points(pts_cam[infront], K)
    inside = (uvs[:, 0] >= 0) & (uvs[:, 0] < w) & (uvs[:, 1] >= 0) & (uvs[:, 1] < h)
    if not np.any(inside):
        return 0.0, 0, (0, 0, 0, 0), np.zeros((0, 2), dtype=np.uint16)

    vis_ratio = float(inside.sum()) / float(points_world.shape[0])
    uv_vis = uvs[inside]
    bbox = (
        int(np.clip(np.floor(uv_vis[:, 0].min()), 0, w - 1)),
        int(np.clip(np.floor(uv_vis[:, 1].min()), 0, h - 1)),
        int(np.clip(np.ceil(uv_vis[:, 0].max()), 0, w - 1)),
        int(np.clip(np.ceil(uv_vis[:, 1].max()), 0, h - 1)),
    )
    pixels = np.unique(np.round(uv_vis).astype(np.uint16), axis=0)
    return vis_ratio, pixels.shape[0], bbox, pixels


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
        scan_id = scan.name
        inst_paths = sorted((scan / "mask/vis_instances").glob("inst_*.ply"))
        inst_pts = [np.asarray(o3d.io.read_point_cloud(str(p)).points) for p in inst_paths]

        img_files = sorted(scan.glob("image_*.*"))
        object2frame = {}

        for inst_idx, pts in enumerate(inst_pts):
            scores = []
            details = {}
            for img_path in img_files:
                try:
                    idx = int(img_path.stem.split("_")[-1])
                except Exception:
                    # if naming differs, fall back to enumeration index
                    idx = img_files.index(img_path)
                meta = scan / f"im_metadata_{idx}.json"
                if not meta.exists():
                    continue
                try:
                    K, T, w, h = load_cam(meta)
                except Exception as e:
                    print(f"[WARN] {scan_id} frame {idx}: {e}")
                    continue
                vis, pix_cnt, bbox, pix_ids = projection_details(pts, K, T, w, h)
                # normalize pixel coordinates to 320x240 so downstream loaders
                # can directly index the resized features
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
                            np.clip(np.round(pix_ids[:, 0] * scale_x), 0, 319),
                            np.clip(np.round(pix_ids[:, 1] * scale_y), 0, 239),
                        ),
                        axis=1,
                    ).astype(np.uint16)
                scores.append((idx, vis))
                # store the frame file name instead of numeric index so the
                # dataset loader can directly resolve the image path
                details[idx] = (img_path.name, pix_cnt, vis, bbox, pix_ids)
            top = [i for i, _ in sorted(scores, key=lambda x: -x[1])[: args.top_k]]
            object2frame[int(inst_idx)] = [details[i] for i in top if i in details]

        with open(out_dir / f"{scan_id}_object2frame.pkl", "wb") as fw:
            pickle.dump(object2frame, fw)
        print(f"{scan_id}: {len(inst_paths)} instances processed")


if __name__ == "__main__":
    main()
