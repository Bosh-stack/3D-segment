import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple

from open3dsg.config.config import CONF
from .get_object_frame_myset import load_cam


def _gather_pose_images(scan_dir: Path) -> List[Path]:
    """Collect pose image files for a scan directory.

    Looks for a ``pose_images`` subdirectory first; if not found, falls back to
    files in ``scan_dir`` starting with ``pose_image``.  Accepts common image
    extensions (``png``/``jpg``)."""
    patterns = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG"]
    img_files: List[Path] = []
    pose_dir = scan_dir / "pose_images"
    if pose_dir.is_dir():
        for ptn in patterns:
            img_files.extend(sorted(pose_dir.glob(ptn)))
    else:
        for ptn in patterns:
            img_files.extend(sorted(scan_dir.glob(f"pose_image_{ptn}")))
    return sorted(img_files)


def read_pointcloud_myset(scan_id: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load per-instance point clouds and concatenate them.

    Parameters
    ----------
    scan_id : str
        Identifier of the scan directory under ``CONF.PATH.MYSET_ROOT``.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        ``points`` (N,3) and integer ``instance`` labels (N,).
    """
    scan_dir = Path(CONF.PATH.MYSET_ROOT) / scan_id
    inst_dir = scan_dir / "mask" / "vis_instances"
    pts_all: List[np.ndarray] = []
    inst_labels: List[np.ndarray] = []
    if inst_dir.is_dir():
        inst_files = sorted(inst_dir.glob("inst_*.ply"))
        for idx, ply in enumerate(inst_files, start=1):
            try:
                import trimesh

                mesh = trimesh.load(ply, process=False)
                pts = np.asarray(mesh.vertices, dtype=np.float32)
            except Exception:
                pts = np.loadtxt(ply, dtype=np.float32, usecols=(0, 1, 2))
            pts_all.append(pts)
            inst_labels.append(np.full(pts.shape[0], idx, dtype=np.int32))
    if pts_all:
        points = np.concatenate(pts_all, axis=0)
        inst = np.concatenate(inst_labels, axis=0)
    else:
        points = np.zeros((0, 3), dtype=np.float32)
        inst = np.zeros((0,), dtype=np.int32)
    return points, inst


def read_scan_info_myset(scan_id: str):
    """Read RGB/depth images and camera metadata for a scan.

    The function searches for pose images and their corresponding metadata
    (``im_metadata_XX.json``).  Depth images are optional; if missing, a zero
    array is returned for that frame.  Depth values are kept in millimetres.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]
        depths, colors, extrinsics, intrinsic matrix, image names.
    """
    scan_dir = Path(CONF.PATH.MYSET_ROOT) / scan_id
    img_files = _gather_pose_images(scan_dir)
    depths: List[np.ndarray] = []
    colors: List[np.ndarray] = []
    extrinsics: List[np.ndarray] = []
    img_names: List[str] = []
    intrinsic = None

    for img_path in img_files:
        try:
            idx = int(img_path.stem.split("_")[-1])
        except Exception:
            idx = img_files.index(img_path)

        meta_candidates = [
            img_path.with_name(f"im_metadata_{idx}.json"),
            scan_dir / f"im_metadata_{idx}.json",
            img_path.with_suffix(".json"),
        ]
        meta = next((m for m in meta_candidates if m.exists()), None)
        if meta is None:
            continue
        try:
            K, T, w, h = load_cam(meta)
        except Exception:
            continue
        if intrinsic is None:
            intrinsic = K
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        colors.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img_names.append(img_path.name)
        extrinsics.append(T)

        depth = None
        depth_candidates = [
            img_path.with_name(f"depth_{idx}.png"),
            scan_dir / "depth" / f"{idx}.png",
            img_path.with_suffix(".png"),
        ]
        for d in depth_candidates:
            if d.exists():
                depth = cv2.imread(str(d), cv2.IMREAD_UNCHANGED)
                break
        if depth is None:
            depth = np.zeros((h, w), dtype=np.float32)
        depths.append(depth.reshape(-1))

    depths_arr = np.stack(depths) if depths else np.zeros((0, 0), dtype=np.float32)
    colors_arr = np.stack(colors) if colors else np.zeros((0, 0, 0), dtype=np.uint8)
    extrinsics_arr = np.stack(extrinsics) if extrinsics else np.zeros((0, 4, 4), dtype=np.float32)
    return depths_arr, colors_arr, extrinsics_arr, intrinsic, img_names
