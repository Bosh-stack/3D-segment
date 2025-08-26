import argparse
import pickle
from pathlib import Path

import numpy as np
import cv2
from sklearn.cluster import DBSCAN

from open3dsg.config.config import CONF
from open3dsg.data.myset_io import (
    read_pointcloud_myset,
    read_scan_info_myset,
)
from open3dsg.data.get_object_frame import compute_mapping


def aggregate_point_features(points, frame_indices, depths, extrinsics, intrinsic, images):
    """Aggregate per-point RGB features across frames.

    Parameters
    ----------
    points : ndarray (N,3)
        3D coordinates of a single instance.
    frame_indices : list[int]
        Frames to aggregate over.
    depths : ndarray (F,H*W)
        Depth images for all frames as flattened arrays in millimetres.
    extrinsics : ndarray (F,4,4)
        Camera extrinsic matrices (world->camera).
    intrinsic : ndarray (3,3)
        Camera intrinsic matrix.
    images : list or ndarray (F,H,W,3)
        RGB images aligned with ``depths``.

    Returns
    -------
    ndarray (N,3)
        Averaged RGB feature for each point, normalized to [0,1].
    """
    if points.size == 0 or len(frame_indices) == 0:
        return np.zeros((points.shape[0], 3), dtype=np.float32)

    image_dim = np.array([images[0].shape[1], images[0].shape[0]])
    feats = np.zeros((points.shape[0], 3), dtype=np.float32)
    counts = np.zeros(points.shape[0], dtype=np.int32)

    for f_idx in frame_indices:
        depth = depths[f_idx].reshape(image_dim[::-1]) / 1000.0
        mapping = compute_mapping(extrinsics[f_idx], points, depth, intrinsic, 0, 0.05, image_dim).T
        vis = mapping[:, 2] == 1
        if not np.any(vis):
            continue
        pix = mapping[vis, :2].astype(int)
        pix[:, 0] = np.clip(pix[:, 0], 0, image_dim[1] - 1)
        pix[:, 1] = np.clip(pix[:, 1], 0, image_dim[0] - 1)
        colors = images[f_idx][pix[:, 0], pix[:, 1]].astype(np.float32) / 255.0
        feats[vis] += colors
        counts[vis] += 1

    counts[counts == 0] = 1
    feats /= counts[:, None]
    return feats


def main():
    parser = argparse.ArgumentParser(
        description="Split 3D instances using pixel features and DBSCAN"
    )
    parser.add_argument("--scan", required=True, help="scan id")
    parser.add_argument(
        "--dataset", default="MYSET", choices=["MYSET"], help="dataset name"
    )
    parser.add_argument("--eps", type=float, default=0.3, help="DBSCAN epsilon")
    parser.add_argument("--min_samples", type=int, default=10, help="DBSCAN min samples")
    args = parser.parse_args()

    pc, inst = read_pointcloud_myset(args.scan)
    depths, colors, extrinsics, intrinsic, img_names = read_scan_info_myset(
        args.scan
    )
    views_dir = Path(CONF.PATH.MYSET_PREPROC_OUT) / "frames"

    obj2frame_file = views_dir / f"{args.scan}_object2frame.pkl"
    if not obj2frame_file.exists():
        raise FileNotFoundError(f"{obj2frame_file} not found")
    obj2frame = pickle.load(open(obj2frame_file, "rb"))

    name_to_idx = {n: i for i, n in enumerate(img_names)}
    new_labels = np.zeros(len(inst), dtype=int)
    new_obj2frame = {}
    next_id = 1

    for inst_id, frames in obj2frame.items():
        if inst_id == "names":
            continue
        inst_id_int = int(inst_id)
        idx = np.where(inst == inst_id_int)[0]
        if idx.size == 0:
            continue
        pts = pc[idx, :3]
        f_indices = [name_to_idx[f[0]] for f in frames]
        feats = aggregate_point_features(pts, f_indices, depths, extrinsics, intrinsic, colors)
        if feats.size == 0:
            continue
        clustering = DBSCAN(eps=args.eps, min_samples=args.min_samples).fit(feats)
        labels = clustering.labels_
        for cl in np.unique(labels):
            if cl == -1:
                continue
            local_idx = idx[labels == cl]
            out_path = views_dir / f"{args.scan}_instance_{next_id}.ply"
            try:
                import trimesh
                trimesh.PointCloud(pc[local_idx, :3]).export(out_path)
            except Exception:
                pass
            new_labels[local_idx] = next_id
            new_obj2frame[str(next_id)] = frames
            next_id += 1

    np.save(views_dir / f"{args.scan}_instances_split.npy", new_labels)
    out_file = views_dir / f"{args.scan}_object2frame_split.pkl"
    with open(out_file, "wb") as fw:
        pickle.dump(new_obj2frame, fw)
    print(f"Split instances written to {out_file}")


if __name__ == "__main__":
    main()
