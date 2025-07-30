import argparse
import os
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import cv2
from open3dsg.config.config import CONF
from open3dsg.data.get_object_frame import (
    read_pointcloud_R3SCAN,
    read_pointcloud_scannet,
    read_scan_info_R3SCAN,
    read_scan_info_scannet,
    compute_mapping,
)


def merge_instances_by_2d_masks(point_cloud, instances, depths, extrinsics, intrinsic, masks, overlap_thresh=0.5):
    """Merge 3D instances if they overlap with the same 2D mask.

    Parameters
    ----------
    point_cloud : ndarray (N,3)
        Point coordinates.
    instances : ndarray (N,)
        Instance id per point.
    depths : list or ndarray (F,H*W)
        Depth image per frame as flattened arrays in millimetres.
    extrinsics : ndarray (F,4,4)
        Camera extrinsic matrices.
    intrinsic : ndarray (3,3)
        Camera intrinsic matrix.
    masks : ndarray (F,H,W)
        2D segmentation mask id per pixel for each frame.
    overlap_thresh : float
        Minimum fraction of points projecting into the same 2D mask to trigger a merge.
    """

    uniq = np.unique(instances)
    image_dim = np.array([masks.shape[2], masks.shape[1]])
    inst_info = {int(i): {} for i in uniq}

    for f_idx in range(len(masks)):
        w2c = np.linalg.inv(extrinsics[f_idx])
        depth = depths[f_idx].reshape(image_dim[::-1]) / 1000.0
        mask = masks[f_idx]
        for inst_id in uniq:
            pts = point_cloud[instances == inst_id]
            if pts.size == 0:
                continue
            mapping = compute_mapping(w2c, pts, depth, intrinsic, 0, 0.05, image_dim).T
            vis = mapping[:, 2] == 1
            pix = mapping[vis, :2].astype(int)
            if pix.size == 0:
                continue
            pix[:, 0] = np.clip(pix[:, 0], 0, image_dim[1] - 1)
            pix[:, 1] = np.clip(pix[:, 1], 0, image_dim[0] - 1)
            labels = mask[pix[:, 0], pix[:, 1]]
            if labels.size == 0:
                continue
            vals, counts = np.unique(labels, return_counts=True)
            major = vals[counts.argmax()]
            frac = counts.max() / float(labels.size)
            inst_info[int(inst_id)][f_idx] = (int(major), float(frac))

    adj = defaultdict(set)
    uniq_list = [int(i) for i in uniq]
    for i, a in enumerate(uniq_list):
        for b in uniq_list[i + 1 :]:
            shared = set(inst_info[a].keys()) & set(inst_info[b].keys())
            for f_idx in shared:
                lab_a, frac_a = inst_info[a][f_idx]
                lab_b, frac_b = inst_info[b][f_idx]
                if lab_a == lab_b and frac_a >= overlap_thresh and frac_b >= overlap_thresh:
                    adj[a].add(b)
                    adj[b].add(a)

    visited = set()
    components = []
    for inst in uniq_list:
        if inst in visited:
            continue
        stack = [inst]
        comp = []
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            comp.append(cur)
            stack.extend(adj.get(cur, []))
        components.append(comp)

    merged_instances = instances.copy()
    for new_id, comp in enumerate(components, 1):
        for old in comp:
            merged_instances[instances == old] = new_id

    return merged_instances, components


def main():
    parser = argparse.ArgumentParser(description="Merge 3D instances using 2D masks")
    parser.add_argument("--scan", required=True, help="scan id")
    parser.add_argument("--dataset", required=True, choices=["R3SCAN", "SCANNET"], help="dataset name")
    parser.add_argument("--masks_dir", required=True, help="directory with 2D masks")
    args = parser.parse_args()

    if args.dataset == "R3SCAN":
        pc, inst = read_pointcloud_R3SCAN(args.scan)
        depths, _, extrinsics, intr_info, img_names = read_scan_info_R3SCAN(args.scan)
        intrinsic = intr_info["m_intrinsic"]
        views_dir = Path(CONF.PATH.R3SCAN) / "views"
    else:
        pc, _, inst = read_pointcloud_scannet(args.scan)
        depths, _, extrinsics, intr_info, img_names = read_scan_info_scannet(args.scan)
        intrinsic = np.loadtxt(os.path.join(CONF.PATH.SCANNET_RAW2D, "intrinsics.txt"))
        views_dir = Path(CONF.PATH.SCANNET) / "views"

    mask_paths = [Path(args.masks_dir) / f"{Path(n).stem}.png" for n in img_names]
    masks = [cv2.imread(str(p), -1) for p in mask_paths]
    masks = np.stack(masks)

    merged_labels, comps = merge_instances_by_2d_masks(pc[:, :3], inst, depths, extrinsics, intrinsic, masks)

    obj2frame_file = views_dir / f"{args.scan}_object2image.pkl"
    if obj2frame_file.exists():
        obj2frame = pickle.load(open(obj2frame_file, "rb"))
    else:
        obj2frame = {}

    new_obj2frame = {}
    for new_id, group in enumerate(comps, 1):
        merged = []
        for old in group:
            merged.extend(obj2frame.get(str(old), []))
        if merged:
            new_obj2frame[str(new_id)] = merged

    out_file = views_dir / f"{args.scan}_object2image_merged.pkl"
    with open(out_file, "wb") as fw:
        pickle.dump(new_obj2frame, fw)

    np.save(views_dir / f"{args.scan}_instances_merged.npy", merged_labels)
    print(f"Merged instances written to {out_file}")


if __name__ == "__main__":
    main()
