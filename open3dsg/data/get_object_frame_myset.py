import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R


def load_cam(meta_file: Path):
    m = json.loads(meta_file.read_text())
    fx, fy = m["fx"], m["fy"]
    cx, cy = m["cx"], m["cy"]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    w, h = m["width"], m["height"]

    quat = [m["qw"], m["qx"], m["qy"], m["qz"]]
    rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
    trans = np.array([m["tx"], m["ty"], m["tz"]], dtype=np.float32)

    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = rot
    T[:3, 3] = trans
    return K, T, w, h


def project_points(pts_cam, K):
    zs = pts_cam[:, 2]
    uvs = (K @ pts_cam[:, :3].T).T
    uvs = uvs[:, :2] / zs[:, None]
    return uvs, zs


def visible_ratio(points_world, K, T_world_cam, w, h):
    pts_world_h = np.concatenate([points_world, np.ones((points_world.shape[0], 1))], axis=1)
    pts_cam = (np.linalg.inv(T_world_cam) @ pts_world_h.T).T[:, :3]
    infront = pts_cam[:, 2] > 0
    if not np.any(infront):
        return 0.0
    uvs, _ = project_points(pts_cam[infront], K)
    inside = (uvs[:, 0] >= 0) & (uvs[:, 0] < w) & (uvs[:, 1] >= 0) & (uvs[:, 1] < h)
    return float(inside.sum()) / float(points_world.shape[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dataset root")
    ap.add_argument("--out", required=True, help="output dir for frames json")
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
        frame_scores = {}

        for inst_idx, pts in enumerate(inst_pts):
            scores = []
            for img_path in img_files:
                idx = int(img_path.stem.split("_")[-1])
                meta = scan / f"im_metadata_{idx}.json"
                if not meta.exists():
                    continue
                K, T, w, h = load_cam(meta)
                score = visible_ratio(pts, K, T, w, h)
                scores.append((idx, score))
            top = sorted(scores, key=lambda x: -x[1])[: args.top_k]
            frame_scores[int(inst_idx)] = [i for i, _ in top]

        (out_dir / f"{scan_id}.json").write_text(json.dumps(frame_scores))
        print(f"{scan_id}: {len(inst_paths)} instances processed")


if __name__ == "__main__":
    main()
