import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from tqdm import tqdm

# number of points to store for each object point cloud
OBJ_SAMPLE = 2000
# number of points to store for each union of two objects
REL_SAMPLE = 5000


def o3d_load(p: Path):
    pc = o3d.io.read_point_cloud(str(p))
    pts = np.asarray(pc.points)
    if pc.has_colors():
        cols = np.asarray(pc.colors)
        arr = np.concatenate([pts, cols], axis=1)
    else:
        arr = pts
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--graphs", required=True, help="graphs/train.json")
    ap.add_argument("--frames", required=True, help="directory with *_object2frame.pkl files")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--max_edges_per_node",
        type=int,
        default=8,
        help="Limit number of outgoing edges per node",
    )
    ap.add_argument(
        "--max_nodes",
        type=int,
        default=100,
        help="Limit number of nodes stored for each scan",
    )
    args = ap.parse_args()

    root = Path(args.root)
    frames_dir = Path(args.frames)
    graphs_raw = json.loads(Path(args.graphs).read_text())
    if isinstance(graphs_raw, dict) and "scans" in graphs_raw:
        graphs = {e["scan"]: e.get("graph", {}) for e in graphs_raw["scans"]}
    else:
        graphs = graphs_raw

    out_cache = Path(args.out) / "cache"
    out_cache.mkdir(parents=True, exist_ok=True)
    out_preproc = Path(args.out) / "preprocessed"
    out_preproc.mkdir(parents=True, exist_ok=True)

    for scan_id, g in tqdm(graphs.items()):
        scan_dir = root / scan_id
        raw_candidates = list(scan_dir.glob(f"{scan_id}.*"))
        if not raw_candidates:
            raise FileNotFoundError(f"No raw cloud for {scan_id}")
        raw_cloud = raw_candidates[0]
        arr = o3d_load(raw_cloud)
        torch.save(torch.from_numpy(arr).float(), out_cache / f"{scan_id}_pc.pt")

        inst_meta = []
        objects_pcl = []
        objects_num = []
        objects_center = []
        objects_scale = []
        objects_id = []

        for idx, node in enumerate(g["nodes"]):
            if idx >= args.max_nodes:
                break
            inst_file = scan_dir / node["file"]
            pcl = o3d_load(inst_file)
            if pcl.shape[0] > OBJ_SAMPLE:
                choice = np.random.choice(pcl.shape[0], OBJ_SAMPLE, replace=False)
                pcl = pcl[choice]
            padded = np.zeros((OBJ_SAMPLE, pcl.shape[1]), dtype=pcl.dtype)
            padded[:pcl.shape[0]] = pcl
            objects_pcl.append(padded)
            objects_num.append(min(pcl.shape[0], OBJ_SAMPLE))
            aabb_min = np.array(node["aabb"][0])
            aabb_max = np.array(node["aabb"][1])
            objects_center.append(((aabb_min + aabb_max) / 2).tolist())
            objects_scale.append((aabb_max - aabb_min).tolist())
            objects_id.append(node["inst_id"])
            inst_meta.append({
                "inst_id": node["inst_id"],
                "file": node["file"],
                "centroid": node["centroid"],
                "aabb": node["aabb"],
                "n_points": node["n_points"],
            })

        with open(out_cache / f"{scan_id}_instances.pkl", "wb") as fw:
            pickle.dump(inst_meta, fw)

        # --- object/relationship info for old pipeline ---
        frames_pkl = frames_dir / f"{scan_id}_object2frame.pkl"
        if frames_pkl.exists():
            with open(frames_pkl, "rb") as fr:
                obj2frame = pickle.load(fr)
        else:
            obj2frame = {}

        # keep only keys corresponding to nodes that are actually stored
        kept_idx = set(range(len(objects_pcl)))
        obj2frame = {k: v for k, v in obj2frame.items() if k in kept_idx}

        pairs = []
        edges = []
        triples = []
        # we use two relation classes: index 0 for 'none' (dummy edges) and 1
        # for the spatial relationships generated above.  Downstream loaders
        # expect one-hot encoded rows per edge similar to the 3RScan/ScanNet
        # preprocessing.
        predicate_cat = []
        predicate_num = []
        predicate_pcl_flag = []
        predicate_dist = []
        predicate_min_dist = []
        edge_count = {}
        rels2frame = {}

        centers = np.array(objects_center)
        # map relation labels to indices; currently only 'spatial'
        rel2idx = {"none": 0, "spatial": 1}

        for e in g.get("edges", []):
            s, o, rel = e
            if s >= len(objects_pcl) or o >= len(objects_pcl):
                continue
            if args.max_edges_per_node is not None:
                c = edge_count.get(s, 0)
                if c >= args.max_edges_per_node:
                    continue
                edge_count[s] = c + 1
            pairs.append([s, o])
            edges.append([s, o])
            triples.append([s, 0, o])
            # one-hot encode relation type
            pred_vec = np.zeros(len(rel2idx), dtype=int)
            pred_vec[rel2idx.get(rel, 0)] = 1
            predicate_cat.append(pred_vec.tolist())
            pcl_s = objects_pcl[s]
            pcl_o = objects_pcl[o]
            flag_s = np.hstack([pcl_s, np.ones((pcl_s.shape[0], 1))])
            flag_o = np.hstack([pcl_o, np.full((pcl_o.shape[0], 1), 2)])
            union = np.concatenate([flag_s, flag_o], axis=0)
            if union.shape[0] > REL_SAMPLE:
                choice = np.random.choice(union.shape[0], REL_SAMPLE, replace=False)
                union = union[choice]
            if union.shape[0] < REL_SAMPLE:
                pad = np.zeros((REL_SAMPLE - union.shape[0], union.shape[1]), dtype=union.dtype)
                union = np.concatenate([union, pad], axis=0)
            predicate_pcl_flag.append(union)
            dist_vec = centers[s] - centers[o]
            predicate_dist.append(dist_vec.tolist())
            predicate_min_dist.append(dist_vec.tolist())

            # rel2frame from shared frames
            sf = obj2frame.get(s, [])
            of_ = obj2frame.get(o, [])
            map_s = {f[0]: f for f in sf}
            map_o = {f[0]: f for f in of_}
            shared = set(map_s.keys()) & set(map_o.keys())
            rels = []
            for fid in shared:
                s_f = map_s[fid]
                o_f = map_o[fid]
                rels.append((fid, s_f[1], o_f[1], s_f[2], o_f[2], s_f[3], o_f[3]))
            rels2frame[(s, o)] = rels

        # pad edges so every node has at least `max_edges_per_node` entries
        for idx in range(len(objects_pcl)):
            c = edge_count.get(idx, 0)
            while c < args.max_edges_per_node:
                pairs.append([idx, idx])
                edges.append([idx, idx])
                triples.append([idx, 0, idx])
                none_vec = np.zeros(len(rel2idx), dtype=int)
                none_vec[rel2idx['none']] = 1
                predicate_cat.append(none_vec.tolist())
                predicate_pcl_flag.append(np.zeros((REL_SAMPLE, 7)))
                predicate_num.append(0)
                predicate_dist.append([0.0, 0.0, 0.0])
                predicate_min_dist.append([0.0, 0.0, 0.0])
                rels2frame[(idx, idx)] = []
                c += 1
            edge_count[idx] = c

        data_dict = {
            "scan_id": scan_id,
            "objects_id": objects_id,
            "objects_cat": [0] * len(objects_id),
            "objects_num": objects_num,
            "objects_pcl": [p.tolist() for p in objects_pcl],
            "objects_pcl_glob": [],
            "objects_center": objects_center,
            "objects_scale": objects_scale,
            "predicate_cat": predicate_cat,
            "predicate_num": predicate_num,
            "predicate_pcl_flag": [p.tolist() for p in predicate_pcl_flag],
            "predicate_dist": predicate_dist,
            "predicate_min_dist": predicate_min_dist,
            "pairs": pairs,
            "edges": edges,
            "triples": triples,
            "objects_count": len(objects_id),
            "predicate_count": len(predicate_cat),
            "tight_bbox": [
                (objects_scale[i] + objects_center[i] + [0]) for i in range(len(objects_id))
            ],
            "object2frame": obj2frame,
            "rel2frame": rels2frame,
            "id2name": {str(i): str(i) for i in objects_id},
        }

        scan_out = out_preproc / scan_id
        scan_out.mkdir(parents=True, exist_ok=True)
        with open(scan_out / "data_dict_0.pkl", "wb") as fw:
            pickle.dump(data_dict, fw)

    print("Preprocessing finished.")


if __name__ == "__main__":
    main()
