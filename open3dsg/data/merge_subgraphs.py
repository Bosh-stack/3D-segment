import argparse
import json
import os
import pickle
from collections import defaultdict
from pathlib import Path

import torch

from open3dsg.config.config import CONF


def load_feature_dirs(features_dir: Path):
    obj_dir = None
    valid_dir = None
    rel_dir = None
    for d in features_dir.iterdir():
        name = d.name
        if "export_obj_clip_emb" in name:
            obj_dir = d
        elif "export_obj_clip_valids" in name:
            valid_dir = d
        elif "export_rel_clip_emb" in name:
            rel_dir = d
    return obj_dir, valid_dir, rel_dir


def merge_subgraphs(split: str, features_dir: Path, out_dir: Path,
                     preproc_dir: Path, subgraph_dir: Path):
    rel_file = subgraph_dir / f"relationships_{split}.json"
    data = json.loads(rel_file.read_text())
    scans = defaultdict(list)
    neighbors = data.get("neighbors", {})
    for r in data["scans"]:
        scans[r["scan"]].append(r)

    obj_dir, valid_dir, rel_dir = load_feature_dirs(features_dir)
    out_obj_dir, out_valid_dir, out_rel_dir = load_feature_dirs(out_dir)
    out_obj_dir.mkdir(parents=True, exist_ok=True)
    out_valid_dir.mkdir(parents=True, exist_ok=True)
    if rel_dir:
        out_rel_dir.mkdir(parents=True, exist_ok=True)

    merged_scans = []
    for scan_id, rels in scans.items():
        objects = {}
        rel_list = []
        obj_feats = {}
        obj_valids = {}
        edge_pairs = set()
        for rel in rels:
            split_idx = rel["split"]
            sg_name = f"{scan_id}-{hex(split_idx)[-1]}"
            pkl_path = preproc_dir / scan_id / f"data_dict_{hex(split_idx)[-1]}.pkl"
            if pkl_path.exists():
                data_dict = pickle.load(open(pkl_path, "rb"))
                obj_ids = data_dict["objects_id"]
                feats = torch.load(obj_dir / f"{sg_name}.pt")
                valids = torch.load(valid_dir / f"{sg_name}.pt")
                for oid, feat, val in zip(obj_ids, feats, valids):
                    if oid not in obj_feats:
                        obj_feats[oid] = feat
                        obj_valids[oid] = val
            objects.update({int(k): v for k, v in rel["objects"].items()})
            rel_list.extend(rel.get("relationships", []))
            edge_pairs.update({(r[0], r[1]) for r in rel.get("relationships", [])})

        # add cross-subgraph edges
        neigh = neighbors.get(scan_id, {})
        for k, nbs in neigh.items():
            k = int(k)
            for nb in nbs:
                pair = (k, nb)
                if pair not in edge_pairs and k in objects and nb in objects:
                    rel_list.append([k, nb, 0])
                    edge_pairs.add(pair)

        # save merged features
        if obj_feats:
            ids_sorted = sorted(obj_feats.keys())
            feats = torch.stack([obj_feats[i] for i in ids_sorted])
            valids = torch.stack([obj_valids[i] for i in ids_sorted])
            torch.save(feats, out_obj_dir / f"{scan_id}.pt")
            torch.save(valids, out_valid_dir / f"{scan_id}.pt")
        merged_scans.append({
            "scan": scan_id,
            "split": 0,
            "objects": {str(i): objects[i] for i in sorted(objects)},
            "relationships": rel_list,
        })

    out_json = out_dir / f"relationships_{split}_full.json"
    with open(out_json, "w") as f:
        json.dump({"scans": merged_scans}, f)


def main():
    parser = argparse.ArgumentParser(description="Merge ScanNet subgraphs")
    parser.add_argument("--split", required=True, choices=["train", "validation", "test"], help="dataset split")
    parser.add_argument("--features_dir", required=True, help="directory with subgraph features")
    parser.add_argument("--out_dir", required=True, help="output directory")
    parser.add_argument("--preproc_dir", default=os.path.join(CONF.PATH.SCANNET, "preprocessed"))
    parser.add_argument("--subgraph_dir", default=os.path.join(CONF.PATH.SCANNET, "subgraphs"))
    args = parser.parse_args()

    merge_subgraphs(args.split, Path(args.features_dir), Path(args.out_dir),
                    Path(args.preproc_dir), Path(args.subgraph_dir))


if __name__ == "__main__":
    main()
