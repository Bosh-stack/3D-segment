#!/usr/bin/env python3
"""Assemble a self-contained graph directory for a single scan.

This utility gathers instance meshes, RGB frames and relation labels into a
`graph/` directory that mirrors the file layout expected by downstream tools.

Inputs
------
* Subgraph JSON (from ``gen_myset_subgraphs.py``) describing nodes/edges.
* Directory with ``<scan>_object2frame.pkl`` produced by
  ``get_object_frame_myset.py``.
* TSV containing edge labels (``extract_open3dsg_style_labels.py`` output).
* Source directories for instance ``.ply`` files and RGB frames.
* ``vis_pred.ply`` and the original point cloud for reference.

The resulting directory contains::

    graph/
      graph.json
      masks/      # instance meshes
      images/     # RGB frames for each node
      full_pc/    # full point cloud and visualised predictions

All paths stored in ``graph.json`` are relative to the ``graph/`` folder.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import shutil
from pathlib import Path
from typing import Dict, List, Set


def _load_object2frame(pkl_dir: Path) -> tuple[Dict[int, List], Dict[int, str]]:
    """Load ``*_object2frame.pkl`` and return (mapping, name_map)."""
    pkl_files = list(pkl_dir.glob("*_object2frame.pkl"))
    if len(pkl_files) != 1:
        raise FileNotFoundError(
            f"Expected exactly one *_object2frame.pkl in {pkl_dir}, found {len(pkl_files)}"
        )
    with open(pkl_files[0], "rb") as f:
        data = pickle.load(f)
    names = {int(k): v for k, v in (data.get("names") or {}).items()}
    obj2frame = {
        int(k): v for k, v in data.items() if isinstance(k, (int, str)) and str(k).isdigit()
    }
    return obj2frame, names


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subgraph_json", required=True)
    ap.add_argument("--object2frame_dir", required=True)
    ap.add_argument("--edge_tsv", required=True)
    ap.add_argument("--ply_src", required=True, help="directory with instance .ply files")
    ap.add_argument("--rgb_src", required=True, help="directory with RGB frames")
    ap.add_argument("--vis_ply", required=True)
    ap.add_argument("--pcd", required=True, help="original point cloud")
    ap.add_argument("--out_dir", required=True, help="destination graph directory")
    ap.add_argument("--top_k", type=int, default=None, help="limit frames per node")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    masks_dir = out_dir / "masks"
    images_dir = out_dir / "images"
    full_pc_dir = out_dir / "full_pc"
    for d in (masks_dir, images_dir, full_pc_dir):
        d.mkdir(parents=True, exist_ok=True)

    with open(args.subgraph_json, "r", encoding="utf-8") as f:
        sg = json.load(f)
    nodes = sg.get("nodes") or sg.get("graph", {}).get("nodes", [])

    obj2frame, name_map = _load_object2frame(Path(args.object2frame_dir))

    node_labels: Dict[int, str] = {}
    node_ids: Set[int] = set()
    edges = []
    with open(args.edge_tsv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            s = int(row["src_idx"])
            t = int(row["dst_idx"])
            node_ids.add(s)
            node_ids.add(t)
            sn = str(row.get("src_name", "")).strip()
            dn = str(row.get("dst_name", "")).strip()
            if sn:
                node_labels[s] = sn
            if dn:
                node_labels[t] = dn
            if s == t:
                continue
            edges.append(
                {
                    "source": s,
                    "target": t,
                    "relation": row.get("relation", ""),
                    "caption": row.get("caption", ""),
                }
            )

    ply_src = Path(args.ply_src)
    rgb_src = Path(args.rgb_src)

    if not nodes:
        nodes = [
            {"inst_id": idx, "file": f"inst_{idx}.ply"}
            for idx in sorted(node_ids)
        ]

    nodes_out = []
    for n in nodes:
        idx = int(n.get("inst_id", n.get("id")))

        label = ""
        for candidate in (
            node_labels.get(idx),
            name_map.get(idx),
            n.get("label"),
            n.get("name"),
        ):
            if candidate:
                label = str(candidate).strip()
                if label:
                    break
        if not label:
            label = str(idx)

        src_ply_name = Path(n.get("file", f"inst_{idx}.ply")).name
        src_ply = ply_src / src_ply_name
        dst_ply_name = f"inst_{idx}.ply"
        shutil.copy2(src_ply, masks_dir / dst_ply_name)

        img_list: List[str] = []
        frames = obj2frame.get(idx, [])
        if args.top_k is not None:
            frames = frames[: args.top_k]
        for j, fr in enumerate(frames):
            frame = fr[0]
            src_img = rgb_src / str(frame)
            if not src_img.exists():
                found_img = None
                for sub in rgb_src.glob("images*"):
                    if not sub.is_dir():
                        continue
                    sub_img = sub / str(frame)
                    if sub_img.exists():
                        found_img = sub_img
                        break
                    sub_candidates = list(sub.glob(f"{frame}.*"))
                    if sub_candidates:
                        found_img = sub_candidates[0]
                        break
                if found_img is not None:
                    src_img = found_img
                else:
                    cand = list(rgb_src.glob(f"{frame}.*"))
                    if cand:
                        src_img = cand[0]
                    else:
                        continue
            dst_img_name = f"{idx}_{j}{src_img.suffix}"
            shutil.copy2(src_img, images_dir / dst_img_name)
            img_list.append(f"images/{dst_img_name}")

        nodes_out.append(
            {
                "id": idx,
                "label": label,
                "ply": f"masks/{dst_ply_name}",
                "images": img_list,
            }
        )

    shutil.copy2(args.vis_ply, full_pc_dir / Path(args.vis_ply).name)
    shutil.copy2(args.pcd, full_pc_dir / Path(args.pcd).name)

    graph = {"nodes": nodes_out, "edges": edges}
    with open(out_dir / "graph.json", "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2)
    print(f"Wrote {out_dir / 'graph.json'}  | nodes={len(nodes_out)}  edges={len(edges)}")


if __name__ == "__main__":
    main()
