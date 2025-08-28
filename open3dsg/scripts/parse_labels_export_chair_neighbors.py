#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parse the TSV produced by `extract_open3dsg_style_labels.py` and, for every node
predicted as "chair", export:

  • ONE coloured PLY per chair node:
      - the chair is a fixed colour,
      - each neighbour (node linked to it by an edge) has a unique colour,
      - all subclouds/instances are merged exactly like in query_node_features.py.

  • ONE LOG (txt) per chair node, listing:
      chair_id, chair_colour_rgb,
      neighbour_id, neighbour_colour_rgb,
      relation (from TSV), and the full caption (if available).

Expected TSV header from the extractor:
  src_idx  dst_idx  src_name  dst_name  relation  caption

Usage
-----
python parse_labels_export_chair_neighbors.py \
  --labels /PATH/edge_relations.tsv \
  --graph /PATH/data_dict_SCENE.pkl \
  --out_dir /PATH/out_dir \
  --chair_word chair \
  --include_caption

Notes
-----
- We only *parse* the relation/caption from the TSV (no LLM here).
- Point clouds are merged using the same logic as in query_node_features.py:
  prefer `objects_pcl_glob`; else reconstruct global coords via
  `objects_pcl` + `objects_center` (+ optional `objects_scale`).
"""

from __future__ import annotations
import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle

import numpy as np
import open3d as o3d


# ------------------------- Colours ------------------------- #

CHAIR_RGB = np.array([255, 0, 0], dtype=np.uint8)  # fixed chair colour

# neighbour palette (cycled if needed)
_NEIGH_PALETTE = np.array([
    [ 60, 180,  75], [  0, 130, 200], [245, 130,  48], [145,  30, 180],
    [ 70, 240, 240], [240,  50, 230], [210, 245,  60], [250, 190, 190],
    [  0, 128, 128], [230, 190, 255], [170, 110,  40], [255, 250, 200],
    [128,   0,   0], [170, 255, 195], [128, 128,   0], [255, 215, 180],
    [  0,   0, 128], [128, 128, 128], [255, 105, 180], [  0, 255, 255],
], dtype=np.uint8)


# ------------------------- IO helpers ------------------------- #

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Parse relations TSV and export per-chair PLY + LOG")
    ap.add_argument("--labels", required=True, help="TSV from extract_open3dsg_style_labels.py")
    ap.add_argument("--graph", required=True, help="Pickled scene dict (data_dict_*.pkl)")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--chair_word", default="chair", help="Class name to treat as 'chair' (case-insensitive)")
    ap.add_argument("--include_caption", action="store_true", help="Include full caption text in logs")
    return ap.parse_args()


def load_graph(graph_pkl: str) -> dict:
    with open(graph_pkl, "rb") as f:
        return pickle.load(f)


def load_clouds_merged(graph: dict) -> List[np.ndarray]:
    """
    Same merging approach as in query_node_features.py.
    Returns a list of per-object point arrays in global coords.
    """
    if "objects_pcl_glob" in graph and len(graph["objects_pcl_glob"]) > 0:
        obj_pcls = np.asarray(graph["objects_pcl_glob"], dtype=np.float32)
        return [p[:, :3] for p in obj_pcls]

    obj_pcls = np.asarray(graph["objects_pcl"], dtype=np.float32)   # (N, P, 6?) -> use first 3 columns
    centers  = np.asarray(graph["objects_center"], dtype=np.float32)  # (N, 3)
    scales   = graph.get("objects_scale")
    if scales is not None:
        scales = np.asarray(scales, dtype=np.float32)

    clouds: List[np.ndarray] = []
    for i, obj in enumerate(obj_pcls):
        pts = obj[:, :3]
        if scales is not None:
            s = np.asarray(scales[i]).reshape(1, -1)
            pts = pts * s
        pts = pts + centers[i]
        clouds.append(pts)
    return clouds


def read_labels_tsv(tsv_path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        # Expected fields: src_idx dst_idx src_name dst_name relation caption
        for row in reader:
            rows.append(row)
    return rows


# ------------------------- Core logic ------------------------- #

def build_chair_edge_index(
    rows: List[Dict[str, str]],
    chair_word: str
) -> Dict[int, List[Dict[str, str]]]:
    """
    Returns a mapping: chair_node_id -> list of TSV rows where that chair participates.
    The same chair may appear as src or dst. Case-insensitive comparison on names.
    """
    chair = chair_word.strip().lower()
    per_chair: Dict[int, List[Dict[str, str]]] = {}

    for r in rows:
        try:
            s = int(r["src_idx"])
            t = int(r["dst_idx"])
        except Exception:
            # skip malformed rows
            continue

        s_name = r.get("src_name", "").strip().lower()
        t_name = r.get("dst_name", "").strip().lower()

        if s_name == chair:
            per_chair.setdefault(s, []).append(r)
        if t_name == chair:
            per_chair.setdefault(t, []).append(r)

    return per_chair


def assign_neighbour_colours(neigh_ids: List[int]) -> Dict[int, np.ndarray]:
    """
    Deterministic colour assignment cycling over palette.
    """
    colours: Dict[int, np.ndarray] = {}
    for i, nid in enumerate(sorted(neigh_ids)):
        colours[nid] = _NEIGH_PALETTE[i % len(_NEIGH_PALETTE)]
    return colours


def export_ply(points: np.ndarray, colours01: np.ndarray, out_path: Path) -> None:
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points.astype(np.float32))
    pc.colors = o3d.utility.Vector3dVector(colours01.astype(np.float32))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(out_path), pc)


# ------------------------- Main routine ------------------------- #

def main() -> None:
    args = parse_args()
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Load inputs
    graph = load_graph(args.graph)
    clouds = load_clouds_merged(graph)  # list of per-node Nx3 arrays
    rows = read_labels_tsv(args.labels)

    # Build mapping chair_node_id -> list of TSV rows involving that chair
    chairs = build_chair_edge_index(rows, chair_word=args.chair_word)

    # For each chair node, collect neighbours and export PLY + LOG
    for chair_id, edge_rows in chairs.items():
        # Collect neighbours for this chair
        neigh_ids: List[int] = []
        # Also store all edges (for logging) keyed by neighbour id (can have multiple entries)
        edges_by_neigh: Dict[int, List[Dict[str, str]]] = {}

        for r in edge_rows:
            s = int(r["src_idx"]); t = int(r["dst_idx"])
            if s == chair_id:
                neigh = t
            else:
                neigh = s
            neigh_ids.append(neigh)
            edges_by_neigh.setdefault(neigh, []).append(r)

        # Unique neighbours and assign colours
        uniq_neighs = sorted(set(neigh_ids))
        neigh_colours = assign_neighbour_colours(uniq_neighs)

        # Build merged point cloud: chair + all neighbours
        parts_pts: List[np.ndarray] = []
        parts_cols: List[np.ndarray] = []

        # Chair points
        if chair_id < 0 or chair_id >= len(clouds):
            # Skip invalid ids
            continue
        chair_pts = np.asarray(clouds[chair_id], dtype=np.float32)
        parts_pts.append(chair_pts)
        parts_cols.append(np.tile(CHAIR_RGB, (chair_pts.shape[0], 1)))

        # Neighbour points
        for nid in uniq_neighs:
            if nid < 0 or nid >= len(clouds):
                continue
            n_pts = np.asarray(clouds[nid], dtype=np.float32)
            n_col = neigh_colours[nid]
            parts_pts.append(n_pts)
            parts_cols.append(np.tile(n_col, (n_pts.shape[0], 1)))

        if parts_pts:
            pts = np.concatenate(parts_pts, axis=0)
            cols = np.concatenate(parts_cols, axis=0).astype(np.float32) / 255.0
        else:
            pts = np.zeros((0, 3), dtype=np.float32)
            cols = np.zeros((0, 3), dtype=np.float32)

        # Export PLY and LOG
        stem = f"chair_{chair_id:04d}"
        ply_path = out_root / f"{stem}.ply"
        log_path = out_root / f"{stem}.txt"

        export_ply(pts, cols, ply_path)

        # Write log
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"# chair_id\t{chair_id}\n")
            f.write(f"# chair_rgb\t{CHAIR_RGB[0]} {CHAIR_RGB[1]} {CHAIR_RGB[2]}\n")
            f.write("# neighbour_id\tneigh_rgb\trelation")
            if args.include_caption:
                f.write("\tcaption")
            f.write("\n")

            for nid in uniq_neighs:
                rgb = neigh_colours[nid]
                # There could be multiple edges (rows) to the same neighbour; write them all
                for r in edges_by_neigh.get(nid, []):
                    rel = r.get("relation", "none")
                    f.write(f"{nid}\t{rgb[0]} {rgb[1]} {rgb[2]}\t{rel}")
                    if args.include_caption:
                        cap = (r.get("caption", "") or "").replace("\n", " ").strip()
                        f.write(f"\t{cap}")
                    f.write("\n")

        print(f"[OK] Wrote {ply_path.name} and {log_path.name}")

    print("[DONE]")


if __name__ == "__main__":
    main()
