#!/usr/bin/env python3
"""
Convert Open3DSG preprocessed graph + edge relations into an LLM-friendly JSON.

Nodes (objects) — minimal fields you have:
  id (int), object_tag (string), bbox_center [x,y,z], bbox_extent [dx,dy,dz]

Edges (relationships):
  source_id (int), target_id (int), relation (string), caption (string)

Example output:
{
  "objects": [
    {"id": 12, "object_tag": "chair", "bbox_center": [x,y,z], "bbox_extent": [dx,dy,dz]},
    ...
  ],
  "relationships": [
    {"source_id": 12, "target_id": 7, "relation": "next to", "caption": "chair is next to desk"},
    ...
  ]
}
"""

import argparse, json, pickle, csv, sys

def _get(obj, *keys, default=None):
    for k in keys:
        if k in obj:
            return obj[k]
    return default

def load_nodes_from_pkl(graph_pkl):
    """
    Expect a dict that contains a 'nodes' list with per-object dicts.
    Tries common key aliases to be robust across Open3DSG scripts.
    """
    with open(graph_pkl, "rb") as f:
        data = pickle.load(f)

    nodes_in = data.get("nodes", data.get("objects", []))
    nodes_out = []

    for n in nodes_in:
        # id
        nid = _get(n, "instance_id", "id")
        if nid is None:
            raise KeyError("Node missing 'instance_id'/'id'")

        # class/tag name
        tag = _get(n, "label", "name", "class", "object_tag")
        if tag is None:
            # fallback to empty string if unknown; you can post-fill later
            tag = ""

        # center/extent
        center = _get(n, "center", "bbox_center", default=None)
        extent = _get(n, "extent", "bbox_extent", default=None)

        # Some preprocessors store a full bbox; add your own parsing here if needed.
        out = {
            "id": int(nid),
            "object_tag": tag,
        }
        if center is not None: out["bbox_center"] = list(map(float, center))
        if extent is not None: out["bbox_extent"] = list(map(float, extent))

        nodes_out.append(out)

    return nodes_out

def load_edges_from_tsv(edge_tsv):
    """
    TSV with headers: src_id, tgt_id, relation, caption
    'caption' optional: if missing, set empty string.
    """
    edges = []
    with open(edge_tsv, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        need = {"src_id", "tgt_id", "relation"}
        if not need.issubset(set(reader.fieldnames or [])):
            raise KeyError(f"TSV must include columns: {need}")

        for row in reader:
            edges.append({
                "source_id": int(row["src_id"]),
                "target_id": int(row["tgt_id"]),
                "relation": row["relation"],
                "caption": row.get("caption", "") or ""
            })
    return edges

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph_pkl", required=True, help="e.g., open3dsg/output/preprocessed/scan0/data_dict_0.pkl")
    ap.add_argument("--edge_tsv", required=True, help="e.g., edge_relations.tsv (with relation + caption)")
    ap.add_argument("--out_json", required=True, help="Output JSON path")
    args = ap.parse_args()

    objects = load_nodes_from_pkl(args.graph_pkl)
    relationships = load_edges_from_tsv(args.edge_tsv)
    scene = {"objects": objects, "relationships": relationships}

    with open(args.out_json, "w") as f:
        json.dump(scene, f, indent=2)
    print(f"Wrote {args.out_json}  | objects={len(objects)}  relationships={len(relationships)}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e, file=sys.stderr)
        sys.exit(1)
