import argparse
import json
import os
import gc

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from open3dsg.config.config import CONF
from open3dsg.data.open_dataset import Open2D3DSGDataset
from open3dsg.scripts.feature_dumper import FeatureDumper


def _load_relationships(dataset: str):
    base = CONF.PATH.MYSET_GRAPHS_OUT if dataset.lower() == "myset" else CONF.PATH.SCANNET
    path = os.path.join(base, "subgraphs", "relationships_train.json")
    return json.load(open(path))["scans"]


def _build_dataset(args, load_features=None, skip_edge_features=False, load_node_features_only=False):
    relationships = _load_relationships(args.dataset)
    img_dim = 336 if args.node_model == "ViT-L/14@336px" else 224
    rel_img_dim = img_dim
    return Open2D3DSGDataset(
        relationships_R3SCAN=None,
        relationships_scannet=relationships,
        openseg=True,
        img_dim=img_dim,
        rel_img_dim=rel_img_dim,
        top_k_frames=args.top_k_frames,
        scales=args.scales,
        max_objects=args.max_nodes,
        max_rels=args.max_edges,
        load_features=load_features,
        blip=True,
        llava=False,
        skip_edge_features=skip_edge_features,
        load_node_features_only=load_node_features_only,
    )


def _compute_node_features(args):
    hparams = {
        "clip_model": "OpenSeg",
        "node_model": "ViT-L/14@336px",
        "edge_model": None,
        "dump_features": True,
        "skip_edge_features": True,
        "max_nodes": args.max_nodes,
        "max_edges": args.max_edges,
    }
    dumper = FeatureDumper(hparams)
    dumper.setup()
    dataset = _build_dataset(args, skip_edge_features=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate_fn)
    feature_dir = args.out_dir or dumper.clip_path
    for batch in tqdm(loader, desc="Nodes"):
        if torch.cuda.is_available() and "object_imgs" in batch:
            batch["object_imgs"] = batch["object_imgs"].cuda(non_blocking=True)
        with torch.no_grad():
            batch = dumper.encode_features(batch)
            emb_dim = batch["clip_obj_encoding"].shape[-1]
            bsz = batch["clip_obj_encoding"].shape[0]
            batch["clip_rel_encoding"] = torch.zeros(bsz, args.max_edges, emb_dim, device=batch["clip_obj_encoding"].device)
            dumper._dump_features(batch, bsz, path=feature_dir)
        del batch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    del loader, dataset, dumper
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return feature_dir


def _compute_edge_features(args, feature_dir):
    hparams = {
        "clip_model": "OpenSeg",
        "node_model": "ViT-L/14@336px",
        "edge_model": None,
        "dump_features": True,
        "skip_edge_features": False,
        "max_nodes": args.max_nodes,
        "max_edges": args.max_edges,
        "blip": True,
    }
    dumper = FeatureDumper(hparams)
    dumper.setup()
    dataset = _build_dataset(
        args, load_features=feature_dir, skip_edge_features=False, load_node_features_only=True
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate_fn)
    for batch in tqdm(loader, desc="Edges"):
        if torch.cuda.is_available() and "relationship_imgs" in batch and torch.is_tensor(batch["relationship_imgs"]):
            batch["relationship_imgs"] = batch["relationship_imgs"].cuda(non_blocking=True)
        if torch.cuda.is_available() and "blip_images" in batch and torch.is_tensor(batch["blip_images"]):
            batch["blip_images"] = batch["blip_images"].cuda(non_blocking=True)
        with torch.no_grad():
            batch = dumper.encode_features(batch)
            bsz = batch["clip_obj_encoding"].shape[0]
            dumper._dump_features(batch, bsz, path=feature_dir)
        del batch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    del loader, dataset, dumper
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _parse_args():
    parser = argparse.ArgumentParser(description="Precompute 2D features in two sequential stages")
    parser.add_argument("--dataset", default="scannet")
    parser.add_argument("--clip_model", choices=["OpenSeg"], default="OpenSeg")
    parser.add_argument("--node_model", default="ViT-L/14@336px")
    parser.add_argument("--top_k_frames", type=int, default=5)
    parser.add_argument("--scales", type=int, default=3)
    parser.add_argument("--max_nodes", type=int, default=1000)
    parser.add_argument("--max_edges", type=int, default=2000)
    parser.add_argument("--out_dir", default=None, help="directory to store features")
    args = parser.parse_args()
    return args


def main():
    args = _parse_args()
    feature_dir = _compute_node_features(args)
    _compute_edge_features(args, feature_dir)


if __name__ == "__main__":
    main()
