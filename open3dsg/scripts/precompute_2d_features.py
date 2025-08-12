import argparse
import gc
import json
import os
import shutil

import torch
from lightning_fabric import Fabric
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from open3dsg.config.config import CONF
from open3dsg.data.open_dataset import Open2D3DSGDataset
from open3dsg.scripts.feature_dumper import FeatureDumper


def _load_relationships(dataset: str):
    base = CONF.PATH.MYSET_GRAPHS_OUT if dataset.lower() == "myset" else CONF.PATH.SCANNET
    path = os.path.join(base, "subgraphs", "relationships_train.json")
    try:
        with open(path) as f:
            return json.load(f)["scans"]
    except Exception as e:  # pragma: no cover - diagnostic output
        print(f"Failed to load relationships file: {path} ({e})")
        raise


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




def _parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute 2D features in two sequential stages. Optionally load precomputed node features."
    )
    parser.add_argument("--dataset", default="scannet")
    parser.add_argument("--clip_model", choices=["OpenSeg"], default="OpenSeg")
    parser.add_argument("--node_model", default="ViT-L/14@336px")
    parser.add_argument("--top_k_frames", type=int, default=5)
    parser.add_argument("--scales", type=int, default=3)
    parser.add_argument("--max_nodes", type=int, default=1000)
    parser.add_argument("--max_edges", type=int, default=2000)
    parser.add_argument("--clip_batch_size", type=int, default=64,
                        help="Micro-batch size for CLIP/OpenSeg image encodes")
    parser.add_argument("--blip_batch_size", type=int, default=32,
                        help="Micro-batch size for BLIP image encodes")
    parser.add_argument("--amp", action="store_true",
                        help="Enable torch.cuda.amp autocast during encodes")
    parser.add_argument("--out_dir", default=None, help="directory to store features")
    parser.add_argument(
        "--load_node_features",
        default=None,
        help="directory containing precomputed node features; if provided, node feature computation is skipped",
    )
    parser.add_argument("--gpus", type=int, default=torch.cuda.device_count())
    args = parser.parse_args()
    return args

def main_worker(fabric: Fabric, args):
    rank_dir = str(fabric.global_rank)

    # Stage 1: compute node features unless provided
    if args.load_node_features:
        feature_root = args.load_node_features
        node_feature_dir = feature_root
    else:
        node_hparams = {
            "clip_model": "OpenSeg",
            "node_model": "ViT-L/14@336px",
            "edge_model": None,
            "dump_features": True,
            "skip_edge_features": True,
            "max_nodes": args.max_nodes,
            "max_edges": args.max_edges,
            "clip_batch_size": args.clip_batch_size,
            "blip_batch_size": args.blip_batch_size,
            "amp": args.amp,
            "top_k_frames": args.top_k_frames,
            "scales": args.scales,
        }
        dumper = FeatureDumper(node_hparams, device=fabric.local_rank)
        dumper.setup()
        dataset = _build_dataset(args, skip_edge_features=True)
        sampler = DistributedSampler(
            dataset,
            num_replicas=fabric.world_size,
            rank=fabric.global_rank,
            shuffle=False,
        )
        loader = DataLoader(
            dataset, batch_size=1, sampler=sampler, collate_fn=dataset.collate_fn
        )
        loader = fabric.setup_dataloaders(loader)
        feature_root = args.out_dir or dumper.clip_path
        node_feature_dir = os.path.join(feature_root, rank_dir)
        os.makedirs(node_feature_dir, exist_ok=True)

        # Diagnostic: show which files this GPU will process
        obj_clip_model = (
            node_hparams["node_model"]
            if node_hparams.get("node_model") and node_hparams["clip_model"] != "OpenSeg"
            else node_hparams["clip_model"]
        )
        obj_path = os.path.join(
            node_feature_dir,
            "export_obj_clip_emb_clip_" + obj_clip_model.replace("/", "-"),
        )
        for idx in list(sampler):
            scan_id = dataset.scene_data[idx].scan_id
            print(
                f"GPU {fabric.global_rank} -> "
                f"{os.path.join(obj_path, scan_id + '.pt')}"
            )

        for batch in tqdm(loader, desc="Nodes"):
            with torch.no_grad():
                batch = dumper.encode_features(batch)
                bsz = batch["clip_obj_encoding"].shape[0]
                dumper._dump_features(batch, bsz, path=node_feature_dir)
            del batch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        del loader, dataset, dumper
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    # Stage 2: compute edge features
    edge_hparams = {
        "clip_model": "OpenSeg",
        "node_model": "ViT-L/14@336px",
        "edge_model": None,
        "dump_features": True,
        "skip_edge_features": False,
        "max_nodes": args.max_nodes,
        "max_edges": args.max_edges,
        "blip": True,
        "clip_batch_size": args.clip_batch_size,
        "blip_batch_size": args.blip_batch_size,
        "amp": args.amp,
        "top_k_frames": args.top_k_frames,
        "scales": args.scales,
    }
    dumper = FeatureDumper(edge_hparams, device=fabric.local_rank)
    dumper.setup()
    edge_dir = os.path.join(feature_root, rank_dir)
    os.makedirs(edge_dir, exist_ok=True)
    dataset_load_path = node_feature_dir if not args.load_node_features else feature_root
    dataset = _build_dataset(
        args,
        load_features=dataset_load_path,
        skip_edge_features=False,
        load_node_features_only=True,
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=fabric.world_size,
        rank=fabric.global_rank,
        shuffle=False,
    )
    loader = DataLoader(
        dataset, batch_size=1, sampler=sampler, collate_fn=dataset.collate_fn
    )
    loader = fabric.setup_dataloaders(loader)
    for batch in tqdm(loader, desc="Edges"):
        with torch.no_grad():
            batch = dumper.encode_features(batch)
            bsz = batch["clip_obj_encoding"].shape[0]
            dumper._dump_features(batch, bsz, path=edge_dir)
        del batch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    del loader, dataset, dumper
    gc.collect()

    fabric.barrier()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if fabric.global_rank == 0:
        for r in range(fabric.world_size):
            r_path = os.path.join(feature_root, str(r))
            if not os.path.isdir(r_path):
                continue
            for item in os.listdir(r_path):
                src = os.path.join(r_path, item)
                dst = os.path.join(feature_root, item)
                if os.path.isdir(src):
                    os.makedirs(dst, exist_ok=True)
                    for f in os.listdir(src):
                        shutil.move(os.path.join(src, f), os.path.join(dst, f))
                else:
                    shutil.move(src, dst)
            shutil.rmtree(r_path)

    fabric.barrier()


def main():
    args = _parse_args()
    fabric = Fabric(accelerator="cuda", devices=args.gpus)
    fabric.launch(main_worker, args)


if __name__ == "__main__":
    main()
