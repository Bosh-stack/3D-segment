import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from open3dsg.models.sgpn import SGPN
from open3dsg.data.open_dataset import Open2D3DSGDataset
from open3dsg.config.config import CONF


def print_vram(tag: str):
    torch.cuda.synchronize()
    mem = torch.cuda.memory_allocated() / 1024 ** 2
    print(f"{tag}: {mem:.2f} MB")


def load_relationships(dataset: str):
    if dataset.lower() == "myset":
        path = os.path.join(CONF.PATH.MYSET_GRAPHS_OUT, "graphs", "train.json")
    else:
        path = os.path.join(CONF.PATH.SCANNET, "subgraphs", "relationships_train.json")
    return json.load(open(path, "r"))["scans"]


def main():
    parser = argparse.ArgumentParser(description="Profile GPU VRAM usage of Open3DSG models")
    parser.add_argument("--dataset", default="myset", help="dataset to load [myset|scannet]")
    parser.add_argument("--clip_model", default="OpenSeg",
                        choices=['ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px', 'OpenSeg'])
    parser.add_argument("--node_model", default=None)
    parser.add_argument("--edge_model", default=None)
    parser.add_argument("--blip", action="store_true")
    parser.add_argument("--llava", action="store_true")
    parser.add_argument("--pointnet2", action="store_true")
    parser.add_argument("--clean_pointnet", action="store_true")
    parser.add_argument("--top_k_frames", type=int, default=5)
    parser.add_argument("--scales", type=int, default=3)
    parser.add_argument("--max_nodes", type=int, default=10)
    parser.add_argument("--max_edges", type=int, default=100)
    args = parser.parse_args()

    hparams = {
        'use_rgb': False,
        'gnn_layers': 4,
        'gconv_dim': 512,
        'hidden_dim': 1024,
        'clip_model': args.clip_model,
        'node_model': args.node_model,
        'edge_model': args.edge_model,
        'blip': args.blip,
        'llava': args.llava,
        'pointnet2': args.pointnet2,
        'clean_pointnet': args.clean_pointnet,
        'max_nodes': args.max_nodes,
        'max_edges': args.max_edges,
        'load_features': None,
        'test': False,
    }

    model = SGPN(hparams).cuda()
    print_vram("Base model")

    if not args.clean_pointnet and not model.rgb and not model.nrm:
        model.load_pretained_cls_model(model.objPointNet)
        model.load_pretained_cls_model(model.relPointNet)
        print_vram("PointNet weights")

    model.CLIP = model.load_pretrained_clip_model(model.CLIP, args.clip_model)
    print_vram(f"CLIP ({args.clip_model})")

    if args.node_model:
        model.CLIP_NODE = model.load_pretrained_clip_model(model.CLIP_NODE, args.node_model)
        print_vram(f"Node model ({args.node_model})")

    if args.edge_model:
        model.CLIP_EDGE = model.load_pretrained_clip_model(model.CLIP_EDGE, args.edge_model)
        print_vram(f"Edge model ({args.edge_model})")

    if args.blip:
        model.load_pretrained_blip_model()
        print_vram("BLIP model")

    if args.llava:
        model.load_pretrained_llava_model()
        print_vram("LLaVA model")

    # Load a small batch of data to measure VRAM
    try:
        scans = load_relationships(args.dataset)
        dataset = Open2D3DSGDataset(relationships_R3SCAN=None,
                                     relationships_scannet=scans,
                                     openseg=args.clip_model == 'OpenSeg',
                                     img_dim=336 if args.clip_model == 'ViT-L/14@336px' else 224,
                                     rel_img_dim=336 if (args.edge_model == 'ViT-L/14@336px') else None,
                                     top_k_frames=args.top_k_frames,
                                     scales=args.scales,
                                     max_objects=args.max_nodes,
                                     max_rels=args.max_edges,
                                     blip=args.blip,
                                     llava=args.llava)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate_fn)
        batch = next(iter(loader))
        batch_gpu = {k: v.cuda(non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
        torch.cuda.synchronize()
        print_vram("First batch moved to GPU")
    except Exception as e:
        print(f"Could not load dataset or move batch to GPU: {e}")


if __name__ == "__main__":
    main()
