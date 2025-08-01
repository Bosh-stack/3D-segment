import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from open3dsg.models.sgpn import SGPN
from open3dsg.data.open_dataset import Open2D3DSGDataset
from open3dsg.config.config import CONF


def get_vram():
    """Return total and per GPU memory usage in MB."""
    torch.cuda.synchronize()
    per_gpu = [
        torch.cuda.memory_allocated(i) / 1024**2
        for i in range(torch.cuda.device_count())
    ]
    total = sum(per_gpu)
    return total, per_gpu


def record_vram(tag: str, results_list):
    total, per_gpu = get_vram()
    results_list.append((tag, total, per_gpu))
    per_gpu_str = ", ".join(
        f"gpu{idx}: {mem:.2f} MB" for idx, mem in enumerate(per_gpu)
    )
    if len(per_gpu) > 1:
        print(f"{tag}: {total:.2f} MB ({per_gpu_str})")
    else:
        print(f"{tag}: {total:.2f} MB")


def print_summary(results_list, header="VRAM Usage Summary"):
    print(f"\n---- {header} ----")
    for tag, total, per_gpu in results_list:
        per_gpu_str = ", ".join(
            f"gpu{idx}: {mem:.2f} MB" for idx, mem in enumerate(per_gpu)
        )
        if len(per_gpu) > 1:
            print(f"{tag}: {total:.2f} MB ({per_gpu_str})")
        else:
            print(f"{tag}: {total:.2f} MB")


def load_relationships(dataset: str):
    if dataset.lower() == "myset":
        path = os.path.join(CONF.PATH.MYSET_GRAPHS_OUT, "graphs", "train.json")
    else:
        path = os.path.join(CONF.PATH.SCANNET, "subgraphs", "relationships_train.json")
    return json.load(open(path, "r"))["scans"]


def profile_gpu(device_id: int, args, hparams):
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    results = []

    model = SGPN(hparams).to(device)

    record_vram("Base model", results)

    target_model = model
    if not args.clean_pointnet and not target_model.rgb and not target_model.nrm:
        target_model.load_pretained_cls_model(target_model.objPointNet)
        target_model.load_pretained_cls_model(target_model.relPointNet)
        record_vram("PointNet weights", results)

    if not args.load_features:
        if args.clip_model == "OpenSeg":
            target_model.OPENSEG = target_model.load_pretrained_clip_model(target_model.OPENSEG, args.clip_model)
        else:
            target_model.CLIP = target_model.load_pretrained_clip_model(target_model.CLIP, args.clip_model)
        record_vram(f"CLIP ({args.clip_model})", results)

        if args.node_model:
            target_model.CLIP_NODE = target_model.load_pretrained_clip_model(target_model.CLIP_NODE, args.node_model)
            record_vram(f"Node model ({args.node_model})", results)

        if args.edge_model:
            target_model.CLIP_EDGE = target_model.load_pretrained_clip_model(target_model.CLIP_EDGE, args.edge_model)
            record_vram(f"Edge model ({args.edge_model})", results)

        if args.blip:
            if args.dump_features:
                target_model.load_pretrained_blipvision_model()
            else:
                target_model.load_pretrained_blip_model()
            record_vram("BLIP model", results)

        if args.llava:
            target_model.load_pretrained_llava_model()
            record_vram("LLaVA model", results)

    scans = load_relationships(args.dataset)
    dataset = Open2D3DSGDataset(
        relationships_R3SCAN=None,
        relationships_scannet=scans,
        openseg=args.clip_model == "OpenSeg",
        img_dim=336 if args.clip_model == "ViT-L/14@336px" else 224,
        rel_img_dim=336 if (args.edge_model == "ViT-L/14@336px") else None,
        top_k_frames=args.top_k_frames,
        scales=args.scales,
        max_objects=args.max_nodes,
        max_rels=args.max_edges,
        load_features=args.load_features,
        blip=args.blip,
        llava=args.llava,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )
    batch = next(iter(loader))
    batch_gpu = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
    torch.cuda.synchronize()
    record_vram("First batch moved to GPU", results)

    with torch.no_grad():
        if args.dump_features:
            if args.clip_model == "OpenSeg":
                _ = model.clip_encode_pixels(
                    batch_gpu["object_raw_imgs"],
                    batch_gpu["object_pixels"],
                    batch_gpu["objects_count"],
                    batch_gpu["relationship_imgs"],
                )
            else:
                _ = model.clip_encode_imgs(batch_gpu["object_imgs"], batch_gpu["relationship_imgs"])
            if args.blip:
                _ = model.blip_encode_images(batch_gpu["blip_images"])
            elif args.llava:
                _ = model.llava_encode_images(batch_gpu["blip_images"])
        else:
            _ = model(batch_gpu)
    record_vram("After inference", results)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Profile GPU VRAM usage of Open3DSG models"
    )
    parser.add_argument(
        "--dataset", default="myset", help="dataset to load [myset|scannet]"
    )
    parser.add_argument(
        "--clip_model",
        default="OpenSeg",
        choices=["ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px", "OpenSeg"],
    )
    parser.add_argument("--node_model", default=None)
    parser.add_argument("--edge_model", default=None)
    parser.add_argument("--blip", action="store_true")
    parser.add_argument("--llava", action="store_true")
    parser.add_argument('--avg_blip_emb', action='store_true',
                        help='Average the blip embeddings across patches')
    parser.add_argument('--blip_proj_layers', type=int, default=3,
                        help='Number of projection layers to match blip embedding')
    parser.add_argument('--avg_llava_emb', action='store_true',
                        help='Average the llava embeddings across patches')
    parser.add_argument(
        "--dump_features",
        action="store_true",
        help="profile 2D feature precomputation mode",
    )
    parser.add_argument(
        "--load_features", default=None, help="path to precomputed 2D features"
    )
    parser.add_argument("--pointnet2", action="store_true")
    parser.add_argument("--clean_pointnet", action="store_true")
    parser.add_argument("--top_k_frames", type=int, default=5)
    parser.add_argument("--scales", type=int, default=3)
    parser.add_argument("--max_nodes", type=int, default=10)
    parser.add_argument("--max_edges", type=int, default=100)
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="number of GPUs to use (DataParallel)",
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default=None,
        help="comma separated list of GPU ids (overrides --gpus)",
    )
    args = parser.parse_args()

    hparams = {
        "use_rgb": False,
        "gnn_layers": 4,
        "gconv_dim": 512,
        "hidden_dim": 1024,
        "clip_model": args.clip_model,
        "node_model": args.node_model,
        "edge_model": args.edge_model,
        "blip": args.blip,
        "llava": args.llava,
        "avg_blip_emb": args.avg_blip_emb,
        "blip_proj_layers": args.blip_proj_layers,
        "avg_llava_emb": args.avg_llava_emb,
        "pointnet2": args.pointnet2,
        "clean_pointnet": args.clean_pointnet,
        "max_nodes": args.max_nodes,
        "max_edges": args.max_edges,
        "load_features": args.load_features,
        "dump_features": args.dump_features,
        "test": False,
    }

    if args.gpu_ids:
        device_ids = [int(x) for x in args.gpu_ids.split(',')]
    else:
        device_ids = list(range(min(args.gpus, torch.cuda.device_count())))

    for dev in device_ids:
        print(f"\nProfiling on GPU {dev}")
        results = profile_gpu(dev, args, hparams)
        print_summary(results, header=f"GPU {dev} VRAM Usage")


if __name__ == "__main__":
    main()
