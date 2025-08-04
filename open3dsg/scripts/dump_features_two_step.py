# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

"""Dump CLIP features in two stages.

Usage:
    python open3dsg/scripts/dump_features_two_step.py --stage nodes ...
    python open3dsg/scripts/dump_features_two_step.py --stage edges --load_node_features <dir> ...
"""

import argparse
import json
import os
import random
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from open3dsg.config.config import CONF
from open3dsg.data.open_dataset import Open2D3DSGDataset
from open3dsg.scripts.feature_dumper import FeatureDumper


def get_args():
    parser = argparse.ArgumentParser()

    # system params
    parser.add_argument('--gpus', type=int, default=-1)
    parser.add_argument("--workers", type=int, default=8, help="number of workers per gpu")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--run_name", type=str, help="dir name for tensorboard and checkpoints")
    parser.add_argument('--mixed_precision', action="store_true", help="Use mixed precision training")

    # optimizer params (unused but kept for compatibility)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument("--epochs", type=int, help="number of epochs", default=50)
    parser.add_argument("--batch_size", type=int, help="batch size", default=1)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="reduce", help="lr_scheduler, options [cyclic, reduce]")
    parser.add_argument("--wd", type=float, help="weight decay", default=1e-2)
    parser.add_argument("--bn_momentum", type=float, default=0.9, help="Initial batch norm momentum [default: 0.9]")
    parser.add_argument("--bn_decay", type=float, default=0.5,   help="Batch norm momentum decay gamma [default: 0.5]")
    parser.add_argument("--decay_step", type=float, default=1e5, help="Learning rate decay step [default: 20]",)

    parser.add_argument('--w_obj', type=float, default=1.0)
    parser.add_argument('--w_rel', type=float, default=1.0)

    # model params
    parser.add_argument('--use_rgb', action="store_true", help="Whether to use rgb features as input the the point net")
    parser.add_argument("--gnn_layers", type=int, default=4, help="number of gnn layers")
    parser.add_argument('--graph_backbone', default="message", nargs='?',
                        choices=['message', 'attention', 'transformer', 'mlp'])
    parser.add_argument('--gconv_dim', type=int, default=512, help='embedding dim for point features')
    parser.add_argument('--hidden_dim', type=int, default=1024, help='hidden dim for graph_convs')
    parser.add_argument('--max_nodes', type=int, default=100, help='max number of nodes in the graph')
    parser.add_argument('--max_edges', type=int, default=800,
                        help='max number of edges in the graph. Should correspond to n*(n-1) nodes')

    # data params
    parser.add_argument('--dataset', default='scannet', help="['scannet']")
    parser.add_argument('--mini_dataset', action='store_true',
                        help="only load a tiny fraction of data for faster debugging")
    parser.add_argument('--augment', action="store_true",
                        help="use basic pcl augmentations that do not collide with scene graph properties")
    parser.add_argument("--top_k_frames", type=int, default=5, help="number of frames to consider for each instance")
    parser.add_argument("--scales", type=int, default=3, help="number of scales for each selected image")
    parser.add_argument('--dump_features', action="store_true", help="precompute 2d features and dump to disk")
    parser.add_argument('--load_features', default=None, help="path to precomputed 2d features")
    parser.add_argument('--skip_edge_features', action='store_true',
                        help='Skip relation image feature computation')

    # model variations params
    parser.add_argument('--clip_model', default="OpenSeg", type=str,
                        choices=['ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px', 'OpenSeg'])
    parser.add_argument('--node_model', default='ViT-L/14@336px', type=str,
                        choices=['ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'])
    parser.add_argument('--edge_model', default=None, type=str,
                        choices=['ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'])
    parser.add_argument('--blip', action="store_true", help="Use blip for relation prediction")
    parser.add_argument('--avg_blip_emb', action='store_true', help="Average the blip embeddings across patches")
    parser.add_argument('--blip_proj_layers', type=int, default=3,
                        help="Number of projection layers to match blip embedding")
    parser.add_argument('--blip_batch_size', type=int, default=32, help='Batch size for BLIP image encoding')
    parser.add_argument('--llava', action="store_true", help="Use llava for relation prediction")
    parser.add_argument('--avg_llava_emb', action="store_true", help="Average the llava embeddings across patches")
    parser.add_argument('--pointnet2', action="store_true",
                        help="Use pointnet++ for feature extraction. However RGB input not working")
    parser.add_argument("--clean_pointnet", action="store_true",
                        help="standard pretrained pointnet for feature extraction")
    parser.add_argument('--supervised_edges', action="store_true", help="Train edges supervised instead of open-vocab")

    # eval params (unused but kept for compatibility)
    parser.add_argument("--test", action="store_true", help="test the model")
    parser.add_argument("--checkpoint", type=str, help="Specify the checkpoint root", default=None)
    parser.add_argument('--weight_2d', type=float, default=0.5, help="2d-3d feature fusion weight")
    parser.add_argument('--n_beams', type=int, default=5, help="number of beams for beam search in LLM output")
    parser.add_argument('--gt_objects', action="store_true", help="Use GT objects for predicate prediction")
    parser.add_argument('--vis_graphs', action="store_true", help="save graph predictions to disk")
    parser.add_argument('--predict_from_2d', action="store_true", help="predict only using 2d models")
    parser.add_argument('--quick_eval', action='store_true', help="only eval on a few samples")
    parser.add_argument('--object_context', action="store_true", help="prompt clip with: A [object] in a scene")
    parser.add_argument('--update_hparams', action="store_true", help="update hparams from checkpoint")
    parser.add_argument('--manual_mapping', action="store_true", help="Manually map some known predicates to GT")

    # two step params
    parser.add_argument('--stage', required=True, choices=['nodes', 'edges'])
    parser.add_argument('--load_node_features', type=str,
                        help='directory containing precomputed node features')

    return parser.parse_args()


def build_dataset(args, load_features, skip_edge_features, load_node_features_only=False):
    def load_scan(base_path, file_path):
        return json.load(open(os.path.join(base_path, file_path)))['scans']

    dataset_name = args.dataset.lower()
    if dataset_name == 'myset':
        base_path = CONF.PATH.MYSET_GRAPHS_OUT
    else:
        base_path = CONF.PATH.SCANNET

    relationships_scannet = load_scan(base_path, "subgraphs/relationships_train.json")

    img_dim = 336 if args.clip_model == 'ViT-L/14@336px' else 224
    rel_img_dim = img_dim
    if args.edge_model:
        rel_img_dim = 336 if args.edge_model == 'ViT-L/14@336px' else 224

    dataset = Open2D3DSGDataset(
        relationships_R3SCAN=None,
        relationships_scannet=relationships_scannet,
        openseg=args.clip_model == 'OpenSeg',
        img_dim=img_dim,
        rel_img_dim=rel_img_dim,
        top_k_frames=args.top_k_frames,
        scales=args.scales,
        mini=args.mini_dataset,
        load_features=load_features,
        blip=args.blip,
        llava=args.llava,
        max_objects=args.max_nodes,
        max_rels=args.max_edges,
        skip_edge_features=skip_edge_features,
        load_node_features_only=load_node_features_only,
    )
    return dataset


def main():
    args = get_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    hparams = vars(args).copy()
    hparams['dump_features'] = True
    hparams['load_features'] = None

    if args.stage == 'nodes':
        hparams['skip_edge_features'] = True
    else:
        hparams['skip_edge_features'] = False
        hparams['load_node_features_only'] = True

    module = FeatureDumper(hparams)
    module.setup()
    module.model.eval()

    if args.stage == 'nodes':
        dataset = build_dataset(args, None, True)
        feature_dir = module.clip_path
    else:
        if args.load_node_features is None:
            raise ValueError("--load_node_features required for edges stage")
        dataset = build_dataset(args, args.load_node_features, False, load_node_features_only=True)
        feature_dir = args.load_node_features

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=dataset.collate_fn,
    )

    device = module.model.clip_device if torch.cuda.is_available() else torch.device('cpu')

    with tqdm(total=len(dataset), desc="Processing scenes") as pbar:
        for data_dict in dataloader:
            for k, v in data_dict.items():
                if isinstance(v, torch.Tensor):
                    data_dict[k] = v.to(device)

            if args.stage == 'nodes':
                data_dict = module._forward(data_dict)
            else:
                if args.blip:
                    data_dict['clip_rel_encoding'] = module.model.blip_encode_images(
                        data_dict['blip_images'], batch_size=args.blip_batch_size
                    )
                elif args.llava:
                    data_dict['clip_rel_encoding'] = module.model.llava_encode_images(data_dict['blip_images'])
                else:
                    dummy = torch.zeros(
                        data_dict['objects_id'].size(0), 1, 1, 3,
                        dataset.img_dim, dataset.img_dim, device=device
                    )
                    rel_imgs = data_dict['relationship_imgs'].to(device)
                    _, rel_feats = module.model.clip_encode_imgs(dummy, rel_imgs)
                    data_dict['clip_rel_encoding'] = rel_feats

            module._dump_features(data_dict, data_dict['objects_id'].size(0), path=feature_dir)
            pbar.update(len(data_dict.get('scan_id', [])))

    print(f"Features saved to {feature_dir}")


if __name__ == "__main__":
    main()

