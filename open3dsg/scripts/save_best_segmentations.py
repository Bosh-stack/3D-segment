#!/usr/bin/env python
# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

"""Save best-view segmentation masks for each object.

This script selects the most visible frame for every object in a scan and
saves its 2D segmentation mask as a binary PNG image. The mask for object
``i`` from scan ``<scan_id>`` is written to ``<out_dir>/<scan_id>/<i>.png``.

Usage
-----
Run the script from the repository root:

```
python open3dsg/scripts/save_best_segmentations.py \
    --dataset scannet \
    --batch_size 4 \
    --top_k_frames 5 \
    --out_dir /tmp/masks \
    [--gpu 0]
```
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch

from open3dsg.config.config import CONF
from open3dsg.data.open_dataset import Open2D3DSGDataset


BLANK_IMG_DIM = (240, 320)


def _load_relationships(dataset: str):
    base = CONF.PATH.MYSET_GRAPHS_OUT if dataset.lower() == "myset" else CONF.PATH.SCANNET
    path = os.path.join(base, "subgraphs", "relationships_train.json")
    with open(path) as f:
        return json.load(f)["scans"]


def _build_dataset(args):
    relationships = _load_relationships(args.dataset)
    return Open2D3DSGDataset(
        relationships_R3SCAN=None,
        relationships_scannet=relationships,
        openseg=True,
        img_dim=224,
        rel_img_dim=224,
        top_k_frames=args.top_k_frames,
        scales=1,
        max_objects=1000,
        max_rels=1,
        skip_edge_features=True,
    )


def _parse_args():
    parser = argparse.ArgumentParser(description="Export best-frame segmentation masks.")
    parser.add_argument("--dataset", default="scannet", help="dataset to load")
    parser.add_argument("--batch_size", type=int, default=1, help="number of scans per batch")
    parser.add_argument("--top_k_frames", type=int, default=5, help="number of candidate frames per object")
    parser.add_argument("--out_dir", required=True, help="directory to store PNG masks")
    parser.add_argument("--gpu", type=int, default=0, help="index of GPU to use")
    return parser.parse_args()


def main():
    args = _parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    dataset = _build_dataset(args)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.collate_fn)

    out_root = Path(args.out_dir)
    for batch in tqdm(loader, desc="Scans"):
        # move tensors to selected device
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        bsz = len(batch["scene_id"])
        for b in range(bsz):
            scan_id = batch["scene_id"][b]
            obj_pixels = batch["object_pixels"][b]
            obj_count = int(batch["objects_count"][b].item())
            scan_dir = out_root / scan_id
            scan_dir.mkdir(parents=True, exist_ok=True)
            for obj_id in range(obj_count):
                pixels = obj_pixels[obj_id][0]
                mask = np.zeros(BLANK_IMG_DIM, dtype=np.uint8)
                if pixels.ndim == 2 and pixels.size:
                    mask[pixels[:, 0], pixels[:, 1]] = 255
                Image.fromarray(mask).save(scan_dir / f"{obj_id}.png")


if __name__ == "__main__":
    main()
