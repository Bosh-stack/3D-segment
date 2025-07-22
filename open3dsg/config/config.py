# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
"""
Central path/config registry for Open3DSG.

This version is relaxed: it does NOT assert the existence of every legacy dataset
path (3RScan / ScanNet). It only enforces/creates what you actually use.

Env vars:
    OPEN3DSG_DATASETS   comma-separated list of datasets you will use (default: "myset")
"""
import os
import sys
from pathlib import Path
from easydict import EasyDict

CONF = EasyDict()
CONF.PATH = EasyDict()

# ----------------------------
# Base locations (edit these)
# ----------------------------
CONF.PATH.HOME = "/NetworkDocker"  # your home root
CONF.PATH.BASE = "/NetworkDocker/Open3DSG"  # repo root
CONF.PATH.DATA = (
    "/NetworkDocker/Data/Open3DSG_trainset"  # root for your custom dataset(s)
)

# Where to put all generated stuff (graphs, preproc caches, features, checkpoints)
CONF.PATH.DATA_OUT = os.path.join(CONF.PATH.BASE, "open3dsg/output")

# ----------------------------
# Legacy/official datasets (optional)
# ----------------------------
CONF.PATH.R3SCAN_RAW = os.path.join(CONF.PATH.DATA, "3RScan")
CONF.PATH.SCANNET_RAW = os.path.join(CONF.PATH.DATA, "SCANNET")
CONF.PATH.SCANNET_RAW3D = os.path.join(CONF.PATH.SCANNET_RAW, "scannet_3d", "data")
CONF.PATH.SCANNET_RAW2D = os.path.join(CONF.PATH.SCANNET_RAW, "scannet_2d")

# Processed (OpenSG/Open3DSG) versions – still optional
CONF.PATH.R3SCAN = os.path.join(CONF.PATH.DATA_OUT, "datasets", "OpenSG_3RScan")
CONF.PATH.SCANNET = os.path.join(CONF.PATH.DATA_OUT, "datasets", "OpenSG_ScanNet")

# Generic outputs
CONF.PATH.CHECKPOINTS = os.path.join(CONF.PATH.DATA_OUT, "checkpoints")
CONF.PATH.FEATURES = os.path.join(CONF.PATH.DATA_OUT, "features")

# ----------------------------
# Custom dataset: MYSET
# ----------------------------
CONF.PATH.MYSET_ROOT = "/NetworkDocker/Data/Open3DSG_trainset"
CONF.PATH.MYSET_CHECKPOINTS = os.path.join(
    CONF.PATH.MYSET_ROOT, "open3dsg/output/checkpoints"
)
CONF.PATH.MYSET_FEATURES_OUT = os.path.join(
    CONF.PATH.BASE, "open3dsg/output/features/myset"
)
CONF.PATH.MYSET_PREPROC_OUT = os.path.join(
    CONF.PATH.BASE, "open3dsg/output/preprocessed/myset"
)
CONF.PATH.MYSET_GRAPHS_OUT = os.path.join(
    CONF.PATH.BASE, "open3dsg/output/graphs/myset"
)

# ----------------------------
# MLOps (optional)
# ----------------------------
CONF.PATH.MLOPS = os.path.join(CONF.PATH.BASE, "mlops")
CONF.PATH.MLFLOW = os.path.join(CONF.PATH.MLOPS, "opensg", "mlflow")
CONF.PATH.TENSORBOARD = os.path.join(CONF.PATH.MLOPS, "opensg", "tensorboards")

# Add to sys.path
for _, p in CONF.PATH.items():
    if p and isinstance(p, str):
        sys.path.append(p)

# ----------------------------
# Helper: ensure dirs / skip asserts
# ----------------------------


def _ensure_dir(path_str: str):
    if path_str and not os.path.exists(path_str):
        Path(path_str).mkdir(parents=True, exist_ok=True)


# Which datasets do we care about?
USED_DATASETS = os.environ.get("OPEN3DSG_DATASETS", "myset").lower().split(",")

for key, path_str in CONF.PATH.items():
    if not path_str:  # empty string
        continue

    is_myset = "MYSET" in key
    # Output-ish dirs we can safely create
    is_output = any(
        tag in key
        for tag in ["OUT", "CHECKPOINT", "FEATURE", "PREPROC", "GRAPHS", "DATA_OUT"]
    )
    if is_output:
        _ensure_dir(path_str)
        continue

    # For dataset roots: enforce only if used
    if any(ds.upper() in key for ds in USED_DATASETS) or is_myset:
        if not os.path.exists(path_str):
            raise FileNotFoundError(
                f"{key}: {path_str} does not exist (and is required)."
            )
    else:
        # Just inform for unused ones
        if not os.path.exists(path_str):
            print(f"[INFO] Skipping unused path {key}: {path_str}")
