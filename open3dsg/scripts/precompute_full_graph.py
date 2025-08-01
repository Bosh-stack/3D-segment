import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from open3dsg.config.config import CONF


def run_cmd(cmd):
    print(" ".join(cmd))
    subprocess.check_call(cmd)


def latest_feature_dir(base: Path) -> Path:
    dirs = [d for d in base.iterdir() if d.is_dir() and d.name.startswith("clip_features_")]
    return max(dirs, key=lambda p: p.stat().st_mtime)


def precompute_full(split: str, out_dir: Path, extra_args: str = ""):
    subgraph_dir = Path(CONF.PATH.SCANNET) / "subgraphs"
    preproc_dir = Path(CONF.PATH.SCANNET) / "preprocessed"

    run_cmd(["python", "open3dsg/data/gen_scannet_subgraphs.py", "--type", split])
    run_cmd(["python", "open3dsg/data/preprocess_scannet.py"])

    dump_cmd = ["python", "open3dsg/scripts/run.py", "--dump_features", "--dataset", "scannet"]
    if extra_args:
        dump_cmd.extend(extra_args.split())
    run_cmd(dump_cmd)

    feat_dir = latest_feature_dir(Path(CONF.PATH.FEATURES))
    merge_cmd = [
        "python",
        "open3dsg/data/merge_subgraphs.py",
        "--split",
        split,
        "--features_dir",
        str(feat_dir),
        "--out_dir",
        str(out_dir),
        "--preproc_dir",
        str(preproc_dir),
        "--subgraph_dir",
        str(subgraph_dir),
    ]
    run_cmd(merge_cmd)


def main():
    parser = argparse.ArgumentParser(description="Precompute full graphs")
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"])
    parser.add_argument("--out_dir", required=True, help="directory to store merged features")
    parser.add_argument("--extra_args", default="", help="additional args passed to run.py")
    args = parser.parse_args()

    precompute_full(args.split, Path(args.out_dir), args.extra_args)


if __name__ == "__main__":
    main()
