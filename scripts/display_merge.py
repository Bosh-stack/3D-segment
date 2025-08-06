#!/usr/bin/env python3
"""Visualise original and merged point clouds side by side."""

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


def _load_pc(path: Path, color=None):
    pc = o3d.io.read_point_cloud(str(path))
    if color is not None:
        pc.paint_uniform_color(color)
    return pc


def visualize(mask_a: Path, mask_b: Path, merged: Path):
    pc_a = _load_pc(mask_a, [1, 0, 0])
    pc_b = _load_pc(mask_b, [0, 1, 0])
    pc_m = _load_pc(merged, [0, 0, 1])

    # Compute offset so the clouds do not overlap
    bounds = np.vstack([
        pc_a.get_max_bound() - pc_a.get_min_bound(),
        pc_b.get_max_bound() - pc_b.get_min_bound(),
        pc_m.get_max_bound() - pc_m.get_min_bound(),
    ])
    offset = bounds.max()

    pc_a.translate([-1.5 * offset, 0, 0])
    pc_b.translate([0, 0, 0])
    pc_m.translate([1.5 * offset, 0, 0])

    o3d.visualization.draw_geometries([pc_a, pc_b, pc_m])


def main():
    ap = argparse.ArgumentParser(description="Visualise merge results")
    ap.add_argument("mask_a", type=Path, help="first input mask")
    ap.add_argument("mask_b", type=Path, help="second input mask")
    ap.add_argument("merged", type=Path, help="merged output mask")
    args = ap.parse_args()

    visualize(args.mask_a, args.mask_b, args.merged)


if __name__ == "__main__":
    main()
