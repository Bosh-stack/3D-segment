#!/usr/bin/env python3
"""List object IDs without frame associations for a given scene."""

import argparse
import json
import pickle
from pathlib import Path


def find_missing(root: Path, scene: str):
    """Return object ids without frame associations for ``scene``."""
    pkl_path = root / f"{scene}_object2frame.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"Pickle file not found: {pkl_path}")

    with open(pkl_path, "rb") as fh:
        obj2frame = pickle.load(fh)
    return [oid for oid, frames in obj2frame.items() if not frames]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root",
        required=True,
        type=Path,
        help="directory containing <scene>_object2frame.pkl",
    )
    ap.add_argument("--scene", required=True, help="scene identifier")
    ap.add_argument(
        "--out",
        type=Path,
        help="optional file to save the results as JSON",
    )
    args = ap.parse_args()

    missing = find_missing(args.root, args.scene)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(missing, fh, indent=2)
    else:
        for oid in missing:
            print(oid)


if __name__ == "__main__":
    main()
