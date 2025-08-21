#!/usr/bin/env python3
"""List objects without frame associations in preprocessed pickles."""

import argparse
import json
import pickle
from pathlib import Path

from open3dsg.config.config import CONF


def find_missing(root: Path):
    """Return mapping of pickle file to missing object ids."""
    missing = {}
    for pkl_path in root.rglob("*_object2frame.pkl"):
        with open(pkl_path, "rb") as fh:
            obj2frame = pickle.load(fh)
        empty = [oid for oid, frames in obj2frame.items() if not frames]
        if empty:
            missing[pkl_path] = empty
    return missing


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        type=Path,
        help="optional file to save the results as JSON",
    )
    args = ap.parse_args()

    root = Path(CONF.PATH.MYSET_PREPROC_OUT)
    missing = find_missing(root)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({str(k): v for k, v in missing.items()}, fh, indent=2)
    else:
        for pkl_path, ids in missing.items():
            print(pkl_path)
            for oid in ids:
                print(f"  {oid}")


if __name__ == "__main__":
    main()
