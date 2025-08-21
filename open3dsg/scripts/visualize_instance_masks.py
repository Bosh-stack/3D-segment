"""Visualise 2D instance masks on RGB frames.

This helper reads ``*_object2frame.pkl`` files produced by
:mod:`open3dsg.data.get_object_frame_myset` and saves overlays of the
stored pixel indices onto their corresponding images.

Example
-------
```bash
python open3dsg/scripts/visualize_instance_masks.py \
    --scan_dir path/to/scan \
    --object2frame scan_id_object2frame.pkl \
    --out masks --top_k 3
```
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay instance masks on their corresponding RGB frames",
    )
    parser.add_argument("--scan_dir", required=True, help="directory with RGB images")
    parser.add_argument(
        "--object2frame",
        required=True,
        help="pickle file mapping instances to frame information",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="output directory for visualisations",
    )
    parser.add_argument(
        "--top_k", type=int, default=None, help="limit frames per instance to TOP_K",
    )
    return parser.parse_args()


def load_mapping(path: str | Path) -> dict:
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    return {str(k): v for k, v in data.items()}


def iter_frames(frames: Iterable, top_k: int | None) -> Iterable[tuple[int, tuple]]:
    if top_k is not None:
        frames = frames[:top_k]
    for idx, entry in enumerate(frames):
        yield idx, entry


def overlay_mask(image: Image.Image, pix_ids: np.ndarray) -> Image.Image:
    """Return ``image`` with ``pix_ids`` overlayed in red."""

    mask = np.zeros((240, 320), dtype=np.uint8)
    if pix_ids.size:
        mask[pix_ids[:, 0], pix_ids[:, 1]] = 1
    rgba = np.zeros((240, 320, 4), dtype=np.uint8)
    rgba[..., 0] = 255  # red
    rgba[..., 3] = 128  # alpha
    overlay = Image.fromarray(rgba * mask[..., None], mode="RGBA")
    return Image.alpha_composite(image.convert("RGBA"), overlay)


def main() -> None:
    args = parse_args()
    obj2frame = load_mapping(args.object2frame)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "visibility.log", "w") as log:
        for inst_id, frames in obj2frame.items():
            for frame_idx, (rel_path, _pix_cnt, vis, bbox, pix_ids) in iter_frames(
                frames, args.top_k
            ):
                img = Image.open(Path(args.scan_dir) / rel_path).convert("RGB")
                img = img.resize((320, 240))
                blended = overlay_mask(img, np.asarray(pix_ids))
                draw = ImageDraw.Draw(blended)
                draw.rectangle(bbox, outline="yellow", width=2)
                out_file = out_dir / f"{inst_id}.png"
                blended.save(out_file)
                log.write(f"{inst_id}: {vis}\n")
                break


if __name__ == "__main__":
    main()
