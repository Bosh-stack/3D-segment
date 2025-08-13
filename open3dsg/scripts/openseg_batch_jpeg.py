"""Batch encode images with OpenSeg and save per-pixel features.

This utility recursively traverses an ``image_dir`` and for every image with
one of the extensions ``.jpg``, ``.jpeg``, ``.png`` or ``.bmp`` it stores

* the raw OpenSeg feature map as ``.npy`` file and
* a colourised segmentation produced by k-means clustering as ``.jpg`` file.

The directory structure of the input is mirrored inside ``output_dir``.

Example
-------
```bash
python open3dsg/scripts/openseg_batch_jpeg.py \
    path/to/images path/to/output \
    --model_dir checkpoints/openseg --clusters 20
```
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.cluster import KMeans


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dump OpenSeg features and colourised segmentation maps for a directory of images",
    )
    parser.add_argument("image_dir", help="Directory containing input images")
    parser.add_argument("output_dir", help="Directory where outputs are written")
    parser.add_argument(
        "--model_dir",
        default="checkpoints/openseg",
        help="Path to the exported OpenSeg SavedModel",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=15,
        help="Number of k-means clusters for colourised segmentation",
    )
    return parser.parse_args()


def load_model(model_dir: str):
    """Load OpenSeg SavedModel from ``model_dir``."""
    return tf.saved_model.load(model_dir)


def extract_feature_map(model, image: np.ndarray) -> np.ndarray:
    """Return per-pixel OpenSeg features for ``image`` (H x W x 3 array)."""
    jpeg_bytes = tf.io.encode_jpeg(image)
    dummy_text_emb = tf.zeros([1, 1, 768])
    result = model.signatures["serving_default"](
        inp_image_bytes=tf.convert_to_tensor(jpeg_bytes),
        inp_text_emb=dummy_text_emb,
    )
    img_info = result["image_info"]
    crop_sz = [
        int(img_info[0, 0] * img_info[2, 0]),
        int(img_info[0, 1] * img_info[2, 1]),
    ]
    feat = result["ppixel_ave_feat"][:, : crop_sz[0], : crop_sz[1]]
    resized = tf.image.resize(feat, image.shape[:2], method="nearest")[0]
    return resized.numpy()


def colourise(features: np.ndarray, clusters: int) -> np.ndarray:
    """Cluster ``features`` and return a colour image."""
    h, w, c = features.shape
    flat = features.reshape(-1, c)
    kmeans = KMeans(n_clusters=clusters, n_init=10, random_state=0).fit(flat)
    labels = kmeans.labels_.reshape(h, w)
    colours = np.random.default_rng(0).integers(0, 255, size=(clusters, 3), dtype=np.uint8)
    return colours[labels]


def process_image(model, img_path: Path, src_root: Path, out_root: Path, clusters: int) -> None:
    rel = img_path.relative_to(src_root)
    feature_path = (out_root / rel).with_suffix(".npy")
    seg_path = (out_root / rel).with_suffix(".jpg")
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    image = np.array(Image.open(img_path).convert("RGB"))
    feat_map = extract_feature_map(model, image)
    np.save(feature_path, feat_map)
    seg = colourise(feat_map, clusters)
    Image.fromarray(seg).save(seg_path)


def main() -> None:
    args = parse_args()
    model = load_model(args.model_dir)
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    img_root = Path(args.image_dir)
    out_root = Path(args.output_dir)
    paths = []
    for ext in exts:
        paths.extend(img_root.rglob(ext))
    for p in sorted(paths):
        process_image(model, p, img_root, out_root, args.clusters)


if __name__ == "__main__":
    main()
