import json
from pathlib import Path

import numpy as np

from open3dsg.data.get_object_frame_sdb import load_cam as load_cam_sdb
from open3dsg.data.get_object_frame_myset import load_cam as load_cam_myset


def _write_meta(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "meta.json"
    p.write_text(json.dumps(data))
    return p


def _check_identity_transform(T: np.ndarray, tx: float, ty: float, tz: float):
    expected = np.eye(4, dtype=np.float32)
    expected[:3, 3] = -np.array([tx, ty, tz], dtype=np.float32)
    assert np.allclose(T, expected)


def _assert_load_cam(meta: dict, tmp_path: Path, expected_wh, trans):
    meta_path = _write_meta(tmp_path, meta)
    for load_cam in (load_cam_sdb, load_cam_myset):
        K, T, w, h = load_cam(meta_path)
        assert w == expected_wh[0]
        assert h == expected_wh[1]
        _check_identity_transform(T, *trans)
        # simple intrinsic check: fx at (0,0)
        assert K[0, 0] == meta.get("fx", K[0, 0])


def test_load_cam_flat_and_image_fields(tmp_path):
    meta = {
        "fx": 100.0,
        "fy": 110.0,
        "cx": 50.0,
        "cy": 60.0,
        "img_width": 128,
        "img_height": 96,
        "qw": 1.0,
        "qx": 0.0,
        "qy": 0.0,
        "qz": 0.0,
        "tx": 1.0,
        "ty": 2.0,
        "tz": 3.0,
    }
    _assert_load_cam(meta, tmp_path, (128, 96), (1.0, 2.0, 3.0))


def test_load_cam_nested_fields(tmp_path):
    meta = {
        "K": [100.0, 0.0, 50.0, 0.0, 110.0, 60.0, 0.0, 0.0, 1.0],
        "image": {"imageWidth": 200, "imageHeight": 150},
        "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
        "translation": {"x": 4.0, "y": 5.0, "z": 6.0},
    }
    _assert_load_cam(meta, tmp_path, (200, 150), (4.0, 5.0, 6.0))

