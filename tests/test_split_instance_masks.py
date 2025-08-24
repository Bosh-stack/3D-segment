import os
import sys
import types
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ---- stub external heavy modules ----
cv2 = types.SimpleNamespace(
    imread=lambda *args, **kwargs: np.zeros((2, 2, 3), dtype=np.uint8),
    cvtColor=lambda img, flag: img,
    COLOR_BGR2RGB=0,
)
sys.modules['cv2'] = cv2

class DummyContext:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        return False

torch = types.SimpleNamespace(no_grad=lambda: DummyContext())
sys.modules['torch'] = torch

trimesh = types.SimpleNamespace(
    load=lambda *args, **kwargs: types.SimpleNamespace(vertices=np.zeros((0, 3)), metadata={'_ply_raw': {'vertex': {'data': {'objectId': np.array([])}}}}),
    PointCloud=lambda pts: types.SimpleNamespace(export=lambda path: None),
)
sys.modules['trimesh'] = trimesh

# tqdm stubs
import types as _types
_tqdm_mod = _types.ModuleType('tqdm')
_tqdm_mod.tqdm = lambda x, *a, **k: x
_tqdm_contrib = _types.ModuleType('tqdm.contrib')
_tqdm_concurrent = _types.ModuleType('tqdm.contrib.concurrent')
_tqdm_concurrent.process_map = lambda *a, **k: None
_tqdm_contrib.concurrent = _tqdm_concurrent
sys.modules['tqdm'] = _tqdm_mod
sys.modules['tqdm.contrib'] = _tqdm_contrib
sys.modules['tqdm.contrib.concurrent'] = _tqdm_concurrent

# matplotlib stub
matplotlib_mod = _types.ModuleType('matplotlib')
pyplot_mod = _types.ModuleType('matplotlib.pyplot')
matplotlib_mod.pyplot = pyplot_mod
sys.modules['matplotlib'] = matplotlib_mod
sys.modules['matplotlib.pyplot'] = pyplot_mod

# easydict stub
ed = _types.ModuleType('easydict')
class EasyDict(dict):
    __getattr__ = dict.get
    def __setattr__(self, k, v):
        self[k] = v
ed.EasyDict = EasyDict
sys.modules['easydict'] = ed

# open3d stub
o3d_stub = types.SimpleNamespace(
    geometry=types.SimpleNamespace(PointCloud=object),
    utility=types.SimpleNamespace(Vector3dVector=lambda x: x),
    io=types.SimpleNamespace(read_point_cloud=lambda *a, **k: None,
                             write_point_cloud=lambda *a, **k: None),
)
sys.modules['open3d'] = o3d_stub

# config requires dataset root
os.makedirs('/NetworkDocker/Data/Open3DSG_trainset', exist_ok=True)
import sys as _sys
if 'scipy' in _sys.modules:
    _sys.modules.pop('scipy', None)
    _sys.modules.pop('scipy.spatial', None)
    _sys.modules.pop('scipy.spatial.transform', None)
import scipy  # ensure real SciPy for sklearn


from open3dsg.data.split_instance_masks import aggregate_point_features


def test_aggregate_point_features_rgb():
    pts = np.array([[0.0, -0.9, 1.0], [1.0, 0.0, 1.0]], dtype=np.float32)
    depth = (np.ones((2, 2), dtype=np.float32) * 1000).reshape(-1)
    depths = np.array([depth])
    extrinsics = np.array([np.eye(4, dtype=np.float32)])
    intrinsic = np.eye(3, dtype=np.float32)
    image = np.array([
        [[1, 2, 3], [4, 5, 6]],
        [[7, 8, 9], [10, 11, 12]],
    ], dtype=np.uint8)
    images = np.array([image])
    feats = aggregate_point_features(pts, [0], depths, extrinsics, intrinsic, images)
    assert np.allclose(feats[0], np.array([7, 8, 9]) / 255.0)
    assert np.allclose(feats[1], np.array([4, 5, 6]) / 255.0)
