import os
import sys
import types
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ---- stub external heavy modules ----
# minimal stubs to satisfy imports
cv2 = types.SimpleNamespace(imread=lambda *args, **kwargs: np.zeros((2,2), dtype=np.uint8),
                           cvtColor=lambda img, flag: img,
                           COLOR_BGR2RGB=0)
sys.modules['cv2'] = cv2

class DummyContext:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        return False

torch = types.SimpleNamespace(no_grad=lambda: DummyContext())
sys.modules['torch'] = torch

trimesh = types.SimpleNamespace(load=lambda *args, **kwargs: types.SimpleNamespace(vertices=np.zeros((0,3)),
                                                                             metadata={'_ply_raw': {'vertex': {'data': {'objectId': np.array([])}}}}))
sys.modules['trimesh'] = trimesh

tqdm_mod = types.ModuleType('tqdm')
tqdm_mod.tqdm = lambda x, *a, **k: x
contrib_mod = types.ModuleType('tqdm.contrib')
concurrent_mod = types.ModuleType('tqdm.contrib.concurrent')
concurrent_mod.process_map = lambda *args, **kwargs: None
contrib_mod.concurrent = concurrent_mod
sys.modules['tqdm'] = tqdm_mod
sys.modules['tqdm.contrib'] = contrib_mod
sys.modules['tqdm.contrib.concurrent'] = concurrent_mod

matplotlib_mod = types.ModuleType('matplotlib')
pyplot_mod = types.ModuleType('matplotlib.pyplot')
matplotlib_mod.pyplot = pyplot_mod
sys.modules['matplotlib'] = matplotlib_mod
sys.modules['matplotlib.pyplot'] = pyplot_mod

# easydict stub
ed = types.ModuleType('easydict')
class EasyDict(dict):
    __getattr__ = dict.get
    def __setattr__(self, k, v):
        self[k] = v
ed.EasyDict = EasyDict
sys.modules['easydict'] = ed

# open3d stub with minimal attributes for annotations
o3d_stub = types.SimpleNamespace(
    geometry=types.SimpleNamespace(PointCloud=object),
    utility=types.SimpleNamespace(Vector3dVector=lambda x: x),
    io=types.SimpleNamespace(read_point_cloud=lambda *a, **k: None,
                             write_point_cloud=lambda *a, **k: None)
)
sys.modules['open3d'] = o3d_stub

# scipy stub for Rotation
scipy_mod = types.ModuleType('scipy')
spatial_mod = types.ModuleType('scipy.spatial')
transform_mod = types.ModuleType('scipy.spatial.transform')
transform_mod.Rotation = type(
    'Rotation',
    (),
    {'from_quat': staticmethod(lambda q: types.SimpleNamespace(as_matrix=lambda: np.eye(3)))}
)
spatial_mod.transform = transform_mod
scipy_mod.spatial = spatial_mod
sys.modules['scipy'] = scipy_mod
sys.modules['scipy.spatial'] = spatial_mod
sys.modules['scipy.spatial.transform'] = transform_mod

# config requires a dataset root to exist
os.makedirs('/NetworkDocker/Data/Open3DSG_trainset', exist_ok=True)

from open3dsg.data.get_object_frame_myset import visible_ratio, projection_details
from scripts.merge_3d_masks import _project
from open3dsg.data.get_object_frame import compute_mapping, image_3d_mapping


def test_projection_details_identity():
    K = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    T = np.eye(4, dtype=np.float32)
    pts = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    ratio = visible_ratio(pts, K, T, 100, 80)
    assert ratio == 1.0
    ratio2, pix_cnt, bbox, pix_ids = projection_details(pts, K, T, 100, 80)
    assert ratio2 == 1.0
    assert pix_cnt == 1
    assert bbox == (50, 40, 50, 40)
    assert np.array_equal(pix_ids, np.array([[50, 40]], dtype=np.uint16))
    pix, mask = _project(pts, K, T)
    assert mask.all()
    assert np.allclose(pix, np.array([[50.0, 40.0]], dtype=np.float32))


def test_compute_mapping_identity():
    w2c = np.eye(4, dtype=np.float32)
    coords = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (13, 1))
    depth = np.ones((2, 2), dtype=np.float32)
    intr = np.eye(4, dtype=np.float32)
    mapping = compute_mapping(w2c, coords, depth, intr, 0, 0.05, np.array([2, 2])).T
    assert np.all(mapping[:, :2] == 0)
    assert np.all(mapping[:, 2] == 1)


def test_image_3d_mapping_identity():
    w2c_list = [np.eye(4, dtype=np.float32)]
    depth_flat = (np.ones((2, 2), dtype=np.float32) * 1000).reshape(-1)
    image_list = [depth_flat]
    color_list = [np.zeros((2, 2, 3), dtype=np.uint8)]
    img_names = ['0']
    point_cloud = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (13, 1))
    instances = np.ones((13, 1), dtype=int)
    instance_names = {1: 'obj'}
    intr = np.eye(4, dtype=np.float32)
    obj2frame = image_3d_mapping('scan', image_list, color_list, img_names, point_cloud,
                                 instances, w2c_list, intr, instance_names,
                                 2, 2, 0, 0.0)
    assert 1 in obj2frame
    entry = obj2frame[1][0]
    assert entry[1] == 13
    assert entry[2] == 1.0
    assert entry[3] == (0, 0, 0, 0)
    assert np.array_equal(entry[4], np.array([[0, 0]], dtype=np.uint16))
