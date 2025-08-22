import os
import subprocess
import sys


def test_blip_encode_images_converts_tensor_and_array():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    script = """
import os, sys, types, numpy as np, torch
from PIL import Image
root = os.environ['PROJ_ROOT']
sys.path.insert(0, root)
clip_utils = types.ModuleType('open3dsg.models.clip_utils')
clip_utils.encode_node_images_in_batches = lambda *a, **k: None
config_module = types.ModuleType('open3dsg.config.config')
config_module.CONF = types.SimpleNamespace(PATH=types.SimpleNamespace(FEATURES='', CHECKPOINTS=''))
sgpn_module = types.ModuleType('open3dsg.models.sgpn')
class SGPN(torch.nn.Module):
    def __init__(self, *args, **kwargs): super().__init__()
sgpn_module.SGPN = SGPN
sys.modules.update({
    'open3dsg.models.clip_utils': clip_utils,
    'open3dsg.config.config': config_module,
    'open3dsg.models.sgpn': sgpn_module,
    'clip': types.ModuleType('clip'),
})
from open3dsg.scripts.feature_dumper import MinimalSGPN
model = MinimalSGPN({})
model.dummy = torch.nn.Parameter(torch.empty(0))
class DummyBatch:
    def __init__(self, bs): self.data={'pixel_values': torch.zeros(bs,3,4,4)}
    def to(self, device): return self
    def __getitem__(self, k): return self.data[k]
class DummyProcessor:
    def __init__(self): self.images=None
    def __call__(self, images, text=None, input_data_format=None, return_tensors=None):
        self.images=images
        assert input_data_format=='channels_last'
        return DummyBatch(len(images))
class DummyBLIP:
    def embedd_image(self, pixel_values):
        bs=pixel_values.shape[0]
        return torch.zeros(bs,1,1)
model.PROCESSOR=DummyProcessor()
model.BLIP=DummyBLIP()
tensor_img=torch.zeros(3,4,4)
ndarray_img=np.zeros((4,4,3),dtype=np.uint8)
out=model.blip_encode_images([[tensor_img, ndarray_img]], batch_size=2)
assert out.shape==(1,2,1,1)
assert all(isinstance(img, Image.Image) for img in model.PROCESSOR.images)
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env={**os.environ, "PROJ_ROOT": root}
    )
    assert result.returncode == 0, result.stderr or result.stdout

