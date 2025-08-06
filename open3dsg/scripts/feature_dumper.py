# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import os
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import clip

from open3dsg.config.config import CONF
from open3dsg.models.sgpn import SGPN


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


class MinimalSGPN(SGPN):
    """Lightweight wrapper exposing only the 2D encoders from :class:`SGPN`."""

    def __init__(self, hparams):
        # Skip heavy initialisation by avoiding ``SGPN.__init__``.
        torch.nn.Module.__init__(self)
        self.hparams = hparams

        # Placeholders populated during setup.
        self.CLIP = None
        self.CLIP_NODE = None
        self.CLIP_EDGE = None
        self.OPENSEG = None
        self.BLIP = None
        self.LLaVA = None
        self.PROCESSOR = None
        self.blip_pos_encoding = None

    @torch.no_grad()
    def blip_encode_images(self, rel_imgs, batch_size: int = 32):
        device = next(self.parameters()).device
        rel_images_tensor = np.array(rel_imgs)
        flat_imgs = rel_images_tensor.flatten().tolist()
        rel_embeds = []
        with torch.no_grad():
            for i in range(0, len(flat_imgs), batch_size):
                batch = flat_imgs[i : i + batch_size]
                inputs = (
                    self.PROCESSOR(images=batch, text=None, return_tensors="pt")
                    .to(device)
                )
                rel_embeds.append(self.BLIP.embedd_image(inputs["pixel_values"]))
                torch.cuda.empty_cache()

        rel_embeds = torch.cat(rel_embeds, dim=0).view(
            (*rel_images_tensor.shape, 257, 1408)
        )
        return rel_embeds


class FeatureDumper:
    """Utility class for precomputing 2D features without Lightning overhead."""

    def __init__(self, hparams, device: int = 0):
        self.hparams = hparams
        self.hparams.setdefault("load_features", False)
        self.hparams.setdefault("test", False)
        self.device_index = device
        # Only keep lightweight 2D encoders.
        self.model = MinimalSGPN(self.hparams)

        # default path for dumping node features when stage == 'nodes'
        self.clip_path = os.path.join(
            CONF.PATH.FEATURES,
            f"clip_features_{datetime.now().strftime('%Y-%m-%d-%H-%M')}",
        )

    def setup(self):
        """Load pretrained 2D models required for feature extraction."""
        if self.hparams['clip_model'] == 'OpenSeg':
            self.model.OPENSEG = self.model.load_pretrained_clip_model(
                target_model=self.model.OPENSEG, model=self.hparams['clip_model']
            )
        else:
            self.model.CLIP = self.model.load_pretrained_clip_model(
                target_model=self.model.CLIP, model=self.hparams['clip_model']
            )

        if self.hparams.get('node_model'):
            self.model.CLIP_NODE = self.model.load_pretrained_clip_model(
                target_model=self.model.CLIP_NODE, model=self.hparams['node_model']
            )
        if self.hparams.get('edge_model'):
            self.model.CLIP_EDGE = self.model.load_pretrained_clip_model(
                target_model=self.model.CLIP_EDGE, model=self.hparams['edge_model']
            )

        if self.hparams.get('blip'):
            if self.hparams.get('dump_features'):
                self.model.load_pretrained_blipvision_model()
            else:
                self.model.load_pretrained_blip_model()
        elif self.hparams.get('llava'):
            self.model.load_pretrained_llava_model()

        device = torch.device(
            f"cuda:{self.device_index}" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(device)

    def encode_features(self, data_dict):
        """Populate ``clip_obj_encoding`` and ``clip_rel_encoding`` using 2D encoders."""

        device = next(self.model.parameters()).device
        model = (
            self.model.module
            if isinstance(
                self.model, torch.nn.parallel.DistributedDataParallel
            )
            else self.model
        )

        obj_imgs = data_dict.get('object_imgs')
        rel_imgs = data_dict.get('relationship_imgs')
        obj_raw_imgs = data_dict.get('object_raw_imgs')
        obj_pixels = data_dict.get('object_pixels')
        obj_nums = data_dict.get('objects_count')

        if obj_imgs is not None and torch.is_tensor(obj_imgs):
            obj_imgs = obj_imgs.to(device)
        if obj_raw_imgs is not None and torch.is_tensor(obj_raw_imgs):
            obj_raw_imgs = obj_raw_imgs.to(device)
        if rel_imgs is not None and torch.is_tensor(rel_imgs):
            rel_imgs = rel_imgs.to(device)

        def _rel_imgs_empty(imgs):
            return imgs is None or (
                isinstance(imgs, list)
                and (not imgs or sum(len(r) for r in imgs) == 0)
            )

        if self.hparams.get('blip') and _rel_imgs_empty(rel_imgs):
            rel_imgs = data_dict.get('blip_images')

        rel_imgs_empty = _rel_imgs_empty(rel_imgs)

        clip_rel_feats = None

        if self.hparams['clip_model'] == 'OpenSeg' and obj_raw_imgs is not None:
            rel_input = rel_imgs if torch.is_tensor(rel_imgs) else None
            if rel_input is None:
                rel_input = torch.zeros(
                    obj_raw_imgs.size(0), 1, 1, 3, obj_raw_imgs.size(-2), obj_raw_imgs.size(-1),
                    device=obj_raw_imgs.device,
                )
            clip_obj_feats, clip_rel_feats = model.clip_encode_pixels(
                obj_raw_imgs, obj_pixels, obj_nums, rel_input
            )
            data_dict['clip_obj_encoding'] = clip_obj_feats
        elif obj_imgs is not None:
            rel_input = rel_imgs if torch.is_tensor(rel_imgs) else None
            if rel_input is None:
                rel_input = torch.zeros(
                    obj_imgs.size(0), 1, 1, 3, obj_imgs.size(-2), obj_imgs.size(-1),
                    device=obj_imgs.device,
                )
            clip_obj_feats, clip_rel_feats = model.clip_encode_imgs(
                obj_imgs, rel_input
            )
            data_dict['clip_obj_encoding'] = clip_obj_feats
        elif isinstance(rel_imgs, torch.Tensor) and not self.hparams.get('blip') and not self.hparams.get('llava'):
            dummy = torch.zeros(
                rel_imgs.size(0), 1, 1, 3, rel_imgs.size(-2), rel_imgs.size(-1),
                device=rel_imgs.device,
            )
            _, clip_rel_feats = model.clip_encode_imgs(dummy, rel_imgs)

        if self.hparams.get('blip'):
            if rel_imgs is not None and not rel_imgs_empty:
                data_dict['clip_rel_encoding'] = model.blip_encode_images(
                    rel_imgs, batch_size=self.hparams.get('blip_batch_size', 32)
                )
            else:
                num_frames = self.hparams.get('top_k_frames', 1) * self.hparams.get('scales', 1)
                data_dict['clip_rel_encoding'] = torch.empty(
                    (0, num_frames, 257, 1408), device=device
                )
        elif self.hparams.get('llava') and rel_imgs is not None:
            data_dict['clip_rel_encoding'] = model.llava_encode_images(rel_imgs)
        elif clip_rel_feats is not None:
            data_dict['clip_rel_encoding'] = clip_rel_feats

        return data_dict

    def _mask_features(self, data_dict, clip_obj_emb, clip_rel_emb, bidx, obj_count, rel_count):
        obj_valids = None
        clip_rel_emb_masked = None
        if isinstance(clip_obj_emb, torch.Tensor):
            if self.hparams['clip_model'] == 'OpenSeg':
                clip_obj2frame_mask = data_dict['obj2frame_raw_mask'][bidx][:obj_count]
            else:
                clip_obj2frame_mask = data_dict['obj2frame_mask'][bidx][:obj_count]

            clip_obj_mask = (
                torch.arange(clip_obj_emb.size(1)).unsqueeze(0).to(clip_obj2frame_mask.device)
                < clip_obj2frame_mask.unsqueeze(1)
            )
            clip_obj_emb[~clip_obj_mask] = np.nan
            clip_obj_emb = torch.nanmean(clip_obj_emb, dim=1)
            obj_valids = ~torch.isnan(clip_obj_emb).all(-1)

        if isinstance(clip_rel_emb, torch.Tensor):
            clip_rel2frame_mask = data_dict['rel2frame_mask'][bidx][:rel_count]
            clip_rel_mask = (
                torch.arange(clip_rel_emb.size(1)).unsqueeze(0).to(clip_rel2frame_mask.device)
                < clip_rel2frame_mask.unsqueeze(1)
            )
            clip_rel_emb[~clip_rel_mask] = np.nan
            clip_rel_emb = torch.nanmean(clip_rel_emb, dim=1)

            clip_rel_emb_masked = torch.zeros_like(clip_rel_emb)
            clip_rel_emb_masked[clip_rel2frame_mask > 0] = clip_rel_emb[clip_rel2frame_mask > 0]

        return obj_valids, clip_obj_emb, clip_rel_emb_masked

    def _dump_features(self, data_dict, batch_size, path=CONF.PATH.FEATURES):
        for bidx in range(batch_size):
            obj_count = int(data_dict['objects_count'][bidx].item())
            rel_count = int(data_dict['predicate_count'][bidx].item())

            clip_obj_emb = data_dict['clip_obj_encoding'][bidx][:obj_count]
            clip_rel_emb = None
            if (
                not self.hparams.get('skip_edge_features')
                and data_dict.get('clip_rel_encoding') is not None
            ):
                clip_rel_emb = data_dict['clip_rel_encoding'][bidx][:rel_count]

            obj_valids, clip_obj_emb, clip_rel_emb_masked = self._mask_features(
                data_dict, clip_obj_emb, clip_rel_emb, bidx, obj_count, rel_count
            )

            obj_clip_model = (
                self.hparams['node_model']
                if self.hparams.get('node_model') and self.hparams['clip_model'] != 'OpenSeg'
                else self.hparams['clip_model']
            )

            obj_path = os.path.join(
                path, 'export_obj_clip_emb_clip_' + obj_clip_model.replace('/', '-')
            )
            obj_valid_path = os.path.join(path, 'export_obj_clip_valids')
            os.makedirs(obj_path, exist_ok=True)
            os.makedirs(obj_valid_path, exist_ok=True)

            torch.save(
                clip_obj_emb.detach().cpu(),
                os.path.join(obj_path, data_dict['scan_id'][bidx] + '.pt'),
            )
            torch.save(
                obj_valids.detach().cpu(),
                os.path.join(obj_valid_path, data_dict['scan_id'][bidx] + '.pt'),
            )
            if clip_rel_emb_masked is not None:
                rel_clip_model = (
                    self.hparams['edge_model']
                    if self.hparams.get('edge_model')
                    else self.hparams['clip_model']
                )
                if self.hparams.get('blip'):
                    rel_clip_model = 'BLIP'
                elif self.hparams.get('llava'):
                    rel_clip_model = 'LLaVa'
                rel_path = os.path.join(
                    path, 'export_rel_clip_emb_clip_' + rel_clip_model.replace('/', '-')
                )
                os.makedirs(rel_path, exist_ok=True)
                torch.save(
                    clip_rel_emb_masked.detach().cpu(),
                    os.path.join(rel_path, data_dict['scan_id'][bidx] + '.pt'),
                )
