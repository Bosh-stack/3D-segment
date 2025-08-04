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


class FeatureDumper:
    """Utility class for precomputing 2D features without Lightning overhead."""

    def __init__(self, hparams):
        self.hparams = hparams
        self.model = SGPN(self.hparams)
        self.model.apply(inplace_relu)

        # default path for dumping node features when stage == 'nodes'
        self.clip_path = os.path.join(
            CONF.PATH.FEATURES,
            f"clip_features_{datetime.now().strftime('%Y-%m-%d-%H-%M')}",
        )
        self.CLIP_NONE_EMB = None

    def setup(self):
        """Load pretrained models required for feature extraction."""
        if not self.hparams.get('clean_pointnet') and not self.model.rgb and not self.model.nrm:
            self.model.load_pretained_cls_model(self.model.objPointNet)
            self.model.load_pretained_cls_model(self.model.relPointNet)

        if self.hparams['clip_model'] == 'OpenSeg':
            self.model.OPENSEG = self.model.load_pretrained_clip_model(
                target_model=self.model.OPENSEG, model=self.hparams['clip_model']
            )
        else:
            self.model.CLIP = self.model.load_pretrained_clip_model(
                target_model=self.model.CLIP, model=self.hparams['clip_model']
            )

        if self.hparams['clip_model'] != 'OpenSeg':
            with torch.no_grad():
                self.CLIP_NONE_EMB = F.normalize(
                    self.model.CLIP.encode_text(
                        clip.tokenize(['none']).to(self.model.clip_device)
                    )
                )

        if self.hparams.get('node_model'):
            self.model.CLIP_NODE = self.model.load_pretrained_clip_model(
                target_model=self.model.CLIP_NODE, model=self.hparams['node_model']
            )
        if self.hparams.get('edge_model'):
            self.model.CLIP_EDGE = self.model.load_pretrained_clip_model(
                target_model=self.model.CLIP_EDGE, model=self.hparams['edge_model']
            )
            with torch.no_grad():
                self.CLIP_NONE_EMB = F.normalize(
                    self.model.CLIP_EDGE.encode_text(
                        clip.tokenize(['none']).to(self.model.clip_device)
                    )
                )

        if self.hparams.get('blip'):
            if self.hparams.get('dump_features'):
                self.model.load_pretrained_blipvision_model()
            else:
                self.model.load_pretrained_blip_model()
        elif self.hparams.get('llava'):
            self.model.load_pretrained_llava_model()

    def _forward(self, data_dict):
        data_dict = self.model(data_dict)
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
            if not self.hparams.get('blip') and not self.hparams.get('llava'):
                clip_rel_emb_masked[clip_rel2frame_mask == 0] = self.CLIP_NONE_EMB.to(
                    clip_rel_emb_masked[clip_rel2frame_mask == 0].dtype
                )
            else:
                clip_rel_emb_masked[clip_rel2frame_mask == 0] = np.nan

        return obj_valids, clip_obj_emb, clip_rel_emb_masked

    def _dump_features(self, data_dict, batch_size, path=CONF.PATH.FEATURES):
        for bidx in range(batch_size):
            obj_count = int(data_dict['objects_count'][bidx].item())
            rel_count = int(data_dict['predicate_count'][bidx].item())

            clip_obj_emb = data_dict['clip_obj_encoding'][bidx][:obj_count]
            clip_rel_emb = data_dict['clip_rel_encoding'][bidx][:rel_count]

            obj_valids, clip_obj_emb, clip_rel_emb_masked = self._mask_features(
                data_dict, clip_obj_emb, clip_rel_emb, bidx, obj_count, rel_count
            )

            obj_clip_model = (
                self.hparams['node_model']
                if self.hparams.get('node_model') and self.hparams['clip_model'] != 'OpenSeg'
                else self.hparams['clip_model']
            )
            rel_clip_model = self.hparams['edge_model'] if self.hparams.get('edge_model') else self.hparams['clip_model']
            if self.hparams.get('blip'):
                rel_clip_model = 'BLIP'
            elif self.hparams.get('llava'):
                rel_clip_model = 'LLaVa'

            obj_path = os.path.join(path, 'export_obj_clip_emb_clip_' + obj_clip_model.replace('/', '-'))
            rel_path = os.path.join(path, 'export_rel_clip_emb_clip_' + rel_clip_model.replace('/', '-'))
            obj_valid_path = os.path.join(path, 'export_obj_clip_valids')
            os.makedirs(obj_path, exist_ok=True)
            os.makedirs(obj_valid_path, exist_ok=True)
            if clip_rel_emb_masked is not None:
                os.makedirs(rel_path, exist_ok=True)

            torch.save(clip_obj_emb.detach().cpu(), os.path.join(obj_path, data_dict['scan_id'][bidx] + '.pt'))
            torch.save(obj_valids.detach().cpu(), os.path.join(obj_valid_path, data_dict['scan_id'][bidx] + '.pt'))
            if clip_rel_emb_masked is not None:
                torch.save(
                    clip_rel_emb_masked.detach().cpu(),
                    os.path.join(rel_path, data_dict['scan_id'][bidx] + '.pt'),
                )
