# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import os
import re
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import numpy as np

import clip
from transformers import InstructBlipProcessor
import torch.nn.functional as F

try:
    from segment_anything import SamPredictor, sam_model_registry
except Exception:  # pragma: no cover - handled at runtime
    SamPredictor = None
    sam_model_registry = {}

from open3dsg.models.pointnet import PointNetEncoder
from open3dsg.models.pointnet2 import Pointnet2_Ssg as PointNet2Encoder
from open3dsg.models.network_PointNet import PointNetRelClsMulti
from open3dsg.models.network_GNN import build_mlp, TripletGCNModel, GraphEdgeAttenNetworkLayers
from open3dsg.models.custom_instruct_blip import InstructBlipForConditionalGeneration, InstructBlipForConditionalGenerationVisionEncoder
from open3dsg.models.blip_patch_projector import InstructBlipEncoder
try:
    from open3dsg.models.LLaVA.llava_image_text_split import LLaVA_Model
except:
    print("Experimental: No LLava model present in your system")
    pass

from open3dsg.config.config import CONF


blip2_positional_encoding = torch.load(
    os.path.join(CONF.PATH.CHECKPOINTS, 'blip2_positional_embedding.pt'),
    weights_only=False
)


class SGPN(nn.Module):
    def __init__(self, hparams
                 ):
        super().__init__()

        self.debug = False
        self.hparams = hparams
        # default edge model to CLIP when not specified
        self.hparams.setdefault('edge_model', self.hparams.get('clip_model'))

        self.rgb = hparams['use_rgb']
        self.nrm = False
        self.channels = 3 + 3*self.rgb + 3*self.nrm
        self.pointnet2 = hparams['pointnet2']
        ########### point encoders ###########
        if self.pointnet2:
            self.objPointNet = PointNet2Encoder(normal_channel=True)
            self.relPointNet = PointNet2Encoder(normal_channel=True)
        else:
            self.objPointNet = PointNetEncoder(global_feat=True, feature_transform=True,
                                               channel=self.channels)  # (x,y,z) + (r,g,b) + (nx,ny,nz)
            # (x,y,z,M) M-> class-agnostic instance segmentation
            self.relPointNet = PointNetEncoder(global_feat=True, feature_transform=True, channel=self.channels+1)

        ########### embedding layers ###########
        box_params = 6
        self.box_gconv_dim = 32
        self.loc_gconv_dim = 64
        self.bbox_emb = nn.Linear(box_params, self.box_gconv_dim)

        self.center_obj_emb = nn.Linear(3, self.loc_gconv_dim)
        self.dist_pred_emb = nn.Linear(3, self.loc_gconv_dim)
        self.angle_emb = nn.Linear(1, self.box_gconv_dim)

        self.pointnet_adapter = build_mlp([256+64 if self.pointnet2 else 1024+64, 512,
                                          self.hparams['gconv_dim']], activation='relu', on_last=True)

        ########### GCN encoding ###########
        if hparams['gnn_layers'] > 0:
            if hparams.get('graph_backbone', None) is None:
                hparams['graph_backbone'] = 'message'
            if hparams['graph_backbone'] == "message":
                self.gconv_net = TripletGCNModel(num_layers=hparams['gnn_layers'],
                                                 dim_node=hparams['gconv_dim'],
                                                 dim_edge=hparams['gconv_dim'],
                                                 dim_hidden=hparams['hidden_dim'],
                                                 aggr='max')
            elif hparams['graph_backbone'] == 'attention':
                self.gconv_net = GraphEdgeAttenNetworkLayers(num_layers=hparams['gnn_layers'],
                                                             dim_node=hparams['gconv_dim'],
                                                             dim_edge=hparams['gconv_dim'],
                                                             dim_hidden=hparams['hidden_dim'],
                                                             dim_atten=hparams['atten_dim'],
                                                             num_heads=hparams['gconv_nheads'],
                                                             DROP_OUT_ATTEN=0.3)
            elif hparams['graph_backbone'] == 'gru_attention':
                pass
            elif hparams['graph_backbone'] == 'transformer':
                pass
            elif hparams['graph_backbone'] == 'mlp':
                pass
            else:
                pass

        rel_emb = (768 if "ViT-L/14" in hparams['edge_model']
                   else 512) if self.hparams['edge_model'] else (768 if "ViT-L/14" in hparams['clip_model'] else 512)
        if self.hparams['blip']:
            rel_emb = 1408
        elif self.hparams.get('llava') and self.hparams.get('avg_llava_emb'):
            rel_emb = 4096  # 4096
        elif self.hparams.get('llava'):
            rel_emb = 1024*576

        self.obj_projector = build_mlp([
            hparams['gconv_dim'],
            1024,
            1024,
            768 if "ViT-L/14" in hparams['clip_model'] else 512
        ], activation='relu', on_last=False)
        self.rel_projector = build_mlp([hparams['gconv_dim'], 1024, 1024, rel_emb], activation='relu', on_last=False)

        self.CLIP = None
        self.CLIP_NODE = None
        self.CLIP_EDGE = None
        self.sam_predictor = None
        self.BLIP = None
        self.LLaVA = None
        self.PROCESSOR = None
        self.blip_pos_encoding = None

        if self.hparams['blip']:
            self.blip_layernorm = nn.LayerNorm(1408, eps=1e-06)
            if not self.hparams['avg_blip_emb']:
                self.blip_projector = InstructBlipEncoder(layers=self.hparams.get('blip_proj_layers', 3))

        if self.hparams.get('supervised_edges'):
            self.rel_classifier = PointNetRelClsMulti(
                k=27,
                in_size=hparams['gconv_dim'],
                batch_norm=True, drop_out=True)

    def load_pretained_cls_model(self, model):
        if self.pointnet2:
            pth = os.path.join(CONF.PATH.CHECKPOINTS, 'pointnet2_ulip.pt')
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
                pretrained_dict = torch.load(
                    pth,
                    weights_only=False,
                    map_location=torch.device(torch.distributed.get_rank())
                )["state_dict"]
            else:
                pretrained_dict = torch.load(pth, weights_only=False)["state_dict"]
        else:
            pth = os.path.join(CONF.PATH.CHECKPOINTS, 'pointnet.pth')
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
                pretrained_dict = torch.load(
                    pth,
                    weights_only=False,
                    map_location=torch.device(torch.distributed.get_rank())
                )["model_state_dict"]
            else:
                pretrained_dict = torch.load(pth, weights_only=False)["model_state_dict"]

            net_state_dict = model.state_dict()
            pretrained_dict_ = {k[5:]: v for k, v in pretrained_dict.items() if 'feat' in k and v.size() == net_state_dict[k[5:]].size()}
            # net_state_dict.update(pretrained_dict_)
            model.load_state_dict(pretrained_dict_, strict=False)

    def load_pretrained_clip_model(self, target_model, model):

        if self.hparams.get("load_features") and not self.hparams.get("test", False):
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
                clip_model, _ = clip.load('ViT-B/32', device=torch.distributed.get_rank())
                self.clip_device = torch.device(torch.distributed.get_rank())
            else:
                clip_model, _ = clip.load('ViT-B/32', device='cuda')
                self.clip_device = "cuda"
            return clip_model

        if torch.distributed.is_initialized():
            torch.distributed.barrier()
            clip_model, _ = clip.load(model, device=torch.device(torch.distributed.get_rank()))
            clip_model.eval()
            target_model = clip_model
            self.clip_device = torch.device(torch.distributed.get_rank())

        else:
            clip_model, _ = clip.load(model, device="cuda:0")
            clip_model.eval()
            target_model = clip_model
            self.clip_device = "cuda:0"
        return target_model

    def load_pretrained_blip_model(self,):
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
            model = InstructBlipForConditionalGeneration.from_pretrained(
                "Salesforce/instructblip-vicuna-7b", torch_dtype=torch.bfloat16).to(torch.device(torch.distributed.get_rank()))
            processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
            self.BLIP, self.PROCESSOR = model, processor
            self.blip_pos_encoding = blip2_positional_encoding.to(torch.device(torch.distributed.get_rank()))
            self.blip_pos_encoding.requires_grad = False
        else:
            model = InstructBlipForConditionalGeneration.from_pretrained(
                "Salesforce/instructblip-vicuna-7b", torch_dtype=torch.bfloat16).to('cuda:0')
            processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
            self.BLIP, self.PROCESSOR = model, processor
            self.blip_pos_encoding = blip2_positional_encoding.to('cuda:0')
            self.blip_pos_encoding.requires_grad = False

    def load_pretrained_blipvision_model(self,):
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
            model = InstructBlipForConditionalGenerationVisionEncoder.from_pretrained(
                "Salesforce/instructblip-vicuna-7b", torch_dtype=torch.bfloat16).to(torch.device(torch.distributed.get_rank()))
            processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
            self.BLIP, self.PROCESSOR = model, processor
            self.blip_pos_encoding = blip2_positional_encoding.to(torch.device(torch.distributed.get_rank()))
            self.blip_pos_encoding.requires_grad = False
        else:
            model = InstructBlipForConditionalGenerationVisionEncoder.from_pretrained(
                "Salesforce/instructblip-vicuna-7b", torch_dtype=torch.bfloat16).to('cuda:0')
            processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
            self.BLIP, self.PROCESSOR = model, processor
            self.blip_pos_encoding = blip2_positional_encoding.to('cuda:0')
            self.blip_pos_encoding.requires_grad = False

    def load_pretrained_llava_model(self,):
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
            model = LLaVA_Model("liuhaotian/llava-v1.6-mistral-7b", device=torch.device(torch.distributed.get_rank()))
            self.LLaVA = model
        else:
            model = LLaVA_Model("liuhaotian/llava-v1.6-mistral-7b", device='cuda:0')
            self.LLaVA = model

    def load_blip_pos_encoding(self,):
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

            self.blip_pos_encoding = blip2_positional_encoding.to(torch.device(torch.distributed.get_rank()))
            self.blip_pos_encoding.requires_grad = False
        else:

            self.blip_pos_encoding = blip2_positional_encoding.to('cuda:0')
            self.blip_pos_encoding.requires_grad = False

    def forward(self, data_dict):
        batch_size = data_dict["objects_id"].size(0)
        obj_num, pred_num = data_dict["objects_count"], data_dict["predicate_count"]

        if not self.hparams['load_features']:
            obj_clip_encoding, rel_clip_encoding = self.sam_clip_encode_pixels(
                data_dict['object_raw_imgs'], data_dict['object_pixels'], obj_num, data_dict['relationship_imgs']
            )
            data_dict['clip_obj_encoding'], data_dict['clip_rel_encoding'] = obj_clip_encoding, rel_clip_encoding
            if self.hparams.get('blip'):
                data_dict['clip_rel_encoding'] = self.blip_encode_images(
                    data_dict['blip_images'],
                    batch_size=self.hparams.get('blip_batch_size', 32),
                )
            elif self.hparams.get('llava'):
                data_dict['clip_rel_encoding'] = self.llava_encode_images(data_dict['blip_images'])

        objects_pcl = data_dict["objects_pcl"][..., :self.channels]
        predicate_pcl_flag = torch.cat([data_dict["predicate_pcl_flag"][..., :self.channels],
                                       data_dict["predicate_pcl_flag"][..., -1].unsqueeze(-1)], dim=-1)

        trans_feat = []
        obj_vecs, pred_vecs, tf1, tf2 = self.encode_pcl(objects_pcl, predicate_pcl_flag)

        trans_feat.append(tf1)
        trans_feat.append(tf2)
        data_dict["trans_feat"] = trans_feat

        box_enc, angle_enc = self.encode_bbox(data_dict["objects_bbox"])
        center_enc, dist_enc = self.encode_center_dist(data_dict["objects_center"], data_dict["predicate_dist"])

        obj_vecs = torch.cat([obj_vecs, box_enc, angle_enc], dim=-1)
        pred_vecs = torch.cat([pred_vecs, dist_enc], dim=-1)

        obj_vecs_batch, pred_vecs_batch = self.encode_gcn(batch_size, obj_vecs, pred_vecs, obj_num, pred_num,  data_dict["edges"])

        data_dict["objects_enc"] = self.obj_projector(torch.stack(obj_vecs_batch))
        pred_encoding = self.rel_projector(torch.stack(pred_vecs_batch))
        if self.hparams['blip'] and not self.hparams['avg_blip_emb']:
            # blip_pos_encoding: 1 x patches x embedding
            # pred_encoding: batch x edges x embedding
            pred_encoding_pos = pred_encoding.view(-1, 1408).unsqueeze(1)+self.blip_pos_encoding
            projector_outputs = self.blip_projector(inputs_embeds=pred_encoding_pos,
                                                    output_attentions=False,
                                                    output_hidden_states=False,
                                                    return_dict=False,)
            last_hidden_state = projector_outputs[0]
            last_hidden_state = self.blip_layernorm(last_hidden_state)
            pred_encoding = last_hidden_state.view((*pred_encoding.shape[:2], *last_hidden_state.shape[1:]))

        elif self.hparams.get('llava') and self.hparams.get('avg_llava_emb'):
            pred_encoding = pred_encoding.unsqueeze(-2)
            pred_encoding = pred_encoding.repeat(1, 1, 576, 1)
            if pred_encoding.shape[-1] == 1024:
                pred_encoding = pred_encoding.view(-1, 576, 1024)
                pred_encoding = self.LLaVA.project_image(pred_encoding, [1024]*576)
                pred_encoding = pred_encoding.view(batch_size, 576, -1)
        elif self.hparams.get('llava'):
            pred_encoding = pred_encoding.view(-1, 576, 1024)
            if pred_encoding.shape[-1] == 1024:
                pred_encoding = self.LLaVA.project_image(pred_encoding, [1024]*576)
                pred_encoding = pred_encoding.view(*pred_vecs_batch.shape[:2], -1)
        data_dict["predicates_enc"] = pred_encoding

        if self.hparams.get('supervised_edges'):
            data_dict['rel_prediction'] = self.rel_classifier(torch.stack(
                pred_vecs_batch).view(-1, pred_vecs_batch[0].shape[-1])).view(*pred_encoding.shape[:2], -1)

        return data_dict

    def encode_pcl(self, objects_pcl, predicate_pcl):
        objects_pcl_batched = objects_pcl.view(-1, *objects_pcl.shape[-2:])
        objects_pcl_batched = objects_pcl_batched.permute(0, 2, 1)
        obj_vecs, _, tf1 = self.objPointNet(objects_pcl_batched)

        predicate_pcl_batched = predicate_pcl.view(-1, *predicate_pcl.shape[-2:])
        predicate_pcl_batched = predicate_pcl_batched.permute(0, 2, 1)
        pred_vecs, _, tf2 = self.relPointNet(predicate_pcl_batched)

        return obj_vecs.view(objects_pcl.shape[0], -1, obj_vecs.shape[-1]), pred_vecs.view(predicate_pcl.shape[0], -1, pred_vecs.shape[-1]), tf1.view(objects_pcl.shape[0], -1, *tf1.shape[1:]), tf2.view(predicate_pcl.shape[0], -1, *tf2.shape[1:])

    def encode_bbox(self, bboxes):
        # bs, n_objs, _ = bboxes.shape
        # bboxes = bboxes.view(-1,7)
        bbox_enc = self.bbox_emb(bboxes[..., :6])
        angle_enc = self.angle_emb(bboxes[..., 6].unsqueeze(-1))
        return bbox_enc, angle_enc

    def encode_center_dist(self, obj_centers, pred_dists):
        obj_center_enc = self.center_obj_emb(obj_centers)
        pred_dist_enc = self.dist_pred_emb(pred_dists)
        return obj_center_enc, pred_dist_enc

    def encode_gcn(self, batch_size, obj_vecs, pred_vecs, objects_count, predicate_count, edges):
        obj_vecs_list = []
        pred_vecs_list = []

        for i in range(batch_size):
            object_num = objects_count[i]
            predicate_num = predicate_count[i]
            edges_batch = edges[i][:predicate_num]
            obj_vecs_batch = obj_vecs[i, :object_num]
            pred_vecs_batch = pred_vecs[i, :predicate_num]
            if isinstance(self.pointnet_adapter, nn.Sequential):
                o_vecs = self.pointnet_adapter(obj_vecs_batch)
                p_vecs = self.pointnet_adapter(pred_vecs_batch)
            else:
                o_vecs, p_vecs = self.gconv(obj_vecs_batch, pred_vecs_batch, edges_batch)

            if self.hparams['gnn_layers'] > 0:
                o_vecs, p_vecs = self.gconv_net(o_vecs, p_vecs, edges_batch)

            o_vecs_out = torch.cat((o_vecs, torch.zeros((self.hparams.get('max_nodes', -1) -
                                   o_vecs.shape[0], o_vecs.shape[1])).to(o_vecs.device)))
            p_vecs_out = torch.cat((p_vecs, torch.zeros((self.hparams.get('max_edges', -1) -
                                   p_vecs.shape[0], p_vecs.shape[1])).to(p_vecs.device)))

            obj_vecs_list.append(o_vecs_out)  # list of batches
            pred_vecs_list.append(p_vecs_out)

        return obj_vecs_list, pred_vecs_list

    @torch.no_grad()
    def clip_encode_imgs(self, obj_imgs, rel_imgs):
        """
        Batched CLIP image encoding for nodes and relations to cap VRAM.
        Preserves original output shapes: (*imgs.shape[:-3], D)
        """
        device = self.clip_device if hasattr(self, "clip_device") else "cuda"
        use_amp = bool(self.hparams.get("amp", False))
        bs = int(self.hparams.get("clip_batch_size", 64))

        # ---------- Objects ----------
        img_dim = obj_imgs.shape[-2:]
        flat_obj = obj_imgs.view(-1, 3, *img_dim)  # (N_all, 3, H, W)
        obj_chunks = []
        enc = self.CLIP_NODE if self.hparams['node_model'] else self.CLIP
        for s in range(0, flat_obj.shape[0], bs):
            e = min(flat_obj.shape[0], s + bs)
            chunk = flat_obj[s:e].to(device)
            with autocast(enabled=use_amp):
                out = enc.encode_image(chunk)
            # L2-normalize chunk, then append to CPU list
            out = out / out.norm(dim=-1, keepdim=True)
            obj_chunks.append(out.cpu())
            del out, chunk
            torch.cuda.empty_cache()
        obj_clip_encoding = torch.cat(obj_chunks, dim=0).view(*obj_imgs.shape[:-3], -1)

        # ---------- Relations ----------
        rel_clip_encoding = None
        if (
            not self.hparams.get('skip_edge_features')
            and self.CLIP is not None
            and not self.hparams.get('blip')
            and not self.hparams.get('llava')
            and rel_imgs is not None
        ):
            rel_img_dim = rel_imgs.shape[-2:]
            flat_rel = rel_imgs.view(-1, 3, *rel_img_dim)
            rel_chunks = []
            enc_rel = self.CLIP_EDGE if self.hparams.get('edge_model') else self.CLIP
            for s in range(0, flat_rel.shape[0], bs):
                e = min(flat_rel.shape[0], s + bs)
                chunk = flat_rel[s:e].to(device)
                with autocast(enabled=use_amp):
                    out = enc_rel.encode_image(chunk)
                out = out / out.norm(dim=-1, keepdim=True)
                rel_chunks.append(out.cpu())
                del out, chunk
                torch.cuda.empty_cache()
            rel_clip_encoding = torch.cat(rel_chunks, dim=0).view(*rel_imgs.shape[:-3], -1)

        return obj_clip_encoding, rel_clip_encoding

    def _get_clip_patch_feats(self, images: torch.Tensor) -> torch.Tensor:
        """Extract patch-wise CLIP features without pooling.

        Parameters
        ----------
        images : Tensor
            Tensor of shape ``(B,3,H,W)`` already normalised for CLIP.

        Returns
        -------
        Tensor
            Patch embeddings with shape ``(B,D,h,w)`` where ``h`` and ``w``
            are the number of patches along height and width.
        """
        visual = self.CLIP.visual
        x = visual.conv1(images)  # (B,C,h,w)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # (B,N,C)
        class_emb = visual.class_embedding.to(x.dtype)
        class_emb = class_emb + torch.zeros(x.shape[0], 1, x.shape[-1], device=x.device, dtype=x.dtype)
        x = torch.cat([class_emb, x], dim=1)
        pos = visual.positional_embedding.to(x.dtype)
        x = x + pos
        x = visual.ln_pre(x)
        x = x.permute(1, 0, 2)  # (N,B,C)
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)  # (B,N,C)
        x = visual.ln_post(x)
        if visual.proj is not None:
            x = x @ visual.proj
        patch_tokens = x[:, 1:, :]  # remove class token
        h = images.shape[-2] // visual.conv1.kernel_size[0]
        w = images.shape[-1] // visual.conv1.kernel_size[0]
        return patch_tokens.view(images.shape[0], h, w, -1).permute(0, 3, 1, 2)

    @torch.no_grad()
    def sam_clip_encode_pixels(self, obj_imgs, obj_points, obj_nums, rel_imgs):
        device = self.clip_device if hasattr(self, "clip_device") else obj_imgs.device
        bs = int(self.hparams.get("clip_batch_size", 64))
        use_amp = bool(self.hparams.get("amp", False))

        if SamPredictor is None:
            raise ImportError("segment_anything is required for SAM based encoding")

        if self.sam_predictor is None:
            sam_type = self.hparams.get("sam_model", "vit_h")
            sam_ckpt = self.hparams.get(
                "sam_checkpoint",
                os.path.join(CONF.PATH.CHECKPOINTS, f"sam_{sam_type}.pth"),
            )
            sam = sam_model_registry[sam_type](checkpoint=sam_ckpt)
            sam.to(device)
            self.sam_predictor = SamPredictor(sam)

        feat_dim = self.CLIP.visual.output_dim
        obj_embeddings = torch.zeros((*obj_imgs.shape[:3], feat_dim), device=device)

        for b in range(obj_imgs.shape[0]):
            obj_count = obj_nums[b]
            for n in range(obj_count):
                for v, pts in enumerate(obj_points[b][n]):
                    image_np = obj_imgs[b, n, v].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                    x0 = pts[:, 1].min()
                    y0 = pts[:, 0].min()
                    x1 = pts[:, 1].max()
                    y1 = pts[:, 0].max()
                    box = np.array([x0, y0, x1, y1])
                    self.sam_predictor.set_image(image_np)
                    masks, _, _ = self.sam_predictor.predict(box=box, multimask_output=False)
                    mask = torch.from_numpy(masks[0]).to(device)

                    clip_in = obj_imgs[b, n, v].unsqueeze(0).to(device)
                    with autocast(enabled=use_amp):
                        patches = self._get_clip_patch_feats(clip_in)
                    mask_ds = F.interpolate(mask.float().unsqueeze(0).unsqueeze(0),
                                            size=patches.shape[-2:],
                                            mode='nearest')[0, 0]
                    feat = (patches[0] * mask_ds).reshape(feat_dim, -1)
                    denom = mask_ds.sum().clamp(min=1e-6)
                    obj_embeddings[b, n, v] = feat.sum(dim=-1) / denom

        rel_clip_encoding = None
        if (
            not self.hparams.get('skip_edge_features')
            and self.CLIP is not None
            and not self.hparams.get('blip')
            and not self.hparams.get('llava')
            and rel_imgs is not None
        ):
            rel_img_dim = rel_imgs.shape[-2:]
            flat_rel = rel_imgs.view(-1, 3, *rel_img_dim)
            rel_chunks = []
            enc_rel = self.CLIP_EDGE if self.hparams.get('edge_model') else self.CLIP
            for s in range(0, flat_rel.shape[0], bs):
                e = min(flat_rel.shape[0], s + bs)
                chunk = flat_rel[s:e].to(device)
                with autocast(enabled=use_amp):
                    out = enc_rel.encode_image(chunk)
                out = out / out.norm(dim=-1, keepdim=True)
                rel_chunks.append(out.cpu())
                del out, chunk
                torch.cuda.empty_cache()
            rel_clip_encoding = torch.cat(rel_chunks, dim=0).view(*rel_imgs.shape[:-3], -1)

        return obj_embeddings.to(obj_imgs.device), rel_clip_encoding

    @torch.no_grad()
    def blip_encode_images(self, rel_imgs, batch_size: int = 32):
        rel_images_tensor = np.array(rel_imgs)
        flat_imgs = rel_images_tensor.flatten().tolist()
        rel_embeds = []
        with torch.no_grad():
            for i in range(0, len(flat_imgs), batch_size):
                batch = flat_imgs[i:i + batch_size]
                inputs = self.PROCESSOR(images=batch, text=None, return_tensors="pt").to(self.clip_device)
                rel_embeds.append(self.BLIP.embedd_image(inputs['pixel_values']))
                torch.cuda.empty_cache()

        rel_embeds = torch.cat(rel_embeds, dim=0).view((*rel_images_tensor.shape, 257, 1408))
        return rel_embeds

    @torch.no_grad()
    def llava_encode_images(self, rel_imgs):
        rel_images_tensor = np.array(rel_imgs)
        rel_images_tensor.shape
        with torch.inference_mode():
            images_tensor, image_sizes = self.LLaVA.preprocess_images(rel_images_tensor.flatten().tolist())
            from torch.cuda.amp import autocast
            with autocast(dtype=torch.bfloat16):
                # chunk images to avoid OOM into 4
                chunks = 4
                chunk = torch.chunk(images_tensor, chunks, dim=0)
                chunk_size = torch.chunk(torch.tensor(image_sizes), chunks, dim=0)
                rel_embeds = []
                for i in range(chunks):
                    rel_embeds.append(self.LLaVA.encode_image(chunk[i], chunk_size[i]))
                    torch.cuda.empty_cache()
                # rel_embeds = self.LLaVA.encode_image(images_tensor[::4], image_sizes[::4])
        rel_embeds = torch.stack(rel_embeds).view(-1, 576, 1024).view((*rel_images_tensor.shape, 576, 1024))
        return rel_embeds

    def predict_nodes(self, objects_enc, candidates):
        objects_enc = (objects_enc/objects_enc.norm(dim=-1, keepdim=True))
        if self.hparams['object_context']:
            candidates = [f'A {c} in a scene' for c in candidates]

        candidates = clip.tokenize(candidates).to(self.clip_device)
        if self.hparams['node_model']:
            candidate_emb = self.CLIP_NODE.encode_text(candidates)
        else:
            candidate_emb = self.CLIP.encode_text(candidates)
        candidate_emb = (candidate_emb/candidate_emb.norm(dim=-1, keepdim=True)).float()
        proposal_similarity = (objects_enc @ candidate_emb.t())*100
        return proposal_similarity, torch.softmax(proposal_similarity, dim=-1)

    @torch.no_grad()
    def blip_predict_relationship(self, rel_img_embeddings, predicted_nodes):
        batch_size = 8
        pred_class_dict = [l.rstrip() for l in open(os.path.join(CONF.PATH.R3SCAN_RAW, "relationships_custom.txt"), "r").readlines()]
        queries = [
            f"Describe the relationship between the {o[0]} and the {'other ' if o[0]==o[1] else ''}{o[1]}. Start the response with: the {o[0]}" if o[0] != o[1]
            else f"Describe the relationship between the {o[0]} and the {'other ' if o[0]==o[1] else ''}{o[1]}. Start the response with: the {o[0]}"
            for o in predicted_nodes]

        predicates = []
        results_relationships = []
        # inputs_batch = {}
        mask = torch.isnan(rel_img_embeddings).all(dim=-1).all(dim=-1)
        for batch in range(0, len(rel_img_embeddings), batch_size):
            img_embeds = rel_img_embeddings[batch:batch+batch_size]
            qs = list(queries[batch:batch+batch_size])
            inputs_batch = self.PROCESSOR(images=None, text=qs, return_tensors="pt", padding=True).to(self.clip_device)
            # del inputs['pixel_values']
            outputs = self.BLIP.generate_caption(img_embeds, **inputs_batch,
                                                 do_sample=False,
                                                 num_beams=self.hparams.get('n_beams', 5),
                                                 max_length=20,
                                                 min_length=15,
                                                 repetition_penalty=1.5,
                                                 length_penalty=0.7,
                                                 temperature=1,
                                                 )
            outputs[outputs == -1] = self.PROCESSOR.tokenizer.pad_token_id
            mask_i = mask[batch:batch+batch_size]
            generated_texts = self.PROCESSOR.batch_decode(outputs, skip_special_tokens=True)
            results = [qs[i].split(':')[-1] + ' ' + generated_texts[i].rstrip().split('.')[0].split('\n')[0]
                       for i in range(len(generated_texts))]
            results = np.array(results)
            results[mask_i.squeeze().cpu().numpy()] = 'none'
            results = results.tolist()
            results_pred = []
            for i, (r, objs) in enumerate(zip(results, predicted_nodes[batch:batch+batch_size])):
                if not (objs[0] in r and objs[1] in r):
                    results_pred.append('none')
                else:
                    try:
                        results_pred.append(re.search(f'{objs[0]}(.*){objs[1]}', r).group(1).replace('the ', ''))
                    except:
                        results_pred.append('none')
            results_relationships.extend(results)
            predicates.extend(results_pred)

        return predicates, results_relationships

    @torch.no_grad()
    def llava_predict_relationship(self, rel_img_embeddings, predicted_nodes):
        batch_size = 12
        pred_class_dict = [l.rstrip() for l in open(os.path.join(CONF.PATH.R3SCAN_RAW, "relationships_custom.txt"), "r").readlines()]
        queries = [
            f"Describe the relationship between the {o[0]} and the {'other ' if o[0]==o[1] else ''}{o[1]}. Format your respone like this: {o[0]} [relationship] {o[1]}" if o[0] != o[1]
            else f"Describe the relationship between the {o[0]} and the {'other ' if o[0]==o[1] else ''}{o[1]}. Format your respone like this: the {o[0]} [relationship] the other {o[1]}"
            for o in predicted_nodes]

        predicates = []
        results_relationships = []
        # inputs_batch = {}
        mask = torch.isnan(rel_img_embeddings).all(dim=-1).all(dim=-1)
        for batch in range(0, len(rel_img_embeddings), batch_size):
            img_embeds = rel_img_embeddings[batch:batch+batch_size]
            qs = list(queries[batch:batch+batch_size])

            prompts = self.LLaVA.process_prompt(qs)
            # del inputs['pixel_values']
            outputs = self.LLaVA.generate_output(img_embeds, prompts,
                                                 temperature=0,
                                                 top_p=None,
                                                 num_beams=2,
                                                 max_new_tokens=30)

            results = [outputs[i].rstrip().split('.')[0].rstrip().split(',')[0].split('\n')[0] for i in range(len(outputs))]
            results = np.array(results)
            # results[mask_i.squeeze().cpu().numpy()]='none'
            results = results.tolist()
            # results_pred = [re.search(f'{objs[0]}(.*){objs[1]}', r).group(1) if (objs[0] in r and objs[1] in r) else 'none' for r,objs in zip(results,predicted_nodes[batch:batch+batch_size])]
            results_pred = []
            for i, (r, objs) in enumerate(zip(results, predicted_nodes[batch:batch+batch_size])):
                if not (objs[0] in r and objs[1] in r):
                    results_pred.append('none')
                else:
                    try:
                        results_pred.append(re.search(f'{objs[0]}(.*){objs[1]}', r).group(1).replace('the ', ''))
                    except:
                        results_pred.append('none')
            results_relationships.extend(results)
            predicates.extend(results_pred)

        return predicates, results_relationships
