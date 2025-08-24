import sys
sys.modules.pop("torch", None)
import torch

def _old_reduce(clip_rel_emb, clip_rel_mask, clip_rel2frame_mask):
    tmp = clip_rel_emb.clone()
    tmp.masked_fill_(~clip_rel_mask.unsqueeze(-1).unsqueeze(-1), float("nan"))
    reduced = torch.nanmean(tmp, dim=1)
    out = torch.zeros_like(reduced)
    out[clip_rel2frame_mask > 0] = reduced[clip_rel2frame_mask > 0]
    return out

def _chunked_reduce(clip_rel_emb, clip_rel_mask, clip_rel2frame_mask, rel_chunk_size):
    rel_count = clip_rel_emb.size(0)
    tokens, dim = clip_rel_emb.size(2), clip_rel_emb.size(3)
    out = torch.empty((rel_count, tokens, dim))
    for start in range(0, rel_count, rel_chunk_size):
        end = min(start + rel_chunk_size, rel_count)
        rel_slice = clip_rel_emb[start:end].clone()
        mask_slice = clip_rel_mask[start:end]
        rel_slice.masked_fill_(~mask_slice.unsqueeze(-1).unsqueeze(-1), float("nan"))
        reduced = torch.nanmean(rel_slice, dim=1)
        valid = clip_rel2frame_mask[start:end] > 0
        out[start:end][valid] = reduced[valid]
    return out

def test_chunked_relation_reduction_matches_full():
    rel_count, frames, tokens, dim = 8, 3, 2, 4
    rel2frame_mask = torch.tensor([3, 2, 1, 3, 2, 1, 3, 2])
    clip_rel_emb = torch.randn(rel_count, frames, tokens, dim)
    clip_rel_mask = (
        torch.arange(frames).unsqueeze(0) < rel2frame_mask.unsqueeze(1)
    )
    full = _old_reduce(clip_rel_emb, clip_rel_mask, rel2frame_mask)
    chunked = _chunked_reduce(clip_rel_emb, clip_rel_mask, rel2frame_mask, rel_chunk_size=2)
    assert torch.allclose(full, chunked, equal_nan=True)
