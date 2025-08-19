import math
from typing import List

import torch
from torch.cuda.amp import autocast


def encode_node_images_in_batches(imgs: torch.Tensor, model: torch.nn.Module, device: torch.device,
                                   batch_size: int, *, amp: bool = True, sync_cuda: bool = False) -> torch.Tensor:
    """Encode images with a CLIP-like model using micro-batches.

    Parameters
    ----------
    imgs: ``Tensor``
        Image tensor of shape ``(..., C, H, W)``.
    model: ``nn.Module``
        Model exposing ``encode_image`` method.
    device: ``torch.device``
        Target device for the model.
    batch_size: ``int``
        Micro-batch size for encoding.
    amp: ``bool``
        Enable ``torch.cuda.amp.autocast`` during encoding.
    sync_cuda: ``bool``
        If ``True`` synchronises CUDA after each step for debugging.

    Returns
    -------
    ``Tensor``
        Encoded features with shape ``(*imgs.shape[:-3], D)``.
    """
    orig_shape = imgs.shape[:-3]
    flat = imgs.view(-1, *imgs.shape[-3:])
    total = flat.shape[0]
    steps = math.ceil(total / batch_size)
    print(f"encode_node_images_in_batches: images={total} batch={batch_size} steps={steps}")

    feats_cpu: List[torch.Tensor] = []
    peak_mem = 0
    model.eval()
    with torch.inference_mode():
        for start in range(0, total, batch_size):
            end = min(total, start + batch_size)
            batch = flat[start:end].to(device, non_blocking=True)
            with autocast(enabled=amp):
                out = model.encode_image(batch)
            out = out / out.norm(dim=-1, keepdim=True)
            feats_cpu.append(out.detach().cpu())
            if sync_cuda:
                torch.cuda.synchronize(device)
            peak_mem = max(peak_mem, torch.cuda.max_memory_allocated(device))
            del batch, out
            torch.cuda.empty_cache()
    print(f"encode_node_images_in_batches: peak_memory={peak_mem/1024**2:.2f} MiB")
    return torch.cat(feats_cpu, dim=0).view(*orig_shape, -1)
