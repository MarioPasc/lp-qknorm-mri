"""Lesion-mask tokenization and window partition utilities.

Maps ground-truth masks from image space to stage-0 token space, then
partitions into windows matching MONAI's ``window_partition`` layout.
Also provides ``compute_logits_with_bias`` to reconstruct full pre-softmax
logits for Probe 4.

Stage-0 geometry (MONAI 2D SwinUNETR, patch_size=2)
----------------------------------------------------
- Image: ``(B, 1, H, W)`` where ``H = W = 224``.
- Token grid: ``(B, H_tok, W_tok)`` where ``H_tok = H // 2 = 112``.
- Windows: ``(B * nW, W² )`` where ``nW = (H_tok // 7)² = 256``, ``W² = 49``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor


if TYPE_CHECKING:
    from lpqknorm.models.hooks import AttentionCapture


def mask_to_token_flags(
    mask: Tensor,
    patch_stride: tuple[int, int] = (2, 2),
) -> Tensor:
    """Downsample a binary mask from image space to stage-0 token space.

    Uses max-pooling with stride equal to the patch embedding stride.
    A token is flagged as lesion if **any** pixel in its receptive field
    patch is positive (logical OR semantics via max-pool).

    Parameters
    ----------
    mask : Tensor
        Shape ``(B, 1, H_img, W_img)``, float32 with values in ``{0, 1}``.
    patch_stride : tuple[int, int]
        Effective stride from image pixels to stage-0 tokens.  Default
        ``(2, 2)`` for MONAI's ``patch_size=2``.

    Returns
    -------
    Tensor
        Shape ``(B, H_tok * W_tok)``, dtype ``bool``, in raster order.
    """
    pooled = F.max_pool2d(mask, kernel_size=patch_stride, stride=patch_stride)
    return (pooled > 0.5).squeeze(1).reshape(mask.shape[0], -1)


def window_partition_flags(
    token_flags: Tensor,
    img_hw_tok: tuple[int, int],
    window_size: int = 7,
    shift_size: int = 0,
) -> Tensor:
    """Partition token-level lesion flags into windows.

    Replicates MONAI's ``window_partition`` layout exactly so that the
    output tensor is aligned with ``AttentionCapture`` tensors.

    For SW-MSA blocks (``shift_size > 0``), applies a cyclic roll of
    ``(-shift_size, -shift_size)`` to the token grid before partitioning,
    matching ``SwinTransformerBlock.forward_part1``.

    Parameters
    ----------
    token_flags : Tensor
        Shape ``(B, H_tok * W_tok)``, dtype ``bool``.
    img_hw_tok : tuple[int, int]
        ``(H_tok, W_tok)``.  Both must be divisible by ``window_size``.
    window_size : int
        Window size (7 for all MONAI default stages).
    shift_size : int
        Cyclic shift size.  ``0`` for W-MSA, ``window_size // 2`` for SW-MSA.

    Returns
    -------
    Tensor
        Shape ``(B * nW, window_size²)``, dtype ``bool``.
    """
    h_tok, w_tok = img_hw_tok
    b = token_flags.shape[0]
    ws = window_size

    grid = token_flags.view(b, h_tok, w_tok)

    if shift_size > 0:
        grid = torch.roll(grid, shifts=(-shift_size, -shift_size), dims=(-2, -1))

    # Window partition: (B, H_tok//ws, ws, W_tok//ws, ws) → permute → flatten
    grid = grid.view(b, h_tok // ws, ws, w_tok // ws, ws)
    grid = grid.permute(0, 1, 3, 2, 4).contiguous()
    return grid.view(-1, ws * ws)


def compute_logits_with_bias(capture: AttentionCapture) -> Tensor:
    """Return full pre-softmax logits including relative position bias.

    Parameters
    ----------
    capture : AttentionCapture
        Must have non-None ``logits`` and ``relative_position_bias``.

    Returns
    -------
    Tensor
        Shape ``(B*nW, num_heads, n, n)``.

    Raises
    ------
    ValueError
        If ``logits`` or ``relative_position_bias`` is ``None``.
    """
    if capture.logits is None:
        msg = "capture.logits is None"
        raise ValueError(msg)
    if capture.relative_position_bias is None:
        msg = "capture.relative_position_bias is None"
        raise ValueError(msg)
    return capture.logits + capture.relative_position_bias
