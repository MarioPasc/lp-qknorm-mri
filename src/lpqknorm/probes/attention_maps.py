"""Attention-map reconstruction, rollout, and overlay helpers.

Three primitives used by Phase 5 qualitative figures and by post-hoc
interpretability scripts:

- :func:`reconstruct_query_heatmap` maps a single (window, intra-window)
  attention row back into a ``(H_tok, W_tok)`` image in token space.
- :func:`attention_rollout` implements the residual-augmented Abnar &
  Zuidema rollout ``R_L = prod_l (1/2)(A_l + I)``.
- :func:`overlay_figure` produces a 3-panel publication figure
  (image | mask | heatmap overlay) using matplotlib.

References
----------
- Abnar & Zuidema.  *Quantifying Attention Flow in Transformers*.
  ACL 2020.  arXiv:2005.00928.
- Chefer, Gur, Wolf.  *Transformer Interpretability Beyond Attention
  Visualization*.  CVPR 2021.  arXiv:2012.09838.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from collections.abc import Sequence

    import matplotlib.axes
    import matplotlib.figure
    from torch import Tensor


def reconstruct_query_heatmap(
    attention: Tensor,
    query_idx: int,
    shift: int,
    grid_hw: tuple[int, int],
    window_size: int,
) -> Tensor:
    """Reconstruct a single query's attention into a ``(H_tok, W_tok)`` map.

    Parameters
    ----------
    attention : Tensor
        Shape ``(n_win, H_heads, W², W²)`` or ``(n_win, W², W²)`` — post-
        softmax attention from one block.  If ``H_heads > 1``, the returned
        heatmap is the mean over heads.
    query_idx : int
        Flat token index into the token grid ``(H_tok, W_tok)`` **in the
        un-shifted (original) coordinate system**.  Row-major (``y * W_tok + x``).
    shift : int
        Cyclic shift amount applied by SW-MSA forward partition
        (``0`` for W-MSA, ``W/2`` for SW-MSA).  Used to map from the
        original grid to the rolled grid in which the windows live.
    grid_hw : tuple[int, int]
        ``(H_tok, W_tok)`` — token-grid dimensions.
    window_size : int
        Window side length ``W``.

    Returns
    -------
    Tensor
        Shape ``(H_tok, W_tok)``, dtype float32.  Sums to the attention-row
        sum (close to 1 for a softmaxed row).
    """
    if attention.ndim == 4:
        attn = attention.mean(dim=1)  # (n_win, W², W²)
    else:
        attn = attention

    h_tok, w_tok = grid_hw
    w = window_size
    if h_tok % w != 0 or w_tok % w != 0:
        raise ValueError(f"Grid {grid_hw} not divisible by window_size={w}.")
    n_win_cols = w_tok // w

    # Original (un-shifted) coordinates of the query.
    y_orig = query_idx // w_tok
    x_orig = query_idx % w_tok

    # Post-shift (rolled) coordinates — forward partition rolls by -shift.
    y_shift = (y_orig - shift) % h_tok
    x_shift = (x_orig - shift) % w_tok

    win_r = y_shift // w
    win_c = x_shift // w
    win_idx = win_r * n_win_cols + win_c
    intra_r = y_shift % w
    intra_c = x_shift % w
    intra_idx = intra_r * w + intra_c

    row = attn[win_idx, intra_idx].to(torch.float32)  # (W²,)

    # Scatter the row into the post-shift grid.
    heatmap_shift = torch.zeros(h_tok, w_tok, dtype=torch.float32)
    base_y = win_r * w
    base_x = win_c * w
    for j in range(w * w):
        jr = j // w
        jc = j % w
        heatmap_shift[base_y + jr, base_x + jc] = row[j]

    if shift == 0:
        return heatmap_shift

    # Roll back so the heatmap lives in the original coordinate system.
    return torch.roll(heatmap_shift, shifts=(shift, shift), dims=(-2, -1))


def attention_rollout(
    attentions: Sequence[Tensor],
    add_residual: bool = True,
) -> Tensor:
    """Abnar–Zuidema attention rollout with residual correction.

    Parameters
    ----------
    attentions : Sequence[Tensor]
        Sequence of attention matrices, one per layer, each shape
        ``(n, n)`` or ``(H_heads, n, n)`` (heads are averaged) or
        ``(B, H_heads, n, n)`` (heads are averaged; batch preserved).
    add_residual : bool
        If ``True`` (default) apply the residual correction
        ``A_l ← 0.5 * (A_l + I)`` and re-normalise rows before
        multiplication, following the standard Abnar–Zuidema formulation.

    Returns
    -------
    Tensor
        Shape ``(n, n)`` (or ``(B, n, n)`` if batched).  The product
        ``A_L' · A_{L-1}' · ... · A_1'``.
    """
    if not attentions:
        raise ValueError("attention_rollout requires at least one layer")

    mats: list[Tensor] = []
    for a in attentions:
        if a.ndim == 3:  # (H, n, n)
            m = a.mean(dim=0)
        elif a.ndim == 4:  # (B, H, n, n)
            m = a.mean(dim=1)
        else:
            m = a
        mats.append(m.to(torch.float32))

    n = mats[0].shape[-1]
    if add_residual:
        eye = torch.eye(n, dtype=mats[0].dtype, device=mats[0].device)
        if mats[0].ndim == 3:
            eye = eye.expand(mats[0].shape[0], n, n)
        mats = [0.5 * (m + eye) for m in mats]
        # Row-renormalise so each row still sums to 1.
        mats = [m / m.sum(dim=-1, keepdim=True).clamp(min=1e-12) for m in mats]

    rollout = mats[0]
    for m in mats[1:]:
        rollout = m @ rollout
    return rollout


def overlay_figure(
    image: Tensor,
    mask: Tensor,
    heatmap: Tensor,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure:
    """Three-panel figure: input | mask | attention overlay.

    Parameters
    ----------
    image : Tensor
        ``(C, H, W)`` or ``(H, W)``.  Channel 0 is shown if ``C > 1``.
    mask : Tensor
        ``(H, W)`` binary lesion mask.
    heatmap : Tensor
        ``(H_tok, W_tok)`` attention heatmap.  Will be bilinearly
        resampled to ``(H, W)``.
    ax : matplotlib.axes.Axes, optional
        Unused; retained for backward compatibility with callers that
        supply an axis (the helper always creates a 3-panel figure).

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt  # local import keeps optional dep lazy
    import torch.nn.functional as F  # noqa: N812

    del ax  # unused
    img = image.detach().cpu()
    if img.ndim == 3:
        img = img[0]
    msk = mask.detach().cpu().float()
    hm = heatmap.detach().cpu().float().unsqueeze(0).unsqueeze(0)
    hm_up = F.interpolate(
        hm,
        size=img.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )[0, 0]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img.numpy(), cmap="gray")
    axes[0].set_title("input")
    axes[0].axis("off")
    axes[1].imshow(msk.numpy(), cmap="Reds", vmin=0, vmax=1)
    axes[1].set_title("lesion mask")
    axes[1].axis("off")
    axes[2].imshow(img.numpy(), cmap="gray")
    axes[2].imshow(hm_up.numpy(), cmap="jet", alpha=0.5)
    axes[2].set_title("attention overlay")
    axes[2].axis("off")
    fig.tight_layout()
    return fig
