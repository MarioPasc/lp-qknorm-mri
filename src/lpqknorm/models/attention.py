"""Lp-QKNorm windowed self-attention module.

Drop-in replacement for MONAI's ``WindowAttention`` (v1.5.2) where the QK
scaling step is replaced by Lp-normalisation followed by a learnable scalar.

Verified against MONAI source at:
    monai.networks.nets.swin_unetr.WindowAttention
    MONAI v1.5.2, confirmed 2025-04 in the lpqknorm conda env.

Canonical input/output shapes (2D SwinUNETR, window_size=(7, 7))
-----------------------------------------------------------------
    Input  x    : (B * nW, n, C)          nW = number of windows, n = W0*W1 = 49, C = dim
    Input  mask : (nW, n, n) | None       shift-window attention mask
    Output      : (B * nW, n, C)          same shape as input x

Internal shapes during forward:
    qkv         : (B*nW, n, 3, num_heads, head_dim)   head_dim = C // num_heads
    q, k, v     : (B*nW, num_heads, n, head_dim)      after permute
    q_hat, k_hat: same as q, k
    attn (raw)  : (B*nW, num_heads, n, n)
    attn (post) : (B*nW, num_heads, n, n)             after softmax

What was changed relative to MONAI's WindowAttention
-----------------------------------------------------
    Original (MONAI):
        q = q * self.scale           # scale = head_dim ** -0.5
        attn = q @ k.transpose(-2, -1)

    Replaced by:
        q_hat, k_hat, alpha = self.lp_qknorm(q, k)
        attn = alpha * (q_hat @ k_hat.transpose(-2, -1))

Nothing else was modified.  The relative position bias, attention dropout,
value aggregation, output projection, and mask handling are byte-for-byte
identical to MONAI v1.5.2.

Architecture invariant
----------------------
This module does NOT subclass MONAI's ``WindowAttention`` to avoid any
unintended inheritance of internal behaviour.  All attributes are constructed
from scratch using the same logic as MONAI v1.5.2 so that weight dicts are
structurally compatible (``qkv``, ``proj``, ``relative_position_bias_table``,
``relative_position_index``).

Capture dict (``self._capture``)
---------------------------------
After each forward pass the following tensors are stored in ``self._capture``
as *references* (not clones) for efficient hook access:

    ``q``       : raw query before normalisation        (B*nW, nh, n, hd)
    ``k``       : raw key before normalisation          (B*nW, nh, n, hd)
    ``q_hat``   : Lp-normalised query                   (B*nW, nh, n, hd)
    ``k_hat``   : Lp-normalised key                     (B*nW, nh, n, hd)
    ``alpha``   : positive scale factor                 scalar Tensor
    ``logits``  : attention logits BEFORE rel-pos bias  (B*nW, nh, n, n)
    ``attention``: attention weights AFTER softmax      (B*nW, nh, n, n)

Callers that need detached, cloned copies (e.g., ``AttentionHookRegistry``)
should call ``.clone().detach()`` on each captured tensor.

References
----------
- Henry et al. *Query-Key Normalization for Transformers*. EMNLP 2020.
  arXiv:2010.04245.
- López-Rubio et al. *Enhanced QKNorm with the Lp Norm*. 2026.
  arXiv:2602.05006.
- Liu et al. *Swin Transformer*. ICCV 2021. arXiv:2103.14030.
- Hatamizadeh et al. *Swin UNETR*. BrainLes 2021. arXiv:2201.01266.
- MONAI v1.5.2 source: ``monai.networks.nets.swin_unetr.WindowAttention``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from torch import Tensor

from lpqknorm.models.lp_qknorm import LpQKNorm, LpQKNormConfig


if TYPE_CHECKING:
    from collections.abc import Sequence

# ---------------------------------------------------------------------------
# trunc_normal_ import with fallback
# ---------------------------------------------------------------------------
# Preferred source: MONAI's own trunc_normal_ (same version used by the stock
# WindowAttention we are replacing).  Fall back to PyTorch's built-in for
# environments where MONAI is unavailable (e.g., doctests, isolated unit tests).
# Both have compatible signatures; the wrapper below normalises them.


def _make_trunc_normal() -> Any:
    """Return a callable ``trunc_normal_(tensor, std=...)`` from MONAI or PyTorch."""
    try:
        from monai.networks.layers import (  # type: ignore[attr-defined]
            trunc_normal_ as _monai_tn,
        )

        return _monai_tn
    except (ImportError, AttributeError):  # pragma: no cover
        from torch.nn.init import trunc_normal_ as _torch_tn

        return _torch_tn


_trunc_normal_ = _make_trunc_normal()

__all__ = ["LpWindowAttention"]

logger = logging.getLogger(__name__)


class LpWindowAttention(nn.Module):
    """Windowed self-attention with Lp query-key normalisation.

    A drop-in replacement for ``monai.networks.nets.swin_unetr.WindowAttention``
    (MONAI v1.5.2).  The only architectural change is that the standard
    ``q * scale`` dot-product is replaced by Lp-normalised attention:

        q_hat, k_hat, alpha = self.lp_qknorm(q, k)
        attn = alpha * (q_hat @ k_hat.transpose(-2, -1))

    All other components (QKV projection, relative position bias, attention
    mask, dropout, output projection) are preserved verbatim.

    Parameters
    ----------
    dim : int
        Number of input feature channels.  Must be divisible by ``num_heads``.
    num_heads : int
        Number of attention heads.
    window_size : Sequence[int]
        Local attention window size, e.g. ``(7, 7)`` for 2D.  Only 2D
        windows are supported; for 3D windows use the stock MONAI module.
    qkv_bias : bool
        If ``True``, adds a learnable bias to the QKV linear projection.
        Default ``False`` (MONAI default).
    attn_drop : float
        Dropout probability applied to attention weights.  Default ``0.0``.
    proj_drop : float
        Dropout probability applied to the output projection.  Default ``0.0``.
    lp_cfg : LpQKNormConfig
        Configuration for :class:`~lpqknorm.models.lp_qknorm.LpQKNorm`.

    Attributes
    ----------
    dim : int
    window_size : Sequence[int]
    num_heads : int
    qkv : nn.Linear
        Fused QKV projection: ``(dim) -> (dim * 3)``.
    proj : nn.Linear
        Output projection: ``(dim) -> (dim)``.
    attn_drop : nn.Dropout
    proj_drop : nn.Dropout
    softmax : nn.Softmax
    relative_position_bias_table : nn.Parameter
        Shape ``((2*W0-1)*(2*W1-1), num_heads)``.
    relative_position_index : Tensor
        Buffer of shape ``(W0*W1, W0*W1)`` holding flattened relative
        position indices.
    lp_qknorm : LpQKNorm
        The Lp normalisation + scaling module.
    _capture : dict[str, Tensor]
        Dictionary populated after each forward pass with references to
        intermediate tensors (see module docstring for keys).

    Raises
    ------
    ValueError
        If ``len(window_size) != 2``.
    ValueError
        If ``dim % num_heads != 0``.

    Notes
    -----
    This class does **not** subclass ``WindowAttention`` to ensure that MONAI
    internals cannot silently change our behaviour across MONAI version updates.
    The constructor logic and ``forward`` flow are copied verbatim from MONAI
    v1.5.2 and then minimally patched, with the single replacement clearly
    delimited by comments.

    Weight compatibility: the parameter names ``qkv.weight``, ``qkv.bias``
    (if enabled), ``proj.weight``, ``proj.bias``, and
    ``relative_position_bias_table`` are identical to MONAI's
    ``WindowAttention``, enabling direct ``state_dict`` transfer.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Sequence[int],
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        lp_cfg: LpQKNormConfig = LpQKNormConfig(p=2.0),
    ) -> None:
        super().__init__()

        if len(window_size) != 2:
            raise ValueError(
                f"LpWindowAttention only supports 2D windows; "
                f"got window_size of length {len(window_size)}."
            )
        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads})."
            )

        # ------------------------------------------------------------------ #
        # Core attributes — identical to MONAI WindowAttention.__init__       #
        # ------------------------------------------------------------------ #
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        # NOTE: self.scale is kept for structural compatibility / external reads,
        # but is NOT used in forward; it is superseded by lp_qknorm.
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        # ------------------------------------------------------------------ #
        # Relative position bias — identical to MONAI 2D branch               #
        # ------------------------------------------------------------------ #
        # Parameter shape: ((2*W0-1)*(2*W1-1), num_heads).
        # Initialised with trunc_normal_(std=0.02) after all attributes are set.
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                num_heads,
            )
        )

        # Build relative position index buffer — verbatim from MONAI v1.5.2.
        # The indexing="ij" kwarg guard replicates MONAI's own mesh_args check.
        mesh_args = torch.meshgrid.__kwdefaults__

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        if mesh_args is not None:
            coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        else:
            coords = torch.stack(torch.meshgrid(coords_h, coords_w))

        coords_flatten = torch.flatten(coords, 1)  # (2, W0*W1)
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # (2, n, n)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (n, n, 2)
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # (n, n)

        self.register_buffer("relative_position_index", relative_position_index)

        # ------------------------------------------------------------------ #
        # Projections and dropouts — identical to MONAI                       #
        # ------------------------------------------------------------------ #
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Initialise relative position bias table with truncated normal (std=0.02).
        _trunc_normal_(self.relative_position_bias_table, std=0.02)

        self.softmax = nn.Softmax(dim=-1)

        # ------------------------------------------------------------------ #
        # Lp-QKNorm module (the only addition relative to MONAI)              #
        # ------------------------------------------------------------------ #
        self.lp_qknorm = LpQKNorm(lp_cfg)

        # Capture dict for hook access — populated by forward().
        # Keys: q, k, q_hat, k_hat, alpha, logits, attention.
        self._capture: dict[str, Tensor] = {}

        logger.debug(
            "LpWindowAttention initialised: dim=%d, num_heads=%d, window_size=%s, p=%.2f",
            dim,
            num_heads,
            window_size,
            lp_cfg.p,
        )

    def forward(self, x: Tensor, mask: Tensor | None) -> Tensor:
        """Compute windowed self-attention with Lp-QKNorm.

        Parameters
        ----------
        x : Tensor
            Input feature tensor of shape ``(B * nW, n, C)`` where ``B`` is
            the batch size, ``nW`` the number of windows, ``n = W0 * W1`` the
            number of tokens per window, and ``C = dim``.
        mask : Tensor | None
            Shift-window attention mask of shape ``(nW, n, n)`` or ``None``
            for non-shifted windows.  Added to attention logits (as a large
            negative number for masked positions) before softmax.

        Returns
        -------
        Tensor
            Output tensor of shape ``(B * nW, n, C)``, same as input.

        Notes
        -----
        The ONLY change relative to MONAI's ``WindowAttention.forward`` is the
        replacement of::

            q = q * self.scale
            attn = q @ k.transpose(-2, -1)

        with::

            q_hat, k_hat, alpha = self.lp_qknorm(q, k)
            attn = alpha * (q_hat @ k_hat.transpose(-2, -1))

        All subsequent steps (relative position bias, mask, softmax, dropout,
        value aggregation, output projection) are identical to MONAI v1.5.2.

        After this method returns, ``self._capture`` contains references to
        ``q``, ``k``, ``q_hat``, ``k_hat``, ``alpha``, ``logits`` (before
        relative-position bias), and ``attention`` (after softmax).
        """
        b, n, c = x.shape

        # ------------------------------------------------------------------ #
        # QKV projection and head splitting — identical to MONAI               #
        # ------------------------------------------------------------------ #
        qkv = (
            self.qkv(x)
            .reshape(b, n, 3, self.num_heads, c // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (b, num_heads, n, head_dim)

        # Store raw Q and K for probe access before any normalisation.
        self._capture["q"] = q
        self._capture["k"] = k

        # ------------------------------------------------------------------ #
        # === REPLACEMENT: Lp-QKNorm instead of q * scale =================== #
        # Original MONAI code (DO NOT RESTORE):                               #
        #     q = q * self.scale                                               #
        #     attn = q @ k.transpose(-2, -1)                                  #
        # ------------------------------------------------------------------ #
        q_hat, k_hat, alpha = self.lp_qknorm(q, k)
        attn = alpha * (q_hat @ k_hat.transpose(-2, -1))
        # ------------------------------------------------------------------ #
        # === END REPLACEMENT ================================================ #
        # ------------------------------------------------------------------ #

        # Store Lp-normalised vectors, scale, and pre-bias logits.
        self._capture["q_hat"] = q_hat
        self._capture["k_hat"] = k_hat
        self._capture["alpha"] = alpha
        self._capture["logits"] = attn  # BEFORE relative position bias

        # ------------------------------------------------------------------ #
        # Relative position bias — identical to MONAI                         #
        # ------------------------------------------------------------------ #
        # relative_position_index is a registered buffer (LongTensor).
        # The type: ignore suppresses mypy's "Tensor not callable" false positive
        # caused by nn.Module's __getattr__ returning Tensor for buffers.
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.clone()[:n, :n].reshape(-1)  # type: ignore[operator]
        ].reshape(n, n, -1)
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # (num_heads, n, n)
        attn = attn + relative_position_bias.unsqueeze(0)

        # ------------------------------------------------------------------ #
        # Attention mask and softmax — identical to MONAI                     #
        # ------------------------------------------------------------------ #
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b // nw, nw, self.num_heads, n, n) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        # Store post-softmax attention weights for probe access.
        self._capture["attention"] = attn

        # ------------------------------------------------------------------ #
        # Value aggregation and output projection — identical to MONAI        #
        # ------------------------------------------------------------------ #
        attn = self.attn_drop(attn).to(v.dtype)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
