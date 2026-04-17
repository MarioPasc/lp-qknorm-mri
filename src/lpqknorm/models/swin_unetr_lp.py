"""Build a 2D SwinUNETR with Lp-QKNorm windowed attention.

Patching strategy: **Option A — monkey-patch after construction.**

1. Construct a stock ``monai.networks.nets.SwinUNETR(spatial_dims=2, ...)``.
2. Walk the module tree, identify every ``WindowAttention`` instance.
3. Replace each with :class:`~lpqknorm.models.attention.LpWindowAttention`,
   copying ``qkv``, ``proj``, ``relative_position_bias_table``, and
   ``relative_position_index`` from the stock module.
4. Return the patched model.

If ``lp_cfg`` is ``None``, the stock model is returned unmodified (vanilla
softmax baseline — the "no QKNorm" lower-bound control).

Module tree (MONAI v1.5.2, ``spatial_dims=2``, ``depths=(2,2,2,2)``)::

    SwinUNETR.swinViT (SwinTransformer)
      .layers{1,2,3,4}[0]  (BasicLayer)
        .blocks[i]  (SwinTransformerBlock)
          .attn  (WindowAttention)   <-- replaced by LpWindowAttention

Verified against MONAI 1.5.2: 8 WindowAttention modules total (4 stages x 2
blocks each), all with ``qkv_bias=True``, ``attn_drop=0.0``, ``proj_drop=0.0``.

Rationale for Option A over Option B (vendoring):
    MONAI's ``WindowAttention`` is a self-contained ``nn.Module`` with no
    cross-module references beyond its parent ``SwinTransformerBlock``.
    Replacing it via ``setattr`` on the parent block is clean, deterministic,
    and preserves all other weights (patch embedding, downsampling, decoder).
    Vendoring would duplicate several hundred lines of MONAI source and
    introduce a maintenance burden without any benefit.

References
----------
- Hatamizadeh et al. *Swin UNETR*. BrainLes 2021. arXiv:2201.01266.
- MONAI v1.5.2 source: ``monai.networks.nets.swin_unetr``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR  # type: ignore[attr-defined]
from monai.networks.nets.swin_unetr import WindowAttention

from lpqknorm.models.attention import LpWindowAttention
from lpqknorm.models.init import AlphaInitScheme, InitScheme, initialize_model
from lpqknorm.utils.exceptions import PatchingError, WeightTransferError


if TYPE_CHECKING:
    from lpqknorm.models.lp_qknorm import LpQKNormConfig


__all__ = ["build_swin_unetr_lp"]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_swin_unetr_lp(
    img_size: tuple[int, int],
    in_channels: int,
    out_channels: int,
    feature_size: int = 24,
    lp_cfg: LpQKNormConfig | None = None,
    patch_base: Literal["monai"] = "monai",
    *,
    init_scheme: InitScheme = "scratch_trunc_normal",
    linear_init_std: float = 0.02,
    alpha_init_scheme: AlphaInitScheme = "log_dk",
    alpha_init_fixed: float | None = None,
) -> nn.Module:
    """Build a 2D SwinUNETR, optionally patched with Lp-QKNorm attention.

    Parameters
    ----------
    img_size : tuple[int, int]
        Spatial dimensions of the input images ``(H, W)``.  Kept in the
        function signature for documentation and forward-compatibility but
        **not** forwarded to ``SwinUNETR`` (MONAI validates input size at
        forward time, not at construction time).
    in_channels : int
        Number of input channels (e.g. 1 for single-modality T1w MRI).
    out_channels : int
        Number of output segmentation classes.
    feature_size : int
        Base embedding dimension.  Must be divisible by 12 (MONAI
        constraint).  Default ``24``.
    lp_cfg : LpQKNormConfig or None
        If provided, every ``WindowAttention`` in the encoder is replaced
        by :class:`~lpqknorm.models.attention.LpWindowAttention` with this
        configuration.  If ``None``, the stock MONAI model is returned
        unmodified (vanilla softmax baseline).
    patch_base : ``"monai"``
        Base model source.  Currently only ``"monai"`` is supported.
    init_scheme : {"scratch_trunc_normal", "pretrained_ssl"}
        Weight-initialization regime.  ``"scratch_trunc_normal"`` is the
        primary from-scratch run and is applied via ``model.apply`` after
        patching.  ``"pretrained_ssl"`` leaves non-``alpha_raw`` tensors
        untouched (the caller is expected to ``load_state_dict`` a Tang
        et al. (2022) SSL checkpoint prior to calling this function with
        the SSL scheme; ``alpha_raw`` is still seeded here).
    linear_init_std : float
        Std for ``trunc_normal_`` on ``nn.Linear`` / ``nn.Conv{2,3}d`` /
        ``relative_position_bias_table`` weights.  Default ``0.02`` per
        Swin (Liu et al., 2021) and ViT (Dosovitskiy et al., 2021).
    alpha_init_scheme : {"log_dk", "sqrt_dk", "fixed"}
        Scheme for ``LpQKNorm.alpha_raw``.  Default ``"log_dk"`` (Henry
        et al., 2020).
    alpha_init_fixed : float or None
        Required iff ``alpha_init_scheme == "fixed"``.

    Returns
    -------
    nn.Module
        The (possibly patched) ``SwinUNETR`` model.

    Raises
    ------
    PatchingError
        If no ``WindowAttention`` modules are found in the constructed model
        (unexpected MONAI internal change) when ``lp_cfg`` is not ``None``.
    LpInitError
        If the initialization spec is invalid (see
        :func:`lpqknorm.models.init.initialize_model`).
    """
    # -- Step 1: construct stock MONAI SwinUNETR --------------------------
    model = SwinUNETR(
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=feature_size,
        spatial_dims=2,
    )

    if lp_cfg is None:
        logger.info(
            "build_swin_unetr_lp: lp_cfg=None — returning stock MONAI SwinUNETR "
            "(vanilla baseline)."
        )
        # Vanilla baseline: no LpQKNorm to seed, but still re-init the shared
        # trunk (trunc_normal_ linear / conv / layernorm) for determinism when
        # the caller opts into scratch initialization.  Skipping silently when
        # the caller chose the SSL scheme.
        if init_scheme == "scratch_trunc_normal":
            initialize_model(
                model,
                init_scheme=init_scheme,
                linear_init_std=linear_init_std,
                alpha_init_scheme=alpha_init_scheme,
                alpha_init_fixed=alpha_init_fixed,
            )
        return model

    # -- Step 2: find and replace all WindowAttention modules -------------
    attention_modules = _find_attention_modules(model)

    if not attention_modules:
        raise PatchingError(
            "No WindowAttention modules found in the SwinUNETR model. "
            "This may indicate an incompatible MONAI version.",
            details={"monai_model_type": type(model).__name__},
        )

    for parent, attr_name, stock_attn in attention_modules:
        lp_attn = LpWindowAttention(
            dim=stock_attn.dim,
            num_heads=stock_attn.num_heads,
            window_size=stock_attn.window_size,
            qkv_bias=stock_attn.qkv.bias is not None,
            attn_drop=stock_attn.attn_drop.p,
            proj_drop=stock_attn.proj_drop.p,
            lp_cfg=lp_cfg,
        )
        _copy_weights(stock_attn, lp_attn)
        setattr(parent, attr_name, lp_attn)

    n_replaced = len(attention_modules)
    logger.info(
        "build_swin_unetr_lp: replaced %d WindowAttention module(s) with "
        "LpWindowAttention (p=%.2f).",
        n_replaced,
        lp_cfg.p,
    )

    # Sanity check: no stock WindowAttention should remain.
    remaining = sum(
        1 for _, m in model.named_modules() if isinstance(m, WindowAttention)
    )
    if remaining > 0:
        raise PatchingError(
            f"{remaining} WindowAttention module(s) were not replaced. "
            f"Only {n_replaced} were found and patched.",
            details={"remaining": remaining, "replaced": n_replaced},
        )

    # -- Step 3: apply weight-initialization spec --------------------------
    # Must run AFTER patching so that LpWindowAttention parameters (qkv, proj,
    # rel-pos-bias table) and LpQKNorm.alpha_raw are present on the module
    # tree that ``initialize_model`` walks.
    initialize_model(
        model,
        init_scheme=init_scheme,
        linear_init_std=linear_init_std,
        alpha_init_scheme=alpha_init_scheme,
        alpha_init_fixed=alpha_init_fixed,
    )

    return model


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_attention_modules(
    model: nn.Module,
) -> list[tuple[nn.Module, str, WindowAttention]]:
    """Walk the module tree and return all ``WindowAttention`` instances.

    Parameters
    ----------
    model : nn.Module
        The top-level model to walk.

    Returns
    -------
    list of (parent_module, attr_name, attention_module)
        Each tuple identifies one ``WindowAttention`` together with its
        parent module and attribute name, enabling replacement via
        ``setattr(parent, attr_name, new_module)``.
    """
    results: list[tuple[nn.Module, str, WindowAttention]] = []

    for name, module in model.named_modules():
        if isinstance(module, WindowAttention):
            # Split "swinViT.layers1.0.blocks.0.attn" into parent path and attr.
            parts = name.rsplit(".", maxsplit=1)
            if len(parts) == 2:
                parent_path, attr_name = parts
                parent = model.get_submodule(parent_path)
            else:
                # Top-level attribute (unlikely for WindowAttention).
                parent = model
                attr_name = name
            results.append((parent, attr_name, module))

    return results


def _copy_weights(
    src: WindowAttention,
    dst: LpWindowAttention,
) -> None:
    """Copy shared weights from a stock ``WindowAttention`` to ``LpWindowAttention``.

    Parameters
    ----------
    src : WindowAttention
        Stock MONAI attention module (source of weights).
    dst : LpWindowAttention
        Lp-QKNorm replacement attention module (destination).

    Raises
    ------
    WeightTransferError
        If any weight shape does not match between ``src`` and ``dst``.
    """
    with torch.no_grad():
        # QKV projection
        _safe_copy(src.qkv.weight, dst.qkv.weight, "qkv.weight")
        if src.qkv.bias is not None and dst.qkv.bias is not None:
            _safe_copy(src.qkv.bias, dst.qkv.bias, "qkv.bias")

        # Output projection
        _safe_copy(src.proj.weight, dst.proj.weight, "proj.weight")
        if src.proj.bias is not None and dst.proj.bias is not None:
            _safe_copy(src.proj.bias, dst.proj.bias, "proj.bias")

        # Relative position bias table (nn.Parameter)
        _safe_copy(
            src.relative_position_bias_table.data,
            dst.relative_position_bias_table.data,
            "relative_position_bias_table",
        )

        # Relative position index (buffer — mypy sees registered buffers as
        # Tensor | Module; the runtime type is always Tensor).
        _safe_copy(
            src.relative_position_index,  # type: ignore[arg-type]
            dst.relative_position_index,  # type: ignore[arg-type]
            "relative_position_index",
        )


def _safe_copy(
    src_tensor: torch.Tensor,
    dst_tensor: torch.Tensor,
    name: str,
) -> None:
    """Copy ``src_tensor`` into ``dst_tensor`` with shape validation.

    Parameters
    ----------
    src_tensor : Tensor
        Source tensor.
    dst_tensor : Tensor
        Destination tensor (modified in-place).
    name : str
        Human-readable name for error messages.

    Raises
    ------
    WeightTransferError
        If ``src_tensor.shape != dst_tensor.shape``.
    """
    if src_tensor.shape != dst_tensor.shape:
        raise WeightTransferError(
            f"Shape mismatch for '{name}': src={src_tensor.shape}, dst={dst_tensor.shape}.",
            details={
                "name": name,
                "src_shape": src_tensor.shape,
                "dst_shape": dst_tensor.shape,
            },
        )
    dst_tensor.copy_(src_tensor)
