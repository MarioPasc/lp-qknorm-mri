"""Weight initialization for Lp-QKNorm Swin-UNETR (from-scratch regime).

Implements the Phase 2 weight-initialization specification for the primary
"trained from scratch" run.  The rationale is methodological: pretrained
weights would have been shaped under vanilla scaled-dot-product attention
(implicitly the ``p = 2`` regime) and would therefore bias the learned
Q/K geometry toward ``l_2``-amenable configurations, confounding the
``p`` effect.

Scheme: ``scratch_trunc_normal``
---------------------------------
Applied inside :func:`lpqknorm.models.swin_unetr_lp.build_swin_unetr_lp`
**after** MONAI ``SwinUNETR`` construction and the ``LpWindowAttention``
patch, via a single ``model.apply(_init_weights)`` pass followed by a
dedicated walk that initializes ``LpQKNorm.alpha_raw`` from
``num_heads``/``dim`` of each containing ``LpWindowAttention``.

=============================  =========================================  ==================
Module type                    Weight                                     Bias
=============================  =========================================  ==================
``nn.Linear``                  ``trunc_normal_(std=linear_init_std)``     ``zeros_``
``nn.Conv{2,3}d``              ``trunc_normal_(std=linear_init_std)``     ``zeros_``
``nn.LayerNorm``               ``ones_``                                  ``zeros_``
``relative_position_bias_table`` ``trunc_normal_(std=linear_init_std)``   --
``LpQKNorm.alpha_raw``         per ``alpha_init_scheme`` (see below)      --
=============================  =========================================  ==================

alpha initialization
--------------------
Let ``d_k = dim // num_heads`` per LpWindowAttention (stage-dependent in
Swin-UNETR, but ``feature_size=24`` yields ``d_k = 8`` at every stage).
Let ``alpha_star`` denote the target effective scale:

============= ==================== =========================================
scheme        ``alpha_star``        reference
============= ==================== =========================================
``log_dk``    ``log(d_k)``          Henry et al. (2020), EMNLP Findings.
``sqrt_dk``   ``sqrt(d_k)``         scaled-dot-product magnitude match.
``fixed``     ``alpha_init_fixed``  ablation only; ``None`` raises.
============= ==================== =========================================

``alpha_raw`` is then set to ``softplus_inverse(alpha_star)`` so that
``softplus(alpha_raw) == alpha_star`` exactly at initialization.

References
----------
- Liu, Z. et al. *Swin Transformer*. ICCV 2021. arXiv:2103.14030.
- Hatamizadeh, A. et al. *Swin UNETR*. BrainLes 2021. arXiv:2201.01266.
- Henry, A. et al. *Query-Key Normalization for Transformers*. EMNLP 2020.
  arXiv:2010.04245.
- Dosovitskiy, A. et al. *An Image is Worth 16x16 Words*. ICLR 2021.
  arXiv:2010.11929.
- Tang, Y. et al. *Self-Supervised Pre-Training of Swin Transformers for 3D
  Medical Image Analysis*. CVPR 2022. arXiv:2111.14791.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Literal

import torch
import torch.nn as nn

from lpqknorm.utils.exceptions import LpInitError


__all__ = [
    "AlphaInitScheme",
    "InitScheme",
    "initialize_model",
    "softplus_inverse",
]

logger = logging.getLogger(__name__)


InitScheme = Literal["scratch_trunc_normal", "pretrained_ssl"]
AlphaInitScheme = Literal["log_dk", "sqrt_dk", "fixed"]


# ---------------------------------------------------------------------------
# trunc_normal_: prefer MONAI, fall back to PyTorch.  Signatures are compatible.
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Numerically stable softplus inverse
# ---------------------------------------------------------------------------


def softplus_inverse(x: float) -> float:
    """Numerically stable inverse of softplus for strictly-positive ``x``.

    Solves ``softplus(y) = x`` for ``y``, i.e. ``y = log(exp(x) - 1)``.  For
    ``x > 20`` uses the stable branch ``y = x + log1p(-exp(-x))`` to avoid
    overflow in ``exp(x)``.

    Parameters
    ----------
    x : float
        Target value, must be ``> 0``.

    Returns
    -------
    float
        The pre-softplus value ``y`` such that ``softplus(y) = x``.

    Raises
    ------
    LpInitError
        If ``x <= 0``.
    """
    if x <= 0.0:
        raise LpInitError(
            f"alpha target must be > 0, got {x}",
            details={"x": x},
        )
    if x > 20.0:
        return x + math.log1p(-math.exp(-x))
    return math.log(math.expm1(x))


# ---------------------------------------------------------------------------
# alpha_star computation
# ---------------------------------------------------------------------------


def _compute_alpha_star(
    d_k: int,
    alpha_init_scheme: AlphaInitScheme,
    alpha_init_fixed: float | None,
) -> float:
    """Compute the target ``alpha_star`` for a single LpQKNorm module.

    Parameters
    ----------
    d_k : int
        Head dimension ``dim // num_heads`` for the containing
        ``LpWindowAttention`` module.
    alpha_init_scheme : {"log_dk", "sqrt_dk", "fixed"}
        Scheme selector.
    alpha_init_fixed : float or None
        Required when ``scheme == "fixed"``; must be ``> 0``.

    Returns
    -------
    float
        The target ``alpha_star`` value.

    Raises
    ------
    LpInitError
        If the scheme is unknown, if ``d_k < 1``, or if
        ``alpha_init_fixed`` is required but missing/non-positive.
    """
    if d_k < 1:
        raise LpInitError(
            f"d_k must be >= 1, got {d_k}",
            details={"d_k": d_k},
        )
    if alpha_init_scheme == "log_dk":
        # d_k == 1 would give log(1) = 0 which is not a valid softplus target.
        if d_k == 1:
            raise LpInitError(
                "alpha_init_scheme='log_dk' requires d_k > 1; got d_k=1.",
                details={"d_k": d_k},
            )
        return math.log(d_k)
    if alpha_init_scheme == "sqrt_dk":
        return math.sqrt(d_k)
    if alpha_init_scheme == "fixed":
        if alpha_init_fixed is None:
            raise LpInitError(
                "alpha_init_scheme='fixed' requires alpha_init_fixed != None.",
                details={"alpha_init_fixed": None},
            )
        if alpha_init_fixed <= 0.0:
            raise LpInitError(
                f"alpha_init_fixed must be > 0, got {alpha_init_fixed}.",
                details={"alpha_init_fixed": alpha_init_fixed},
            )
        return float(alpha_init_fixed)
    raise LpInitError(
        f"unknown alpha_init_scheme: {alpha_init_scheme!r}",
        details={"alpha_init_scheme": alpha_init_scheme},
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def initialize_model(
    model: nn.Module,
    *,
    init_scheme: InitScheme = "scratch_trunc_normal",
    linear_init_std: float = 0.02,
    alpha_init_scheme: AlphaInitScheme = "log_dk",
    alpha_init_fixed: float | None = None,
) -> nn.Module:
    """Initialize model weights per the Phase-2 from-scratch specification.

    Applies ``_init_weights`` via :meth:`torch.nn.Module.apply` to seed all
    ``nn.Linear``, ``nn.Conv{2,3}d``, ``nn.LayerNorm`` parameters, as well as
    any ``relative_position_bias_table`` parameter found on windowed-attention
    modules.  Then walks every ``LpWindowAttention`` / ``LpQKNorm`` pair to
    initialize ``alpha_raw`` according to ``alpha_init_scheme``.

    Parameters
    ----------
    model : nn.Module
        Model to initialize.  Modified in-place.
    init_scheme : {"scratch_trunc_normal", "pretrained_ssl"}
        ``"scratch_trunc_normal"`` is the primary regime.  ``"pretrained_ssl"``
        is reserved for the single ablation row (handled by a different
        code path that loads a Tang et al. SSL checkpoint); calling this
        function with ``"pretrained_ssl"`` only seeds ``alpha_raw``,
        leaving the rest of the model untouched.
    linear_init_std : float
        Standard deviation for ``trunc_normal_`` of linear / conv /
        relative-position-bias weights.  Default ``0.02``, matching Swin
        (Liu et al., 2021) and ViT (Dosovitskiy et al., 2021).
    alpha_init_scheme : {"log_dk", "sqrt_dk", "fixed"}
        Scheme for initializing ``LpQKNorm.alpha_raw``.  Default ``"log_dk"``.
    alpha_init_fixed : float or None
        Required iff ``alpha_init_scheme == "fixed"``.

    Returns
    -------
    nn.Module
        The same ``model`` object, with parameters initialized in-place.

    Raises
    ------
    LpInitError
        If the init scheme is unknown or ``alpha_init_fixed`` is missing
        when required.
    """
    if init_scheme not in ("scratch_trunc_normal", "pretrained_ssl"):
        raise LpInitError(
            f"unknown init_scheme: {init_scheme!r}",
            details={"init_scheme": init_scheme},
        )

    if linear_init_std <= 0.0:
        raise LpInitError(
            f"linear_init_std must be > 0, got {linear_init_std}",
            details={"linear_init_std": linear_init_std},
        )

    if init_scheme == "scratch_trunc_normal":
        model.apply(lambda m: _init_weights(m, std=linear_init_std))

    _init_alpha_raw_all(
        model,
        alpha_init_scheme=alpha_init_scheme,
        alpha_init_fixed=alpha_init_fixed,
    )

    logger.info(
        "initialize_model: scheme=%s, linear_init_std=%.4f, alpha_scheme=%s, "
        "alpha_fixed=%s.",
        init_scheme,
        linear_init_std,
        alpha_init_scheme,
        alpha_init_fixed,
    )
    return model


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _init_weights(module: nn.Module, *, std: float) -> None:
    """Single-module initializer suitable for :meth:`nn.Module.apply`.

    Parameters
    ----------
    module : nn.Module
        The module to initialize.  Only ``nn.Linear``, ``nn.Conv2d``,
        ``nn.Conv3d``, and ``nn.LayerNorm`` are touched; all other
        module types are a no-op.  Modules holding a
        ``relative_position_bias_table`` ``nn.Parameter`` also have
        that tensor re-initialized (for byte-level determinism across
        ``p`` values that would otherwise consume RNG differently).
    std : float
        Standard deviation for truncated-normal init of linear/conv
        weights and the relative-position-bias table.
    """
    if isinstance(module, nn.Linear):
        _trunc_normal_(module.weight, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        return

    if isinstance(module, (nn.Conv2d, nn.Conv3d)):
        _trunc_normal_(module.weight, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        return

    if isinstance(module, nn.LayerNorm):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        return

    # Relative-position-bias table: re-initialize for determinism across p.
    # The attribute is an nn.Parameter on the containing attention module,
    # not a submodule, so isinstance-based dispatch misses it.
    rpbt = getattr(module, "relative_position_bias_table", None)
    if isinstance(rpbt, nn.Parameter):
        _trunc_normal_(rpbt, std=std)


def _init_alpha_raw_all(
    model: nn.Module,
    *,
    alpha_init_scheme: AlphaInitScheme,
    alpha_init_fixed: float | None,
) -> None:
    """Walk ``model`` and seed ``LpQKNorm.alpha_raw`` on every LpWindowAttention.

    ``d_k = dim // num_heads`` is computed per-containing-``LpWindowAttention``
    so that stage-dependent head-dim ratios are honored.  In the default
    Swin-UNETR layout (``feature_size=24``, heads=(3, 6, 12, 24)),
    ``d_k = 8`` at every stage, so the target ``alpha_star`` is identical
    across stages.

    Parameters
    ----------
    model : nn.Module
        Top-level model (typically a patched ``SwinUNETR``).
    alpha_init_scheme : {"log_dk", "sqrt_dk", "fixed"}
    alpha_init_fixed : float or None
    """
    # Late import to avoid a circular dependency at module load time.
    from lpqknorm.models.attention import LpWindowAttention

    seen = 0
    for module in model.modules():
        if not isinstance(module, LpWindowAttention):
            continue
        d_k = int(module.dim // module.num_heads)
        alpha_star = _compute_alpha_star(
            d_k=d_k,
            alpha_init_scheme=alpha_init_scheme,
            alpha_init_fixed=alpha_init_fixed,
        )
        alpha_raw_init = softplus_inverse(alpha_star)
        with torch.no_grad():
            # ``alpha_raw`` is either an nn.Parameter (learnable) or a buffer
            # (frozen).  ``.data.fill_`` works for both without touching autograd.
            module.lp_qknorm.alpha_raw.data.fill_(alpha_raw_init)
        seen += 1

    if seen == 0:
        logger.debug(
            "_init_alpha_raw_all: model contains no LpWindowAttention (vanilla "
            "baseline or pretrained-backbone path); skipping alpha init."
        )
    else:
        logger.debug(
            "_init_alpha_raw_all: seeded alpha_raw on %d LpWindowAttention "
            "module(s) via scheme=%s.",
            seen,
            alpha_init_scheme,
        )
