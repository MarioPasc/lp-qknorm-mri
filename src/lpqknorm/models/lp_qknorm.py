"""Lp query-key normalization module.

Implements the generalized Lp-QKNorm of Lopez-Rubio et al. (2026,
arXiv:2602.05006), which reduces to the original QKNorm of Henry et al.
(2020, arXiv:2010.04245) when ``p = 2``.

Mathematical specification
--------------------------
Given query vector ``q_i`` in R^{d_k} and key vector ``k_j`` in R^{d_k}::

    q_hat_i^(p) = q_i / (||q_i||_p + eps)
    k_hat_j^(p) = k_j / (||k_j||_p + eps)
    s_ij^(p)    = alpha * <q_hat_i^(p), k_hat_j^(p)>

where ``alpha = softplus(alpha_raw)`` is a learnable positive scalar and
``eps = 1e-6`` is a numerical safety constant.

Numerically stable norm computation
-------------------------------------
Two regimes avoid NaN/inf gradients:

- **p >= 2** (main sweep range):
    ``||v||_p = (sum_h |v_h|^p)^(1/p) + eps``
    Epsilon is added *after* the power sum (outside the root). This matches
    Henry et al.'s L2 form exactly at p=2:
        ``||v||_2 = (sum_h v_h^2)^(1/2) + eps``  [eps post-root]
    which is identical to ``v.norm(p=2, dim=-1, keepdim=True) + eps``.

- **p < 2**:
    ``||v||_p = (sum_h (|v_h| + eps)^p)^(1/p)``
    Epsilon is added *inside* the absolute value, before exponentiation.
    This prevents the gradient ``d/dv_h (|v_h|^p) = p |v_h|^{p-1} sign(v_h)``
    from diverging to infinity when ``v_h -> 0`` and ``p < 1`` (or becoming
    poorly conditioned for ``p in (1, 2)``).

p=2 equivalence (Critical invariant)
--------------------------------------
For ``p = 2`` the p >= 2 branch computes::

    norm = (sum_h |v_h|^2)^(1/2) + eps  ==  ||v||_2 + eps

which is the exact Henry et al. reference formula. We deliberately add eps
*after* taking the root so the two are identical. Verified numerically to
< 1e-7 absolute error over 100 random inputs.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from lpqknorm.utils.exceptions import ModelConfigError


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LpQKNormConfig:
    """Frozen configuration for the Lp-QKNorm normalization module.

    Parameters
    ----------
    p : float
        The Lp norm exponent. Must satisfy ``p >= 1``. The main sweep uses
        ``p in {2.0, 2.5, 3.0, 3.5, 4.0}``. Values ``p < 2`` are supported
        for ablations and unit tests.
    learnable_alpha : bool
        If ``True`` (default), ``alpha_raw`` is registered as a learnable
        ``nn.Parameter`` and ``alpha = softplus(alpha_raw)`` is trained via
        back-propagation. If ``False``, it is registered as a non-learnable
        buffer fixed at ``init_alpha``.
    init_alpha : float
        The desired initial value of ``alpha`` (i.e., the output of softplus).
        Must be strictly positive. The module stores the inverse-softplus
        of this value as ``alpha_raw`` so that at initialisation
        ``softplus(alpha_raw) == init_alpha``.
    eps : float
        Numerical safety constant added to the norm. Must be strictly
        positive. Default ``1e-6`` matches the project specification.

    Raises
    ------
    ModelConfigError
        If any of the constraints (``p >= 1``, ``eps > 0``,
        ``init_alpha > 0``) are violated.
    """

    p: float
    learnable_alpha: bool = True
    init_alpha: float = 1.0
    eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.p < 1.0:
            raise ModelConfigError(
                f"p must be >= 1, got {self.p}.",
                details={"p": self.p},
            )
        if self.eps <= 0.0:
            raise ModelConfigError(
                f"eps must be strictly positive, got {self.eps}.",
                details={"eps": self.eps},
            )
        if self.init_alpha <= 0.0:
            raise ModelConfigError(
                f"init_alpha must be strictly positive, got {self.init_alpha}.",
                details={"init_alpha": self.init_alpha},
            )


# ---------------------------------------------------------------------------
# Private helper
# ---------------------------------------------------------------------------


def _lp_normalize(
    x: Tensor,
    p: float,
    eps: float,
    dim: int = -1,
) -> Tensor:
    """Normalize ``x`` to unit Lp norm along ``dim``.

    Implements numerically stable Lp normalization in two regimes:

    - **p >= 2**: ``norm = (sum_h |v_h|^p)^{1/p} + eps``  (eps post-root)
      Matches Henry et al. (2020) L2-QKNorm exactly when ``p = 2``.

    - **p < 2**: ``norm = (sum_h (|v_h| + eps)^p)^{1/p}`` (eps pre-absolute-value)
      Prevents gradient blow-up at zero-crossings for fractional exponents.

    The norm is always evaluated in ``float32`` regardless of the caller's
    autocast context.  For non-integer ``p`` (e.g. ``p=2.5``) the operations
    ``|v|^p`` and ``(⋅)^{1/p}`` are numerically unstable under ``bfloat16``
    (7-bit mantissa) and readily produce NaN/Inf that poisons the attention
    logits.  Running the norm in fp32 adds negligible cost (~1 % of the
    downstream QK matmul) and eliminates the mixed-precision NaN regime.

    Parameters
    ----------
    x : Tensor
        Input tensor of arbitrary shape. Normalization is applied along
        ``dim``.
    p : float
        Lp norm exponent. Must satisfy ``p >= 1``.
    eps : float
        Numerical safety constant. Must be strictly positive.
    dim : int
        Dimension along which the norm is computed. Default ``-1`` (last
        dimension, i.e., the head/feature dimension for QK tensors).

    Returns
    -------
    Tensor
        Tensor of the same shape and dtype as ``x``, with Lp norm along
        ``dim`` approximately equal to 1 (within ``eps`` tolerance).

    Notes
    -----
    The returned tensor satisfies ``||output||_p ≈ 1`` (equality up to
    ``eps / ||x||_p`` correction). For the p=2 branch this equals the
    standard L2-normalized form
    ``x / (x.norm(p=2, dim=dim, keepdim=True) + eps)`` to within 1e-7.
    """
    orig_dtype = x.dtype
    device_type = x.device.type if x.device.type in {"cuda", "cpu"} else "cpu"

    # Disable autocast around the pow/sum/pow chain so the operation runs in
    # high precision even when the outer training loop is in bf16-mixed /
    # fp16-mixed.  We up-cast bf16/fp16 inputs to fp32, but leave fp32/fp64
    # inputs untouched to preserve gradcheck (double-precision) semantics.
    if orig_dtype in (torch.bfloat16, torch.float16):
        compute_dtype = torch.float32
    else:
        compute_dtype = orig_dtype

    with torch.amp.autocast(device_type=device_type, enabled=False):
        x_c = x.to(compute_dtype)
        if p >= 2.0:
            # Post-root eps: (sum |v_h|^p)^{1/p} + eps.
            # At p=2: norm = (sum v_h^2)^{1/2} + eps == ||v||_2 + eps,
            # which is the exact Henry et al. reference formula.
            norm = x_c.abs().pow(p).sum(dim=dim, keepdim=True).pow(1.0 / p) + eps
        else:
            # Pre-absolute-value eps: (sum (|v_h| + eps)^p)^{1/p}.
            # Gradient: d/dv_h[(|v_h| + eps)^p] = p*(|v_h| + eps)^{p-1}*sign(v_h),
            # which is bounded at v_h = 0 for all p >= 1.
            norm = (x_c.abs() + eps).pow(p).sum(dim=dim, keepdim=True).pow(1.0 / p)
        normed = x_c / norm

    # Cast back to the caller's dtype so the subsequent q_hat @ k_hat.T runs
    # in the expected (bf16/fp16) precision for throughput.
    return normed.to(orig_dtype)


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------


class LpQKNorm(nn.Module):
    """Lp query-key normalization with learnable scaling.

    Normalizes query and key tensors by their Lp norm along the last
    dimension and scales the resulting attention logits by a learnable
    positive scalar ``alpha``.

    The forward computation is::

        q_hat = q / (||q||_p + eps)
        k_hat = k / (||k||_p + eps)
        alpha = softplus(alpha_raw)

    The caller computes the scaled dot-product as
    ``alpha * (q_hat @ k_hat.T)``.

    At ``p = 2`` this is numerically identical to Henry et al. (2020)
    QKNorm: ``q / (||q||_2 + eps)``.

    Parameters
    ----------
    cfg : LpQKNormConfig
        Frozen configuration object.

    Attributes
    ----------
    p : torch.Tensor
        Non-learnable buffer holding the norm exponent. Registered via
        ``register_buffer`` so it moves with the module across devices but
        is excluded from ``parameters()``.
    alpha_raw : torch.Tensor
        If ``cfg.learnable_alpha`` is ``True``, this is a learnable
        ``nn.Parameter`` initialized to ``softplus_inverse(cfg.init_alpha)``.
        If ``False``, it is a non-learnable buffer fixed at the same value.

    Notes
    -----
    **Why register ``p`` as a buffer?** The project specification states
    that ``p`` must not be learned (it is a controlled experimental variable
    swept externally). Registering as a buffer ensures it is saved in
    ``state_dict`` for reproducibility and moves to the correct device with
    ``.to(device)``.

    **Inverse softplus initialization.** We initialize ``alpha_raw`` such
    that ``softplus(alpha_raw) == cfg.init_alpha`` exactly. The inverse is:
        ``alpha_raw = log(exp(init_alpha) - 1)``
    For large ``init_alpha`` this simplifies to ``alpha_raw ~= init_alpha``.

    References
    ----------
    - Henry et al. *Query-Key Normalization for Transformers*. EMNLP 2020.
      arXiv:2010.04245.
    - Lopez-Rubio et al. *Enhanced QKNorm with the Lp Norm*. 2026.
      arXiv:2602.05006.
    """

    def __init__(self, cfg: LpQKNormConfig) -> None:
        super().__init__()

        self.cfg = cfg
        self.eps: float = cfg.eps

        # Declare buffer types explicitly so mypy can resolve their types.
        # register_buffer stores tensors as attributes but doesn't annotate them;
        # the explicit declaration here supplements the dynamic registration.
        self.p: Tensor
        self.alpha_raw: Tensor

        # Register p as a non-learnable buffer so it is saved in state_dict
        # and moves with .to(device), but is excluded from parameters().
        self.register_buffer(
            "p",
            torch.tensor(cfg.p, dtype=torch.float32),
            persistent=True,
        )

        # Compute inverse softplus: alpha_raw s.t. softplus(alpha_raw) = init_alpha.
        # log(exp(x) - 1) is the exact closed-form inverse of softplus(x).
        # math.log(math.expm1(x)) is numerically equivalent and avoids
        # catastrophic cancellation for small x.
        alpha_raw_init = math.log(math.expm1(cfg.init_alpha))

        if cfg.learnable_alpha:
            # nn.Parameter is a Tensor subclass; compatible with the class-level annotation.
            self.alpha_raw = nn.Parameter(
                torch.tensor(alpha_raw_init, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "alpha_raw",
                torch.tensor(alpha_raw_init, dtype=torch.float32),
                persistent=True,
            )

        logger.debug(
            "LpQKNorm initialised: p=%.2f, learnable_alpha=%s, init_alpha=%.4f, eps=%g",
            cfg.p,
            cfg.learnable_alpha,
            cfg.init_alpha,
            cfg.eps,
        )

    def forward(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Normalize queries and keys by their Lp norm and return the scale.

        Parameters
        ----------
        q : Tensor
            Query tensor of shape ``(B, n, d_k)`` or any shape where the
            last dimension is the head/feature dimension.
        k : Tensor
            Key tensor of the same shape as ``q``.

        Returns
        -------
        q_hat : Tensor
            Lp-normalized query tensor, same shape as ``q``.
            Satisfies ``||q_hat[b, i, :]||_p ~= 1`` for all ``b, i``.
        k_hat : Tensor
            Lp-normalized key tensor, same shape as ``k``.
            Satisfies ``||k_hat[b, j, :]||_p ~= 1`` for all ``b, j``.
        alpha : Tensor
            Positive learnable scale scalar: ``alpha = softplus(alpha_raw)``.
            Shape ``()``, dtype matching ``q``.

        Notes
        -----
        The caller computes attention logits as:
            ``logits = alpha * (q_hat @ k_hat.transpose(-2, -1))``
        The relative position bias is added by the caller after this step,
        preserving the Swin-UNETR architecture invariant.
        """
        p_val: float = float(self.p)  # buffer guaranteed to be a scalar Tensor

        q_hat = _lp_normalize(q, p=p_val, eps=self.eps, dim=-1)
        k_hat = _lp_normalize(k, p=p_val, eps=self.eps, dim=-1)

        alpha = F.softplus(self.alpha_raw)

        return q_hat, k_hat, alpha
