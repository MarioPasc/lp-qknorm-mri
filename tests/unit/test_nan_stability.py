"""Tests guarding against the bf16 + non-integer-p NaN regression.

The original OOM crash at step 157 (A100 40 GB, B=8, bf16-mixed, p=2.5) was
preceded by ``train_loss_step: nan`` on earlier steps.  Root cause: the Lp
norm ``(sum |v|^p)^{1/p}`` evaluated under autocast produced bf16
intermediates that underflow / overflow for fractional ``p`` — especially
on the backward pass where ``pow`` gradients scale as
``p * |v|^{p-1}``.  The fix is a fp32 island inside ``_lp_normalize``.

These tests ensure the island is present and that the loss is finite under
the same bf16 regime used on Picasso (`bf16-mixed`, non-integer p).  They
run on CPU with ``torch.autocast`` spoofing the GPU bf16 path so they can
be executed in CI without a GPU.
"""

from __future__ import annotations

import pytest
import torch

from lpqknorm.models.attention import LpWindowAttention
from lpqknorm.models.lp_qknorm import LpQKNorm, LpQKNormConfig, _lp_normalize
from lpqknorm.training.losses import CompoundSegLoss


# Bf16 operations on CPU are supported since PyTorch 1.12 for most ops we
# need (matmul, sum, pow, softmax).  Skip individual ops that are CPU-bf16
# unsupported would simply raise — the tests here only stress the pieces we
# actually hit on GPU.


@pytest.mark.parametrize("p", [2.0, 2.5, 3.0, 3.5, 4.0])
def test_lp_normalize_fp32_island_preserves_finiteness_bf16_input(p: float) -> None:
    """Norms with fractional p on bf16 inputs must not NaN/Inf.

    The fp32 island inside ``_lp_normalize`` should lift the pow/sum/pow
    chain into fp32 even when the caller passes bf16 tensors.  The
    normalised vector is cast back to bf16.
    """
    torch.manual_seed(0)
    # Distribution deliberately mixes large and small magnitudes to hit
    # both ends of the bf16 dynamic range.
    x = torch.randn(4, 8, dtype=torch.bfloat16) * 3.0
    x[0, 0] = 1.0e-4  # tiny element — stresses (|v|+eps)^p in p<2 branch
    x[0, 1] = 1.0e2  # large element — stresses the post-root eps branch

    out = _lp_normalize(x, p=p, eps=1e-6, dim=-1)
    assert out.dtype == torch.bfloat16, f"Expected bfloat16 output, got {out.dtype}"
    assert torch.isfinite(out).all(), f"NaN/Inf in _lp_normalize output for p={p}"

    # Sanity: each row should have unit (bf16-approximate) p-norm.
    norms = out.float().abs().pow(p).sum(dim=-1).pow(1.0 / p)
    torch.testing.assert_close(
        norms,
        torch.ones_like(norms),
        atol=2e-2,  # bf16 round-off allowance
        rtol=2e-2,
    )


@pytest.mark.parametrize("p", [2.5, 3.0, 3.5, 4.0])
def test_lp_normalize_under_cpu_autocast_bf16_stays_finite(p: float) -> None:
    """Emulates the ``bf16-mixed`` CUDA regime on CPU via ``autocast``."""
    torch.manual_seed(1)
    x = torch.randn(2, 16, 8) * 5.0  # larger magnitude -> more stressful
    with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = _lp_normalize(x, p=p, eps=1e-6, dim=-1)
    assert torch.isfinite(out).all(), f"NaN/Inf inside autocast for p={p}"


@pytest.mark.parametrize("p", [2.5, 3.0, 3.5])
def test_lp_qknorm_forward_backward_bf16_no_nan_grads(p: float) -> None:
    """End-to-end LpQKNorm forward + backward under bf16 produces finite grads."""
    torch.manual_seed(2)
    module = LpQKNorm(LpQKNormConfig(p=p))
    q = (torch.randn(3, 4, 8, dtype=torch.bfloat16) * 2.0).detach().requires_grad_(True)
    k = (torch.randn(3, 4, 8, dtype=torch.bfloat16) * 2.0).detach().requires_grad_(True)

    with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
        q_hat, k_hat, alpha = module(q, k)
        loss = (alpha * (q_hat * k_hat).sum()).float()

    assert torch.isfinite(loss), f"NaN/Inf forward loss for p={p}"

    loss.backward()
    assert q.grad is not None
    assert k.grad is not None
    assert torch.isfinite(q.grad).all(), f"NaN grad on q for p={p}"
    assert torch.isfinite(k.grad).all(), f"NaN grad on k for p={p}"


@pytest.mark.parametrize("p", [2.0, 2.5, 3.0, 3.5, 4.0])
def test_lp_window_attention_bf16_no_nan(p: float) -> None:
    """A full LpWindowAttention forward/backward under bf16 must stay finite."""
    torch.manual_seed(3)
    module = LpWindowAttention(
        dim=24,
        num_heads=3,
        window_size=(7, 7),
        qkv_bias=True,
        lp_cfg=LpQKNormConfig(p=p),
    )
    x = (
        (torch.randn(2, 49, 24, dtype=torch.bfloat16) * 2.0)
        .detach()
        .requires_grad_(True)
    )

    with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
        y = module(x, mask=None)
        loss = y.float().square().mean()

    assert torch.isfinite(loss), f"NaN forward loss for p={p}"
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all(), f"NaN grad for p={p}"


def test_compound_loss_fp32_island_bf16_inputs_finite() -> None:
    """CompoundSegLoss must evaluate BCE+Dice in fp32 even when the upstream
    logits are bf16 — otherwise soft Dice's 1e-5 smoothing underflows."""
    torch.manual_seed(4)
    loss_fn = CompoundSegLoss(bce_weight=0.5, dice_weight=0.5)

    # Logits with extreme positive / negative magnitudes to stress BCE's
    # ``log(1+exp(-|x|))`` stabilisation under bf16.
    logits = torch.empty(2, 1, 8, 8, dtype=torch.bfloat16).uniform_(-30.0, 30.0)
    target = (torch.rand(2, 1, 8, 8) > 0.7).float()

    with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
        total, bce, dice = loss_fn(logits, target)

    for name, t in [("total", total), ("bce", bce), ("dice", dice)]:
        assert torch.isfinite(t), f"Non-finite {name} loss"
