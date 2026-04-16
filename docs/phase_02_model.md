# Phase 2 — Model: Swin-UNETR with Lp-QKNorm

## Goal

Produce a drop-in generalization of the QKNorm attention mechanism
(*Henry et al., 2020, arXiv:2010.04245*) based on the Lp norm
(*López-Rubio et al., 2026, arXiv:2602.05006*), and splice it into the
windowed self-attention of MONAI's 2D `SwinUNETR` in a way that is
**numerically identical to the original QKNorm when `p = 2`** and
architecturally unchanged everywhere else.

This phase contains no training. It produces a module that is exercised by
unit tests in isolation and in integration with a frozen-weights forward
pass.

## Mathematical specification

Let `X ∈ ℝ^{B × n × d}` be the input to a window of size `W` after the
standard Swin partitioning, so `n = W²`. For each head `h` with per-head
dimension `d_k`:

```
Q = X W_Q,   K = X W_K,   V = X W_V,        Q, K, V ∈ ℝ^{B × n × d_k}

||v||_p = (Σ_{h=1}^{d_k} |v_h|^p)^(1/p),     p ≥ 1

q̂_i^(p) = q_i / (||q_i||_p + ε)
k̂_j^(p) = k_j / (||k_j||_p + ε)

s_ij^(p) = α · ⟨q̂_i^(p), k̂_j^(p)⟩

A = softmax(S^(p) + B_rel),     output = A V
```

where `α` is a learnable positive scalar (parameterised as
`α = softplus(α_raw)` to enforce positivity without a hard constraint),
`ε = 1e-6` is an intrinsic-safety epsilon, and `B_rel` is Swin's relative
position bias (unchanged — this is attached after the dot product, not
absorbed into Q/K).

**Gradient note.** The `Lp` norm is non-differentiable at `v = 0` for
`p < 2`. Training regularly produces small-norm vectors, so we add `ε` inside
the norm computation and use the numerically stable form:

```
||v||_p = (Σ_h (|v_h| + ε)^p)^(1/p)   for p < 2
||v||_p = (Σ_h |v_h|^p + ε)^(1/p)     for p ≥ 2
```

The sweep range `p ∈ [2, 4]` sits in the latter regime, so the simpler form
applies, but the module must handle `p < 2` for unit tests and ablations.

## Open questions the agent must resolve before coding

1. **Which class in MONAI holds the windowed attention for 2D SwinUNETR**.
   Inspect `monai.networks.nets.SwinUNETR(spatial_dims=2, ...)` at runtime,
   walk the module tree, and identify the attention module. As of MONAI
   ≥ 1.3.0 this should be `monai.networks.blocks.patchembedding.WindowAttention`
   or equivalent in `monai.networks.blocks.selfattention`. **Confirm**, do
   not assume.
2. **The exact signature of that attention's forward pass**, specifically
   how it handles `qkv` projection, scaling (typically `head_dim ** -0.5`),
   softmax, and relative position bias. Record the canonical
   input/output shapes in a docstring comment at the top of
   `models/attention.py`.
3. **Whether relative position biases are registered per-layer or shared**.
   This determines whether replacing the module per-layer preserves learned
   biases across layers.
4. **The simplest patching strategy**: (a) subclass + monkey-patch module
   tree after construction, or (b) copy the MONAI source once, paste into
   this repo under `models/_monai_vendored.py` with provenance comment, and
   modify the attention class directly. The agent should choose based on
   MONAI's API stability; document the decision in `models/swin_unetr_lp.py`.
   Prefer (a) if feasible; vendor only if MONAI internals resist clean
   substitution.

## I/O contract

### Public API (`src/lpqknorm/models/`)

```python
# lp_qknorm.py
@dataclass(frozen=True)
class LpQKNormConfig:
    p: float                  # p ≥ 1
    learnable_alpha: bool = True
    init_alpha: float = 1.0   # initial value of softplus(alpha_raw)^-1
    eps: float = 1e-6

class LpQKNorm(nn.Module):
    """
    Normalises Q and K by the Lp norm along the last dim and scales by
    a positive learnable alpha. Returns (q_hat, k_hat, alpha).
    """
    def __init__(self, cfg: LpQKNormConfig) -> None: ...
    def forward(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor, Tensor]: ...

# attention.py
class LpWindowAttention(nn.Module):
    """
    Drop-in replacement for MONAI's WindowAttention where the QK^T scaling
    is replaced by alpha * normalize_p(Q) @ normalize_p(K)^T. Relative
    position bias and attention dropout are preserved.
    """

# swin_unetr_lp.py
def build_swin_unetr_lp(
    img_size: tuple[int, int],
    in_channels: int,
    out_channels: int,
    feature_size: int,
    lp_cfg: LpQKNormConfig | None = None,
    patch_base: Literal["monai"] = "monai",
) -> nn.Module:
    """Build a 2D SwinUNETR with Lp-QKNorm attention.

    If ``lp_cfg`` is None, returns the stock MONAI SwinUNETR with
    unmodified WindowAttention (vanilla softmax baseline). This is the
    "no QKNorm" lower-bound control.
    """
    ...

# hooks.py
@dataclass
class AttentionCapture:
    q: Tensor | None
    k: Tensor | None
    q_hat: Tensor | None
    k_hat: Tensor | None
    logits: Tensor | None
    attention: Tensor | None
    alpha: Tensor | None
    stage_index: int
    block_index: int

class AttentionHookRegistry:
    def register(self, model: nn.Module, stages: Iterable[int]) -> None: ...
    def captures(self) -> list[AttentionCapture]: ...
    def clear(self) -> None: ...
    def remove(self) -> None: ...
```

Hooks must store references (not `.clone()`) during forward, then clone and
detach inside `captures()` to avoid keeping the autograd graph alive when
probing in `torch.inference_mode()`.

## Design notes

**Why `softplus(α_raw)` instead of `exp(α_raw)`.** Softplus is better
conditioned near zero and has bounded gradients for large negative inputs.
The original QKNorm paper is not explicit on parameterisation; `softplus` is
the safer default.

**Why preserve the relative position bias.** Removing it would conflate two
effects (Lp normalization and positional geometry). This is an ablation for
a future paper, not for this one.

**Why `eps` inside the norm**. Without it, lesion tokens that happen to
produce near-zero feature vectors at initialization would produce infinite
gradients on the first few steps, preventing training. The toy-model
prediction of an interior maximum at `p* ∈ (2, 4)` does not depend on `eps`
for `ε ≪ min_h |v_h|`.

**Why hooks rather than modifying the attention's return values.** Keeping
the forward signature identical to MONAI's original class means the rest of
SwinUNETR is unaware of the modification — important both for correctness
and for future ablations where the hook-based probe collection should work
on the stock MONAI model for baseline measurements.

## Implementation checklist

1. `models/lp_qknorm.py`:
   - `LpQKNormConfig` dataclass (frozen, typed).
   - `LpQKNorm` module with `p` as a buffer (not parameter) and `alpha_raw`
     as the only learnable parameter.
   - Private helper `_lp_normalize(x, p, eps, dim=-1)` with the
     numerically stable form described above. Unit-test this in isolation.
2. `models/attention.py`:
   - Replicate MONAI's `WindowAttention` interface (same `__init__` args,
     same `forward(x, mask)` signature, same output shape).
   - Replace the `attn = (q @ k.transpose(-2, -1)) * scale` line with the
     Lp-normalised analogue.
   - Keep all other code paths identical: qkv projection, head splitting,
     relative position bias addition, softmax, attention dropout, proj,
     proj dropout.
3. `models/swin_unetr_lp.py`:
   - Construct MONAI `SwinUNETR(spatial_dims=2, ...)`.
   - Walk the module tree, identify every `WindowAttention` instance,
     replace with `LpWindowAttention` preserving the learned weights
     (`W_qkv`, `proj`, relative position bias) from the freshly
     constructed MONAI module.
   - Return the patched model.
4. `models/hooks.py`:
   - `AttentionHookRegistry` with `register_forward_hook` on every
     `LpWindowAttention` at specified stages.
   - Capture `q`, `k`, normalised versions, final attention, and `alpha`.
5. Expose all of the above via `src/lpqknorm/models/__init__.py`.

## Acceptance tests

All in `tests/unit/test_lp_qknorm.py`,
`tests/unit/test_attention_equivalence.py`,
`tests/integration/test_forward_pass.py`.

### 1. `LpQKNorm` basic properties

```python
def test_lp_norm_unit_length():
    x = torch.randn(4, 16, 64)  # B, n, d_k
    for p in [1.5, 2.0, 2.5, 3.0, 4.0, 8.0]:
        cfg = LpQKNormConfig(p=p)
        module = LpQKNorm(cfg)
        q_hat, _, _ = module(x, x)
        norms = q_hat.abs().pow(p).sum(-1).pow(1.0 / p)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)
```

### 2. **Critical equivalence test**: `p = 2` recovers original QKNorm

Construct a reference `RefQKNorm` implementing Henry et al.'s formulation
exactly:

```python
q_hat_ref = q / (q.norm(p=2, dim=-1, keepdim=True) + eps)
k_hat_ref = k / (k.norm(p=2, dim=-1, keepdim=True) + eps)
logits_ref = alpha * (q_hat_ref @ k_hat_ref.transpose(-2, -1))
```

Then assert `torch.allclose(LpQKNorm(p=2)(q, k)[0], q_hat_ref, atol=1e-5)`
over 100 random inputs. This is the single most important test in the repo.
If it fails, every downstream claim about "`p = 2` is the QKNorm baseline"
is false.

### 3. Monotone norm ordering

For fixed `x`, `||x||_p` is non-increasing in `p`. Test that
`||LpQKNorm.lp_normalize(x, p=1).norm(p=1) >= ...norm(p=2) >= ...norm(p=4)`
up to numerical tolerance on 50 random vectors.

### 4. Gradient flow at the edge

Verify that `torch.autograd.gradcheck` passes for `_lp_normalize` with
`p ∈ {1.5, 2.0, 3.0}` on double-precision inputs of shape `(4, 8, 16)`
with `eps = 1e-6`. Skip `p = 1.0` (non-differentiable at zero crossings;
not in the sweep).

### 5. Drop-in compatibility

```python
def test_attention_shape_preserved():
    model_stock = monai.networks.nets.SwinUNETR(
        img_size=(224, 224), in_channels=1, out_channels=1,
        feature_size=24, spatial_dims=2,
    )
    model_lp = build_swin_unetr_lp(
        img_size=(224, 224), in_channels=1, out_channels=1,
        feature_size=24, lp_cfg=LpQKNormConfig(p=3.0),
    )
    x = torch.randn(2, 1, 224, 224)
    assert model_stock(x).shape == model_lp(x).shape
```

### 6. Weight transfer integrity

When patching, the `W_qkv`, `proj`, and relative position biases must equal
those of the freshly constructed stock model:

```python
def test_weights_transferred():
    stock = monai.networks.nets.SwinUNETR(...)
    patched = build_swin_unetr_lp(..., lp_cfg=LpQKNormConfig(p=2.0))
    for (_, stock_mod), (_, patched_mod) in zip(
        find_attentions(stock), find_attentions(patched)
    ):
        assert torch.equal(stock_mod.qkv.weight, patched_mod.qkv.weight)
        assert torch.equal(
            stock_mod.relative_position_bias_table,
            patched_mod.relative_position_bias_table,
        )
```

### 7. Hook capture test

```python
def test_hooks_capture_expected_tensors():
    model = build_swin_unetr_lp(..., lp_cfg=LpQKNormConfig(p=3.0))
    registry = AttentionHookRegistry()
    registry.register(model, stages=[0])
    with torch.inference_mode():
        _ = model(torch.randn(1, 1, 224, 224))
    captures = registry.captures()
    assert len(captures) >= 1  # at least one stage-0 block
    for c in captures:
        assert c.q.shape == c.k.shape
        assert c.q_hat.shape == c.q.shape
        assert c.attention.shape[-1] == c.attention.shape[-2]  # square
        norm = c.q_hat.abs().pow(3.0).sum(-1).pow(1/3.0)
        assert torch.allclose(norm, torch.ones_like(norm), atol=1e-3)
```

### 8. **Interior-maximum sanity check on synthetic input**

Build a controlled test input matching the toy model from the design
discussion: `s = 4, d_k = 64`, one query aligned with lesion, two keys
(lesion-like peaky, background-like diffuse). Compute the logit gap
`Δ(p) = s_{q,k_L} - s_{q,k_B}` for `p ∈ {1.5, 2, 2.5, 3, 4, 8}` and assert
the empirical maximum lies at `p* ∈ {3, 4}`. This connects the implementation
to the theoretical prediction and catches sign errors early.

## References

- Henry, Dachapally, Pawar, Chen. *Query-Key Normalization for Transformers*.
  Findings of EMNLP 2020. arXiv:2010.04245.
- López-Rubio, Montes-Pérez, Palomo. *Enhanced QKNorm Normalization for
  Neural Transformers with the Lp Norm*. 2026. arXiv:2602.05006.
- Liu et al. *Swin Transformer*. ICCV 2021. arXiv:2103.14030.
- Hatamizadeh et al. *Swin UNETR*. BrainLes 2021. arXiv:2201.01266.
- MONAI documentation, `monai.networks.nets.SwinUNETR`.
