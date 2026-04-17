# Phase 2 — Model: Swin-UNETR with Lp-QKNorm

## Goal

Produce a drop-in generalization of the QKNorm attention mechanism
(*Henry et al., 2020, arXiv:2010.04245*) based on the Lp norm
(*López-Rubio et al., 2026, arXiv:2602.05006*), and splice it into the
windowed self-attention of MONAI's `SwinUNETR` in a way that is
**numerically identical to the original QKNorm when `p = 2`** and
architecturally unchanged everywhere else.

The model must support both `spatial_dims=2` (2D slice-level training) and
`spatial_dims=3` (3D volume-level training). The Phase 1 standardized HDF5
stores complete 3D volumes; the DataModule's `spatial_mode` determines
which input shape the model receives. The attention patching logic is
identical for both — only the `SwinUNETR` constructor argument changes.

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

1. **Which class in MONAI holds the windowed attention for SwinUNETR**.
   Inspect both `monai.networks.nets.SwinUNETR(spatial_dims=2, ...)` and
   `SwinUNETR(spatial_dims=3, ...)` at runtime, walk the module tree, and
   identify the attention module. Confirm the same `WindowAttention` class
   is used for both spatial dims (expected). As of MONAI ≥ 1.3.0 this should
   be `monai.networks.blocks.patchembedding.WindowAttention` or equivalent
   in `monai.networks.blocks.selfattention`. **Confirm**, do not assume.
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
    img_size: tuple[int, ...],
    in_channels: int,
    out_channels: int,
    feature_size: int,
    spatial_dims: Literal[2, 3] = 2,
    lp_cfg: LpQKNormConfig | None = None,
    patch_base: Literal["monai"] = "monai",
) -> nn.Module:
    """Build a SwinUNETR with Lp-QKNorm attention.

    Supports both 2D (slice-level) and 3D (volume-level) via
    ``spatial_dims``. The attention patching is identical for both;
    only the SwinUNETR constructor argument changes.

    The ``in_channels`` and ``out_channels`` are read from the HDF5
    header at training time (``header.n_modalities`` and
    ``header.n_label_classes``), making the model dataset-agnostic.

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

### 5. Drop-in compatibility (2D and 3D)

```python
@pytest.mark.parametrize("spatial_dims,img_size,x_shape", [
    (2, (224, 224), (2, 1, 224, 224)),
    (3, (96, 96, 96), (1, 1, 96, 96, 96)),
])
def test_attention_shape_preserved(spatial_dims, img_size, x_shape):
    model_stock = monai.networks.nets.SwinUNETR(
        img_size=img_size, in_channels=1, out_channels=1,
        feature_size=24, spatial_dims=spatial_dims,
    )
    model_lp = build_swin_unetr_lp(
        img_size=img_size, in_channels=1, out_channels=1,
        feature_size=24, spatial_dims=spatial_dims,
        lp_cfg=LpQKNormConfig(p=3.0),
    )
    x = torch.randn(*x_shape)
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

======

# Phase 2 — Weight Initialization Spec (From-Scratch Regime)

## Context

All experiments in the primary run are trained **from scratch** (no
pretrained backbone). The rationale is methodological: this is a
mechanistic study of the $\ell_p$ parameter inside `LpWindowAttention`,
and pretrained weights — which were shaped under vanilla scaled-dot-product
attention, i.e., implicitly the $p = 2$ regime — would bias the learned
Q/K geometry toward $\ell_2$-amenable configurations and confound the
$p$ effect.

Dataset budget for the primary sweep: **BraTS-MEN, ~1000 subjects**
(meningioma). This is sufficient for a 2D slice-level Swin-UNETR with
`feature_size=24` when combined with aggressive augmentation and
early stopping. Pretrained-backbone runs are deferred to an ablation row
(see [§ Ablation](#ablation)).

## Required additions to `ModelConfig`

Add three fields to `lpqknorm.training.module.ModelConfig`:

```python
@dataclass(frozen=True)
class ModelConfig:
    img_size: tuple[int, int] = (224, 224)
    in_channels: int = 1
    out_channels: int = 1
    feature_size: int = 24

    # --- New: initialization spec ---
    init_scheme: Literal["scratch_trunc_normal", "pretrained_ssl"] = "scratch_trunc_normal"
    linear_init_std: float = 0.02
    alpha_init_scheme: Literal["log_dk", "sqrt_dk", "fixed"] = "log_dk"
    alpha_init_fixed: float | None = None  # used only if alpha_init_scheme == "fixed"
```

Mirror the same fields in `configs/model/default.yaml`:

```yaml
img_size: [224, 224]
in_channels: 1
out_channels: 1
feature_size: 24
p: null

init_scheme: scratch_trunc_normal
linear_init_std: 0.02
alpha_init_scheme: log_dk
alpha_init_fixed: null
```

## Initialization rules (scheme = `scratch_trunc_normal`)

Applied inside `build_swin_unetr_lp` **after** MONAI `SwinUNETR`
construction and the `LpWindowAttention` patch, via a single
`model.apply(_init_weights)` pass. The function must be implemented in
`lpqknorm/models/init.py` and unit-tested in isolation.

| Module type | Weight | Bias | Notes |
|---|---|---|---|
| `nn.Linear` (incl. `qkv`, `proj`, MLP) | `trunc_normal_(std=linear_init_std)` | `zeros_` | `std = 0.02` per Swin (Liu et al., 2021) and ViT (Dosovitskiy et al., 2021). |
| `nn.Conv{2,3}d` (incl. patch embed) | `trunc_normal_(std=linear_init_std)` | `zeros_` | Keep patch-embed consistent with Swin original. |
| `nn.LayerNorm` | `ones_` ($\gamma$) | `zeros_` ($\beta$) | Standard pre-norm Transformer init. |
| `relative_position_bias_table` | `trunc_normal_(std=linear_init_std)` | — | Already handled by MONAI; re-apply for determinism. |
| `LpQKNorm.alpha_raw` | per `alpha_init_scheme` (below) | — | Must be identical across all $p$ values in the sweep. |

### $\alpha$ initialization

Let $d_k = \texttt{feature\_size} / \texttt{num\_heads\_per\_stage}$
(computed **per stage**, since Swin-UNETR heads grow with depth).

Let $\alpha^\star$ denote the target effective scale. Set
`alpha_raw = softplus_inverse(alpha_star)` so that
`softplus(alpha_raw) = alpha_star` exactly.

| `alpha_init_scheme` | $\alpha^\star$ | Reference |
|---|---|---|
| `log_dk` (**default**) | $\log d_k$ | Henry et al., 2020, §3.2. |
| `sqrt_dk` | $\sqrt{d_k}$ | Alternative matching scaled-dot-product magnitude. |
| `fixed` | `alpha_init_fixed` | Ablation only. Raise `LpInitError` if `None`. |

At $p = 2$ with `alpha_init_scheme = log_dk`, the functional scale of the
logits $\alpha \langle \hat{q}, \hat{k} \rangle \in [-\log d_k, \log d_k]$
matches Henry et al. (2020). This is the clean baseline against which the
$p$ sweep is compared.

**Hard constraint (controlled experiment):** `alpha_init_scheme`,
`alpha_init_fixed`, and `linear_init_std` must be **identical across
all $p$ values and all folds** in a sweep. The `Manifest` writer should
hash these fields and assert equality across runs grouped by experiment.

## Numerical stability

`softplus_inverse(x)` for $x > 0$:

$$
\text{softplus}^{-1}(x) = \log(e^{x} - 1).
$$

For $x > 20$ use the stable branch
`softplus_inverse(x) = x + log(1 - exp(-x))` to avoid `inf`. Implement as:

```python
def softplus_inverse(x: float) -> float:
    """Numerically stable inverse of softplus for x > 0."""
    if x <= 0.0:
        raise LpInitError(f"alpha target must be > 0, got {x}")
    if x > 20.0:
        return x + math.log1p(-math.exp(-x))
    return math.log(math.expm1(x))
```

## Ablation

A single ablation row uses pretrained self-supervised Swin-UNETR weights
(Tang et al., 2022; arXiv:2111.14791), selected via
`init_scheme = pretrained_ssl`. When loading, `LpQKNorm` state
(`alpha_raw`) is **not** in the checkpoint and falls back to the scheme
above; all other weights are loaded strict=False with an assertion that
the set of missing keys equals exactly
`{"*.attn.lp_qknorm.alpha_raw"}`. Any other missing key aborts with
`LpInitError`.

## Acceptance tests

Add to `tests/unit/test_init.py`:

1. **Default init runs end-to-end.** Build a tiny model
   (`feature_size=12`) with `init_scheme="scratch_trunc_normal"`;
   assert no `NaN`/`Inf` in any parameter.
2. **Linear-weight empirical std.** For each `nn.Linear` with
   $\geq 1024$ elements, assert
   `abs(weight.std() - 0.02) < 0.004` (sample tolerance).
3. **LayerNorm defaults.** Every `nn.LayerNorm`: `weight == 1`,
   `bias == 0`.
4. **$\alpha$ target scale.** Per stage $s$,
   `softplus(alpha_raw) ≈ log(d_k_s)` within `1e-6`.
5. **Determinism across $p$.** Initialize models at
   $p \in \{2, 2.5, 3, 3.5, 4\}$ with the same seed; assert all
   `qkv`, `proj`, MLP, LN, and `relative_position_bias_table` tensors
   are **byte-identical**. Only `alpha_raw` may differ — and only if it
   was configured to differ (by default it must not).
6. **Forward pass smoke.** $16 \times 1 \times 224 \times 224$ input,
   assert output shape and finite activations.

## References

- Liu, Z. et al. *Swin Transformer*. ICCV 2021. arXiv:2103.14030.
- Hatamizadeh, A. et al. *Swin UNETR*. BrainLes 2021. arXiv:2201.01266.
- Henry, A. et al. *Query-Key Normalization for Transformers*.
  Findings of EMNLP 2020. arXiv:2010.04245.
- López-Rubio, E. et al. *Enhanced QKNorm with the Lp Norm*. 2026.
  arXiv:2602.05006.
- Dosovitskiy, A. et al. *An Image is Worth 16×16 Words*. ICLR 2021.
  arXiv:2010.11929.
- Tang, Y. et al. *Self-Supervised Pre-Training of Swin Transformers for
  3D Medical Image Analysis*. CVPR 2022. arXiv:2111.14791.
- He, K. et al. *Delving Deep into Rectifiers*. ICCV 2015.
  arXiv:1502.01852.