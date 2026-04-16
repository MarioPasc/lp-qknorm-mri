# Phase 4 — Mechanistic Probes

## Goal

Collect the five mechanistic probes defined in the design discussion, at
designated epochs for every run, using forward hooks on stage-1 (finest-
resolution) windowed attention. These probes operationalise the claim:

*higher feature peakiness* (Probe 1) →
*lower per-query attention entropy* (Probe 2) →
*more attention mass on lesion tokens* (Probe 3) →
*larger logit gap between lesion and background keys* (Probe 4) →
*tighter spatial alignment between attention and lesion mask* (Probe 5).

If the `p`-dependence of these probes follows the predicted ordering while
the headline Dice / small-lesion recall does not, we know the mechanism is
present but not sufficient. If both agree, we have a mechanistically
interpretable result. Either outcome is informative.

This phase assumes Phases 1–3 have passed their acceptance tests. It
depends on the hook infrastructure defined in `src/lpqknorm/models/hooks.py`
from Phase 2.

## Mathematical definitions

Let stage-1 contain `L` windowed-attention blocks, each producing attention
tensors of shape `(B, n_windows, H, W², W²)` where `W` is the window size,
`H` is the number of heads, and `n_windows` is the number of non-overlapping
windows per image. For each block, hooks expose `q, k ∈ ℝ^{… × d_k}` pre-
and post-normalisation, logits `s ∈ ℝ^{… × W² × W²}`, and final attention
`A ∈ [0, 1]^{… × W² × W²}`.

Each token is tagged as **lesion** or **background** by downsampling the
ground-truth mask to the stage-1 resolution (`image_size / 4 × image_size / 4`
for MONAI's 2D SwinUNETR with `patch_size=2, embed_dim=feature_size`) and
rebuilding the window partition exactly as Swin does.

### Probe 1 — Feature peakiness

For each token `i` and each vector `v_i ∈ {q_i, k_i}` (pre-Lp-normalisation):

```
ρ(v_i) = ||v_i||_∞ / (||v_i||_2 + ε),        ρ ∈ [1 / √d_k, 1]
```

Lower bound `1 / √d_k` is attained when `v_i` is uniform; upper bound `1` is
attained when `v_i` is one-hot. A larger `ρ` means a peakier coordinate
distribution. The probe tests whether training under `Lp` with `p > 2`
causes lesion tokens to develop systematically peakier `q` / `k` vectors
than background tokens.

### Probe 2 — Per-query attention entropy

```
H_i = -Σ_{j=1}^{W²} A_{ij} log A_{ij},       H_i ∈ [0, log(W²)]
```

Predicted to decrease (sharper attention) on lesion queries as `p` grows,
up to the interior maximum of the toy model.

### Probe 3 — Lesion attention mass

For lesion queries `i ∈ L` and the set `L_win(i)` of lesion tokens in the
same window:

```
M_i = Σ_{j ∈ L_win(i)} A_{ij},              M_i ∈ [0, 1]
```

Predicted to increase with `p` on the small-lesion stratum.

### Probe 4 — Logit gap

```
Δ_i = max_{j ∈ L_win(i)} s_{ij}  -  median_{j ∉ L_win(i)} s_{ij}
```

Empirical analogue of the toy-model `Δ(p)`. Should show an interior maximum
in `p`, consistent with Phase 2's interior-maximum test.

### Probe 5 — Attention–mask IoU

For each lesion query `i`, binarise attention by the top-`k` tokens where
`k = |L_win(i)|`, producing mask `T_i ∈ {0, 1}^{W²}`. Let
`M ∈ {0, 1}^{W²}` be the ground-truth lesion window mask:

```
IoU_i = |T_i ∩ M| / |T_i ∪ M|
```

## Open questions the agent must resolve

1. **Exact downsampling geometry** from image space to stage-1 token space
   in MONAI's 2D SwinUNETR. Walk the patch-embed module; read out the
   effective stride (usually `patch_size × 1` at stage 0 and
   `patch_size × 2` at stage 1 after merging). Write a unit test that
   takes an image with a known single-pixel mask and verifies the
   corresponding lesion token index.
2. **Window partition reconstruction**. The hook receives `x` as
   `(num_windows * B, W², C)` after `window_partition`. Reconstruct the
   spatial layout to match tokens to lesion masks; verify by inverting the
   partition on a known test pattern.
3. **Whether relative position bias should be included in `s` for Probe 4**.
   Yes — `s` means the full pre-softmax logit, bias included, because that
   is what the softmax sees. Document this choice.
4. **Handling the shifted-window alternation**. Stage 1 in Swin contains
   two blocks: W-MSA and SW-MSA. Run probes on both, label by block index,
   and report both. Do not average across them — the shift changes which
   tokens belong to which window.

## I/O contract

### When probes run

At a fixed schedule:
- Epoch 0 (before any training) — baseline.
- Epochs 1, 5, 10, 25, 50 — trajectory.
- Best-dice checkpoint.
- Best-small-recall checkpoint.
- Final epoch.

The probe batch is **fixed across checkpoints and across runs**: the first
`N_probe = 32` validation slices of fold `fold`, sorted by `subject_id` then
`slice_index`. Deterministic; no augmentation.

### Output format (`probes/epoch_{N}.h5`)

One HDF5 file per checkpoint. Grouped by block:

```
/block_0_wmsa/
    peakiness_q      (N_tokens,) float32
    peakiness_k      (N_tokens,) float32
    entropy          (N_queries,) float32
    lesion_mass      (N_lesion_queries,) float32
    logit_gap        (N_lesion_queries,) float32
    attention_iou    (N_lesion_queries,) float32
    is_lesion        (N_tokens,) bool       # per-token tag
    subject_id       (N_queries,) S16       # for per-patient analysis
    slice_index      (N_queries,) int32
    window_index     (N_queries,) int32
    head_index       (N_queries,) int8
    alpha            ()       float32       # scalar
/block_1_swmsa/
    ...
```

All tensors stored per-query or per-token (no pre-aggregation). This keeps
Phase 5 free to aggregate any way it likes — per-patient, per-lesion-size,
per-head, across or within windows.

### Public API (`src/lpqknorm/probes/`)

```python
# base.py
class Probe(Protocol):
    name: str
    def compute(self, capture: AttentionCapture,
                lesion_mask_tokens: Tensor) -> ProbeResult: ...

@dataclass(frozen=True)
class ProbeResult:
    name: str
    per_token: Tensor | None       # shape (N_tokens,) or None
    per_query: Tensor | None
    metadata: dict[str, Any]

# recorder.py
class ProbeRecorder:
    """Orchestrates hook registration, probe computation, and HDF5 writing."""
    def __init__(self, probes: Sequence[Probe], output_dir: Path) -> None: ...
    def run(self, model: nn.Module, dataloader: DataLoader,
            epoch_tag: str | int, device: torch.device) -> Path: ...
```

Each probe is its own small module (≈ 30 LOC each):

```python
# peakiness.py
class FeaturePeakiness(Probe):
    def __init__(self, target: Literal["q", "k"]) -> None: ...

# entropy.py
class AttentionEntropy(Probe): ...

# lesion_mass.py
class LesionAttentionMass(Probe): ...

# logit_gap.py
class LesionBackgroundLogitGap(Probe): ...

# attention_iou.py
class AttentionMaskIoU(Probe): ...
```

## Design notes

**Why store per-token / per-query rather than aggregated statistics.** The
aggregation strategy (per-patient? per-lesion-size? conditional on detection
success?) should not be committed at probe time. Raw tensors compress well
in HDF5 with blosc and are under ~100 MB per checkpoint.

**Why HDF5 instead of parquet.** Per-token arrays are not tabular — they
have variable length per batch (depends on how many lesion tokens each
window contains), and they live alongside small scalars like `alpha`. HDF5
handles heterogeneous numeric groups naturally; parquet does not.

**Why a fixed probe batch.** Comparing Probe 3 across epochs and across `p`
values is only meaningful if the inputs are identical. Otherwise an
epoch-over-epoch change could be "the model changed" or "this epoch
happened to see harder slices".

**Handling no-lesion windows.** Not every window in a lesion-containing
slice actually contains lesion tokens. Per-query probes (2, 3, 4, 5) are
computed only for **windows that contain at least one lesion token**, and
Probe 3–5 only for **queries that are themselves lesion tokens**. Document
the filtering rule and expose token counts in the output metadata.

## Implementation checklist

1. `probes/base.py` — `Probe` protocol, `ProbeResult` dataclass, registry.
2. `probes/peakiness.py`, `entropy.py`, `lesion_mass.py`, `logit_gap.py`,
   `attention_iou.py` — one probe each.
3. `probes/recorder.py` — orchestrator. Takes a list of `Probe` instances,
   registers hooks via `AttentionHookRegistry`, iterates the fixed probe
   loader, concatenates results across batches, writes HDF5.
4. `training/callbacks.py::ProbeCallback` — invokes `ProbeRecorder` at the
   scheduled epochs and on best-checkpoint saves.
5. `cli/probe.py` — standalone post-hoc probe extraction given a checkpoint
   path (useful for re-running probes without re-training).
6. Lesion-mask token tagger utility in `probes/tokenization.py`: maps
   `(B, 1, H, W)` masks to per-token lesion flags respecting the stage-1
   window partition. Carefully tested.

## Acceptance tests

### 1. **Probe correctness on synthetic attention** (`test_probes_synthetic.py`)

These are the most important probe tests. They verify theoretical predictions
on controlled inputs, independent of the model.

```python
def test_peakiness_bounds():
    # One-hot: peakiness = 1
    v = torch.eye(64)[0].unsqueeze(0)
    assert FeaturePeakiness("q").compute_value(v) == pytest.approx(1.0)
    # Uniform: peakiness = 1/sqrt(d)
    v = torch.ones(1, 64) / math.sqrt(64)
    assert FeaturePeakiness("q").compute_value(v) == pytest.approx(
        1.0 / math.sqrt(64), abs=1e-4
    )

def test_entropy_bounds():
    W2 = 49  # 7x7 window
    uniform = torch.ones(1, W2) / W2
    H = AttentionEntropy().compute_value(uniform)
    assert H == pytest.approx(math.log(W2), abs=1e-5)
    onehot = torch.zeros(1, W2); onehot[0, 0] = 1.0
    assert AttentionEntropy().compute_value(onehot) == pytest.approx(0.0, abs=1e-5)

def test_lesion_mass_range():
    A = torch.softmax(torch.randn(10, 49), dim=-1)
    lesion_mask = torch.zeros(49, dtype=torch.bool); lesion_mask[:4] = True
    M = LesionAttentionMass().compute_per_query(A, lesion_mask)
    assert (M >= 0).all() and (M <= 1).all()
```

### 2. Toy-model interior maximum (repeat of Phase 2's test 8 end-to-end)

Run a synthetic forward pass where Q/K contain a known peaky lesion token
and a known diffuse background token, evaluate Probe 4 for
`p ∈ {1.5, 2, 2.5, 3, 3.5, 4, 5}`, assert the empirical argmax of `Δ(p)`
is in `{3, 4}`. This connects real-tensor geometry to the theoretical
prediction.

### 3. Token-level lesion tagging (`test_tokenization.py`)

```python
def test_lesion_token_tagging_roundtrip():
    # Place a 2x2 lesion at pixel (100, 100) in a 224x224 image
    mask = torch.zeros(1, 1, 224, 224)
    mask[0, 0, 100:102, 100:102] = 1
    token_flags = mask_to_token_flags(
        mask, stage_stride=(4, 4), window_size=(7, 7)
    )
    # Recover the expected token index
    expected = (100 // 4) * (224 // 4) + (100 // 4)
    assert token_flags[0, expected]
    assert token_flags.sum() >= 1
```

### 4. Window partition reconstruction

Given a known integer-pattern image, partition into windows using the same
utility Swin uses and verify shape, token-to-window-to-head mapping.

### 5. End-to-end probe pipeline smoke test

```python
def test_probe_recorder_end_to_end(tmp_path):
    model = build_swin_unetr_lp(..., lp_cfg=LpQKNormConfig(p=3.0))
    recorder = ProbeRecorder(
        probes=[FeaturePeakiness("q"), FeaturePeakiness("k"),
                AttentionEntropy(), LesionAttentionMass(),
                LesionBackgroundLogitGap(), AttentionMaskIoU()],
        output_dir=tmp_path,
    )
    loader = _tiny_fixed_loader()
    out = recorder.run(model, loader, epoch_tag="test", device="cpu")
    with h5py.File(out) as f:
        assert "block_0_wmsa" in f
        assert "peakiness_q" in f["block_0_wmsa"]
        assert f["block_0_wmsa/peakiness_q"].shape[0] > 0
```

### 6. Probe values are invariant across identical runs

Two calls to `ProbeRecorder.run` on the same model weights, same fixed
loader, same device, produce bit-identical probe tensors (assert equality,
not allclose — hooks must not introduce non-determinism).

### 7. No autograd retention

`ProbeRecorder.run` must be wrapped in `torch.inference_mode()`; test that
after running, `torch.cuda.memory_allocated()` has returned to the
pre-probe level within a small tolerance.

## References

- Vig. *A Multiscale Visualization of Attention in the Transformer Model*.
  ACL 2019. (probe-style attention analysis).
- Michel et al. *Are Sixteen Heads Really Better than One?* NeurIPS 2019.
  (head-level analysis, relevant for per-head probe breakdowns).
- Rogers et al. *A Primer in BERTology*. TACL 2020. doi:10.1162/tacl_a_00349.
  (methodology for mechanistic attention probes).
- Liu et al. *Swin Transformer*. ICCV 2021. arXiv:2103.14030.
