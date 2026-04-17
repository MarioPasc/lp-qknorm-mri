# Phase 4 — Mechanistic Probes

## Goal

Collect the mechanistic evidence required to interpret the $L^p$-QKNorm
sweep. The evidence is organised in four tiers, all computed on the same
fixed probe batch at designated checkpoints:

1. **Scalar probes on attention and features** (Probes 1–8). Operationalise
   the causal chain:

   *higher feature peakiness* (P1) →
   *lower per-query attention entropy* (P2) →
   *more attention mass on lesion tokens* (P3) →
   *larger logit gap between lesion and background keys* (P4) →
   *tighter spatial alignment between attention and lesion mask* (P5 / P6) →
   *lesion-linearly-separable token features* (P7) →
   *lower effective feature dimensionality on lesion tokens* (P8).

2. **Attention-map persistence.** Store the full per-window attention
   tensor, the input slice, and the lesion mask for every probe sample.
   This is the substrate for qualitative overlay figures, Attention
   Rollout, and any post-hoc analysis that cannot be committed at probe
   time.

3. **Causal interventions (activation patching).** Swap the stage-0
   attention intermediates from one $p$ run into the forward pass of
   another, and measure the Dice delta. This converts the correlational
   probe chain into a causal one.

4. **Controls.** Trajectories of $\alpha$ and the relative position bias,
   an init-time (epoch 0) probe baseline, and window-boundary
   stratification of lesion tokens.

If the $p$-dependence of the scalar probes follows the predicted ordering,
**and** activation patching at stage-0 recovers a substantial fraction of
the Dice gap between $p^*$ and $p=2$, the mechanism is both present and
causal. If the probes agree but patching is null, the mechanism is present
but not localised at the probed stage. If the headline Dice does not move
with any of this, the mechanism is decoupled from the outcome. Each
outcome is publishable.

This phase assumes Phases 1–3 have passed their acceptance tests. It
depends on the hook infrastructure defined in `src/lpqknorm/models/hooks.py`
from Phase 2.

**2D/3D note.** All scalar probes operate on the flattened window token
sequence regardless of spatial dimensionality. The only dimensionality-
specific code is the tokenisation mapping (`mask_to_token_flags`): 2D maps
`(B, 1, H, W)` masks via a 2D window partition; 3D maps `(B, 1, D, H, W)`
via a 3D window partition. Attention-map persistence and activation
patching are likewise dimension-agnostic once tokenisation is correct.
The current study runs 2D probes on stage-0 attention (0-indexed; the
finest-resolution stage); 3D probes are reserved for future work.

**Naming convention.** Throughout this document "stage 0" is the
0-indexed finest-resolution stage (`model.swinViT.layers1[0]` in MONAI).
Earlier drafts used "stage 1" (1-indexed). The code in
`src/lpqknorm/models/hooks.py` is authoritative; see the stage mapping
in its module docstring.

## Mathematical definitions

Let the probed stage contain $L=2$ windowed-attention blocks (W-MSA and
SW-MSA), each producing

- raw pre-norm $q, k \in \mathbb{R}^{B\, n_{\text{win}} \times H \times W^2 \times d_k}$,
- post-Lp-norm $\hat q, \hat k$ with $\|\hat q_i\|_p = \|\hat k_j\|_p = 1$,
- logits $s_{ij} = \alpha \langle \hat q_i, \hat k_j \rangle + b^{\text{rel}}_{ij}$,
- attention $A_{ij} = \operatorname{softmax}_j(s_{ij}) \in [0,1]$,

where $W$ is the window size, $H$ the number of heads, $n_{\text{win}}$
the number of non-overlapping windows per image, $d_k$ the per-head
dimension, $\alpha = \operatorname{softplus}(\alpha_{\text{raw}})$ the
learnable scalar, and $b^{\text{rel}}$ the (learned) relative position
bias.

Each token is tagged **lesion** or **background** by downsampling the
ground-truth mask to the stage-0 token resolution (stride
`patch_stride` — see `ProbeRecorder.patch_stride`) and rebuilding the
window partition exactly as Swin does.

### Probe 1 — Feature peakiness (per-vector sparsity)

For each token $i$ and each vector $v_i \in \{q_i, k_i\}$ (pre-Lp-norm):

$$
\rho(v_i) = \frac{\|v_i\|_\infty}{\|v_i\|_2 + \varepsilon},
\qquad \rho \in \left[\tfrac{1}{\sqrt{d_k}},\, 1\right].
$$

Lower bound $1/\sqrt{d_k}$ attained for uniform $v_i$; upper bound $1$ for
one-hot. Tests whether $p > 2$ drives lesion tokens toward peakier
coordinate distributions than background tokens.

### Probe 2 — Per-query attention entropy

$$
H_i = -\sum_{j=1}^{W^2} A_{ij} \log A_{ij}, \qquad H_i \in [0, \log W^2].
$$

Predicted to decrease (sharper attention) on lesion queries as $p$ grows,
up to the interior maximum of the toy model.

### Probe 3 — Lesion attention mass

For lesion queries $i \in \mathcal L$ and the set $\mathcal L_{\text{win}}(i)$
of lesion tokens in the same window:

$$
M_i = \sum_{j \in \mathcal L_{\text{win}}(i)} A_{ij}, \qquad M_i \in [0, 1].
$$

Predicted to increase with $p$ on the small-lesion stratum.

### Probe 4 — Logit gap

$$
\Delta_i = \max_{j \in \mathcal L_{\text{win}}(i)} s_{ij}
          - \operatorname{median}_{j \notin \mathcal L_{\text{win}}(i)} s_{ij}.
$$

Empirical analogue of the toy-model $\Delta(p)$. Should show an interior
maximum in $p$. Note $s$ includes the relative position bias; this is
deliberate (it is what the softmax sees).

### Probe 5 — Attention–mask IoU

For each lesion query $i$, binarise attention by the top-$k$ tokens with
$k = |\mathcal L_{\text{win}}(i)|$ to obtain $T_i \in \{0,1\}^{W^2}$. Let
$M \in \{0,1\}^{W^2}$ be the ground-truth lesion window mask:

$$
\operatorname{IoU}_i = \frac{|T_i \cap M|}{|T_i \cup M|}.
$$

### Probe 6 — Spatial localisation error (NEW)

Peak-based complement to Probe 5. For each lesion query $i$, let
$j^*_i = \arg\max_j A_{ij}$ be the spatial argmax of the attention row,
and let $\bar\mu_i = \operatorname{centroid}(\mathcal L_{\text{win}}(i))$
be the centroid of lesion tokens in the same window, both in
$(y, x)$ intra-window token coordinates. Then

$$
\operatorname{SLE}_i = \|\mathrm{coord}(j^*_i) - \bar\mu_i\|_2
\in [0,\, W\sqrt{2}].
$$

Probe 5 measures *coverage* (set overlap); Probe 6 measures *peakedness
location* (does the attention peak sit on the lesion?). Predicted to
decrease with $p$. Reported alongside IoU because the two can disagree:
diffuse attention with correct centroid gives high IoU and high SLE;
sharply off-centre attention gives low IoU and low SLE.

### Probe 7 — Linear-probe separability (NEW)

Substitute for sparse-autoencoder dictionary learning at this scale
(*Alain & Bengio, "Understanding intermediate layers using linear
classifier probes", ICLR workshop 2016, arXiv:1610.01644*). For each
checkpoint, each block, and each head $h$, fit an $L^1$-penalised
logistic regression

$$
\mathcal L_{\text{LP}}(w, b) = \frac{1}{N}\sum_{i=1}^N \log\bigl(1 + e^{-y_i (w^\top x_i^{(h)} + b)}\bigr) + \lambda \|w\|_1,
$$

on token features $x_i^{(h)} \in \mathbb{R}^{d_k}$ (pre-norm $q^{(h)}$),
with binary lesion label $y_i \in \{-1, +1\}$. Report three scalars per
$(block, head)$:

- Balanced accuracy $\operatorname{BA} = \tfrac12(\mathrm{TPR} + \mathrm{TNR})$
  under 5-fold cross-validation within the probe batch,
- Decision-boundary sparsity $\sigma(w) = \|w\|_1 / (\|w\|_2 + \varepsilon)$
  (analogue of Probe 1 for the *classifier*, not the features),
- Mean margin $\bar m = \tfrac{1}{N}\sum_i y_i (w^\top x_i^{(h)} + b)$.

Predicted: $\operatorname{BA}(p)$ and $\bar m(p)$ grow with $p$ on lesion
queries; $\sigma(w)$ decreases (fewer coordinates carry the decision) as
$p$ grows. Use `sklearn.linear_model.LogisticRegression` with
`penalty="l1", solver="liblinear"` and $\lambda$ selected by inner
cross-validation.

**Why not a Sparse Autoencoder.** With $d_k \in [16, 48]$ and
$N \approx 10^5$ pooled tokens per checkpoint, the expansion factor and
activation pool are well below the regime where SAEs disentangle
polysemantic features (*Bricken et al., "Towards Monosemanticity",
Anthropic 2023*; *Cunningham et al., "Sparse Autoencoders Find Highly
Interpretable Features in Language Models", ICLR 2024,
arXiv:2309.08600*). An SAE on the stage-3 bottleneck ($d = 8 \cdot
\text{feature\_size}$) would be geometrically defensible; we defer it as
an optional appendix via `sae_lens`
(<https://github.com/jbloomAus/SAELens>) if reviewers request it.

### Probe 8 — Spectral / participation-ratio (NEW)

Population-level complement to the per-vector peakiness of Probe 1.
Pool token features $X^{(\text{lesion})} \in \mathbb{R}^{N_L \times d_k}$ and
$X^{(\text{bg})} \in \mathbb{R}^{N_B \times d_k}$, centred. Let
$\lambda_1 \geq \dots \geq \lambda_{d_k} \geq 0$ be the eigenvalues of the
empirical covariance $\Sigma = \tfrac{1}{N-1}(X-\bar X)^\top(X-\bar X)$.
Report:

- Participation ratio
  $\operatorname{PR} = \bigl(\sum_i \lambda_i\bigr)^2 / \sum_i \lambda_i^2 \in [1, d_k]$
  (*Gao et al., bioRxiv 2017, doi:10.1101/214262*).
- Stable rank $\operatorname{SR} = \|X\|_F^2 / \|X\|_2^2$.
- Normalised eigenvalue vector $\tilde\lambda_i = \lambda_i / \sum_j \lambda_j$.

Predicted: lesion tokens show lower PR under $p > 2$ (activity concentrated
on fewer principal directions); background tokens unchanged.

---

## Attention-map persistence (NEW)

The scalar probes reduce each attention tensor to a handful of numbers.
For overlay figures, Attention Rollout (*Abnar & Zuidema, "Quantifying
Attention Flow in Transformers", ACL 2020, arXiv:2005.00928*),
transformer-specific CAM variants (*Chefer et al., "Transformer
Interpretability Beyond Attention Visualization", CVPR 2021,
arXiv:2012.09838*), and post-hoc claims we cannot anticipate at probe
time, the raw attention tensor and input slice must be stored verbatim.

### What is stored

Per probe sample $s$, per block $b \in \{\text{W-MSA}, \text{SW-MSA}\}$:

- `attention_full[s, b]` — $A \in [0,1]^{n_{\text{win}} \times H \times W^2 \times W^2}$, `float16`.
- `logits_full[s, b]` — $s \in \mathbb{R}^{n_{\text{win}} \times H \times W^2 \times W^2}$, `float16` (for Probe 4 reproducibility without re-running).
- `alpha[b]` — scalar $\alpha$ for the block, `float32`.

Per probe sample (block-independent):

- `image[s]` — input slice $\in \mathbb{R}^{C \times H_{\text{img}} \times W_{\text{img}}}$, `float16`.
- `mask[s]` — lesion mask $\in \{0,1\}^{H_{\text{img}} \times W_{\text{img}}}$, `uint8`.
- `subject_id[s]`, `slice_index[s]` — provenance.

Per block (scalar metadata): `window_size`, `shift_size`, `patch_stride`,
`n_heads`, `stage_index`, `block_index`.

### Storage budget

For the default configuration (stage 0, `patch_stride = 2`, W=7, 3 heads,
224×224 slices, $N_{\text{probe}} = 32$, 9 checkpoints):

$$
\underbrace{n_{\text{win}}\,H\,W^4\,2\,\text{B}}_{\approx 3.7\text{ MB / slice / block}}
\times 2\text{ blocks} \times 32\text{ slices} \approx 240\text{ MB / ckpt},
$$

times 9 checkpoints × 18 runs gives $\approx 40$ GB. Within budget on the
Sandisk 2TB; `blosc:lz4` compression halves this in practice.

### Reconstruction to image space

Given a capture from block $b$ with shift $\Delta = 0$ for W-MSA and
$\Delta = W/2$ for SW-MSA, token-grid size
$(H_{\text{tok}}, W_{\text{tok}})$, a query with flat window index
$w \in \{0,\dots,n_{\text{win}}-1\}$ and intra-window index
$r \in \{0,\dots,W^2-1\}$, and an attended intra-window index
$c \in \{0,\dots,W^2-1\}$, the attended token's coordinates in the
(circularly-shifted) token grid are

$$
\begin{aligned}
(y_c^{\text{tok}}, x_c^{\text{tok}}) &=
\bigl(\, \lfloor w / (W_{\text{tok}}/W) \rfloor \cdot W + \lfloor c/W \rfloor,\;
(w \bmod (W_{\text{tok}}/W)) \cdot W + (c \bmod W) \,\bigr), \\
(y_c, x_c) &= \bigl(\, (y_c^{\text{tok}} - \Delta) \bmod H_{\text{tok}},\;
(x_c^{\text{tok}} - \Delta) \bmod W_{\text{tok}} \,\bigr).
\end{aligned}
$$

Image-space upsampling from $(H_{\text{tok}}, W_{\text{tok}})$ to
$(H_{\text{img}}, W_{\text{img}})$ is by the product of the patch-embed
stride and any prior merging. Use `torch.nn.functional.interpolate` with
`mode="bilinear", align_corners=False`, not a custom kernel.

### Deliverables in `probes/attention_maps.py`

- `reconstruct_query_heatmap(cap, query_idx, shift, grid_hw)` — returns a
  `(H_tok, W_tok)` attention map for a specified query.
- `attention_rollout(attentions)` — Abnar–Zuidema rollout with residual
  correction $\tfrac12 A + \tfrac12 I$.
- `overlay_figure(image, mask, heatmap)` — the publication-figure helper.

---

## Causal interventions — activation patching

Following *Meng et al., "Locating and Editing Factual Associations in GPT"
(ROME), NeurIPS 2022, arXiv:2202.05262* and *Heimersheim & Nanda, "How to
use and interpret activation patching", arXiv:2404.15255*. The target of
the intervention is the stage-0 attention of the $p = 2$ (baseline) model;
the source is the stage-0 attention of the $p = p^*$ model, selected per
fold as the argmax of small-lesion recall.

### Protocol

Let $f_\theta(\cdot)$ denote the full forward pass of a model with
parameters $\theta$. At stage 0, block $b \in \{0, 1\}$, for a single
probe slice $x$:

1. Forward pass through the **source** model $\theta^*$; cache the raw
   pre-norm capture $\mathcal C^{*}_b(x) = (q^*_b, k^*_b)$.
2. Forward pass through the **target** model $\theta^{(2)}$ with a
   replacement hook: when the target's stage-0 block-$b$ attention
   module fires, overwrite its $q$ (or $k$, or both) with
   $q^*_b$ (respectively $k^*_b$).
3. Let $\hat y^{\text{patch}}(x; b, \cdot)$ be the output of the patched
   target forward. Compute per-slice Dice $\mathrm D^{\text{patch}}$.

Unpatched references:

- $\mathrm D^{(*)} = \mathrm{Dice}(f_{\theta^*}(x), y)$ (source-only, upper comparison point).
- $\mathrm D^{(2)} = \mathrm{Dice}(f_{\theta^{(2)}}(x), y)$ (target-only, lower comparison point).

### Patching-effect score

The normalised recovery fraction

$$
\operatorname{PE}(b, \text{variant}) \;=\;
\frac{\mathrm D^{\text{patch}}(b, \text{variant}) - \mathrm D^{(2)}}{\mathrm D^{(*)} - \mathrm D^{(2)} + \varepsilon}
\in (-\infty,\, 1]
$$

over the variant set $\{q, k, (q,k), (\hat q, \hat k), s\}$. Interpretation:

- $\operatorname{PE} \approx 1$ — the mechanism is fully localised at
  stage-0 block $b$.
- $\operatorname{PE} \approx 0$ — the mechanism is elsewhere (deeper
  stages, decoder, loss surface).
- $\operatorname{PE} < 0$ — the patch destroys the target; the stage-0
  representation is tightly coupled with the rest of $\theta^{(2)}$ and
  cannot be swapped naively.

Both directions are run: $\theta^{(2)} \to \theta^*$ ("denoising"; what
is sufficient to recover the gain) and $\theta^* \to \theta^{(2)}$
("noising"; what is necessary for the gain). See Heimersheim & Nanda,
§"Denoising vs. noising".

### What is stored

Per (source $p^*$, target $p = 2$, fold, slice, block $b$, variant):

- `dice_patched`, `dice_source`, `dice_target`,
- `pe` (the normalised recovery fraction),
- `prediction_patched` (small, $\mathrm{uint8}$, compressed).

Total volume: scalar per (slice, block, variant) + one compressed
prediction per (slice, block, variant). $\approx 5$ MB per
(fold, checkpoint); negligible.

### Deliverables in `probes/patching.py`

- `PatchingConfig` — dataclass: source_checkpoint, target_checkpoint,
  stage, blocks, variants, direction.
- `ActivationPatcher` — registers *replacement* hooks on target's
  `LpWindowAttention` that overwrite `q`, `k`, `q_hat`, `k_hat`, or
  `logits` from a pre-computed source cache. Disables the target's own
  Lp norm when patching post-norm tensors (document why).
- `run_patching_sweep(probe_loader, config) -> h5 path`.

### Caveats and scope

- Patching requires the source and target to have **structurally
  identical** models (same `feature_size`, head count, window size). The
  Lp-QKNorm patching in Phase 2 is a drop-in replacement of the
  normalisation; shapes are preserved, so this holds by construction.
- Vanilla baseline ($p = \emptyset$) has no `q_hat, k_hat`; the variants
  $\{(\hat q, \hat k), s\}$ are undefined for vanilla sources. Document
  and skip.
- The stage-0 relative position bias is **not** patched by default. Patch
  the bias in a separate ablation variant (`rel_pos_bias`) to
  disentangle its contribution.

---

## Controls

### α trajectory

$\alpha$ absorbs magnitude effects and is trained alongside the weights.
A systematic $\alpha(p)$ dependence across training would confound any
claim that "$p$ causes peakiness" — the alternative is "$p$ causes $\alpha$
to rise, which causes peakiness via the softmax". Store

$$
\alpha_b^{(t)} = \operatorname{softplus}\bigl(\alpha_{\text{raw}, b}^{(t)}\bigr)
$$

for every block $b$ and every optimizer step $t$, not just probe
checkpoints. Hook this into `training/callbacks.py::AlphaLogger` (append
one line per step to `alpha_trajectory.jsonl`). One float per step is
negligible I/O.

Phase 5 must report Pearson correlation between $\alpha(p)$ at the best
checkpoint and the empirical logit gap $\Delta(p)$. A high correlation
($|r| > 0.7$) would require a discount on the "peakiness → logit gap"
claim.

### Relative position bias

$b^{\text{rel}} \in \mathbb{R}^{(2W-1)^2 \times H}$ is learned. It enters
the logits and therefore the softmax. Store the full bias tensor at every
probe checkpoint (size: $13^2 \cdot H \cdot 4 \text{ B} \approx 2$ KB per
block). Also store its entropy after softmax-normalisation per head,

$$
H^{\text{rel}}_h = -\sum_{r} \tilde b^{\text{rel}}_{r,h} \log \tilde b^{\text{rel}}_{r,h},
\qquad \tilde b^{\text{rel}}_{r,h} = \frac{\exp b^{\text{rel}}_{r,h}}{\sum_{r'} \exp b^{\text{rel}}_{r',h}},
$$

as a scalar per head. Tests whether the bias drifts differently under
different $p$.

### Init-time baseline (epoch 0)

The probe schedule already includes epoch 0 (before any training).
Explicitly flag these as the *architectural prior*: any probe effect
present at epoch 0 is a consequence of $L^p$-normalisation on
random-init weights, not of learning. Figure 4 in Phase 5 must overlay
epoch 0 curves in muted tone behind the trained-model curves.

### Window-boundary stratification

Small lesions sitting on or near window boundaries get radically
different neighbourhoods under W-MSA vs SW-MSA. Define the
window-boundary distance of a lesion token as

$$
d_{\text{wb}}(i) = \min\bigl(\, r_y^{(i)},\, W - 1 - r_y^{(i)},\, r_x^{(i)},\, W - 1 - r_x^{(i)} \,\bigr),
$$

where $(r_y^{(i)}, r_x^{(i)})$ is the intra-window coordinate of token
$i$ under W-MSA. Store $d_{\text{wb}}(i)$ per lesion query (int8, one
byte). Phase 5 stratifies Probes 3–6 by
$d_{\text{wb}} \in \{0, 1, 2, 3\}$ and tests whether the $p$ effect is
stronger for boundary-proximal lesions. This is a secondary analysis,
but it is the *one* place where the W-MSA / SW-MSA distinction in Swin
actually matters mechanistically.

---

## Open questions the agent must resolve

1. **Exact downsampling geometry** from image space to stage-0 token space
   in MONAI's 2D SwinUNETR. Walk the patch-embed module; read out the
   effective stride. Write a unit test that takes an image with a known
   single-pixel mask and verifies the corresponding lesion token index.
   This is a **blocker** — every downstream quantity depends on token-
   level alignment. See acceptance test AT3.
2. **Window partition reconstruction**. The hook receives $x$ as
   $(n_{\text{win}} B, W^2, C)$ after `window_partition`. Reconstruct the
   spatial layout to match tokens to lesion masks; verify by inverting the
   partition on a known test pattern. See AT4.
3. **Whether relative position bias should be included in $s$ for Probe 4**.
   Yes — $s$ means the full pre-softmax logit, bias included, because that
   is what the softmax sees. Probe 4 reports this value; the "bias-free"
   logit $\alpha \langle \hat q, \hat k\rangle$ can be recomputed offline
   from stored $(\hat q, \hat k, \alpha)$ if needed.
4. **Handling the shifted-window alternation**. Stage 0 contains W-MSA and
   SW-MSA. Run every probe on both, label by block index, report both.
   Do not average across them.
5. **Activation-patching direction of primary interest**. Default:
   denoising ($\theta^{(2)} \to \theta^*$ with source patches) because
   the headline claim is that $p^*$ *adds* capability over $p = 2$. Run
   noising as a secondary check.
6. **Selection of $p^*$ per fold** for activation patching. Default:
   $p^*_{\text{fold}} = \arg\max_p \,\overline{\text{small-lesion-recall}}_{\text{fold}}$
   on the validation set of that fold. Document ties.

---

## I/O contract

### When probes run

At a fixed schedule:

- Epoch 0 (before any training) — init baseline.
- Epochs 1, 5, 10, 25, 50 — training trajectory.
- Best-dice checkpoint.
- Best-small-recall checkpoint.
- Final epoch.

The probe batch is **fixed across checkpoints, across folds within a
dataset, and across $p$**: the first $N_{\text{probe}} = 32$ validation
slices of fold `fold`, sorted by `(subject_id, slice_index)`. Deterministic;
no augmentation.

Activation patching runs **once per fold**, at the best-small-recall
checkpoint of both source and target, on the same probe batch.

### Output format (`probes/epoch_{N}.h5`)

One HDF5 file per checkpoint. Groups:

```
/metadata/
    epoch_tag                      str
    commit_sha                     str
    patch_stride                   (2,) int32
    window_size                    () int32
    n_heads                        () int32
    stage_index                    () int32
    n_probe_samples                () int32
    rng_seed                       () int64

/block_0_wmsa/
    # --- Tier 1: scalars (Probes 1–6) ---
    peakiness_q                    (N_tokens,)           float32
    peakiness_k                    (N_tokens,)           float32
    entropy                        (N_queries,)          float32
    lesion_mass                    (N_lesion_queries,)   float32
    logit_gap                      (N_lesion_queries,)   float32
    attention_iou                  (N_lesion_queries,)   float32
    spatial_localization_error     (N_lesion_queries,)   float32
    # --- Tier 1: Probe 7 (linear probe) ---
    lp_balanced_accuracy           (n_heads,)            float32
    lp_weight_sparsity             (n_heads,)            float32
    lp_margin                      (n_heads,)            float32
    # --- Tier 1: Probe 8 (spectral) ---
    pr_lesion                      ()                    float32
    pr_background                  ()                    float32
    stable_rank_lesion             ()                    float32
    stable_rank_background         ()                    float32
    eigenvalues_lesion             (d_k,)                float32
    eigenvalues_background         (d_k,)                float32
    # --- Provenance per query / per token ---
    is_lesion                      (N_tokens,)           bool
    subject_id                     (N_queries,)          S16
    slice_index                    (N_queries,)          int32
    window_index                   (N_queries,)          int32
    head_index                     (N_queries,)          int8
    window_boundary_distance       (N_lesion_queries,)   int8
    # --- Control: alpha & relative position bias ---
    alpha                          ()                    float32
    rel_pos_bias                   ((2W-1)^2, n_heads)   float32
    rel_pos_bias_entropy           (n_heads,)            float32
    # --- Tier 2: attention maps ---
    attention_full                 (N_probe, n_win, H, W^2, W^2)  float16   [chunked, blosc:lz4]
    logits_full                    (N_probe, n_win, H, W^2, W^2)  float16   [chunked, blosc:lz4]

/block_1_swmsa/
    ... same groups as block_0_wmsa ...

/inputs/
    image                          (N_probe, C, H_img, W_img)     float16
    mask                           (N_probe, H_img, W_img)        uint8
    subject_id                     (N_probe,)                     S16
    slice_index                    (N_probe,)                     int32
```

All tensors are stored per-query or per-token (no pre-aggregation).
Phase 5 handles aggregation.

### Training-time trajectory file (`probes/alpha_trajectory.jsonl`)

Append-only JSONL; one record per optimizer step:

```json
{"step": 1234, "epoch": 5, "block": 0, "alpha": 1.234, "p": 3.0, "fold": 0}
```

### Activation-patching file (`probes/patching_best_dice.h5`)

One file per (fold, source $p^*$, target $p$) pair at the best-dice
checkpoint. Structure:

```
/metadata/
    source_p                       float32
    target_p                       float32
    source_checkpoint              str
    target_checkpoint              str
    direction                      str        # "denoising" or "noising"

/block_0/
    variant_q/
        dice_patched               (N_probe,) float32
        pe                         (N_probe,) float32
        prediction                 (N_probe, H_img, W_img) uint8  [compressed]
    variant_k/
        ...
    variant_qk/
        ...
    variant_qhat_khat/
        ...
    variant_logits/
        ...
    variant_rel_pos_bias/          # optional
        ...
    # Unpatched references (replicated across variants for convenience)
    dice_source                    (N_probe,) float32
    dice_target                    (N_probe,) float32
/block_1/
    ...
```

### Public API (`src/lpqknorm/probes/`)

```python
# base.py
class Probe(Protocol):
    name: str
    def compute(self, capture: AttentionCapture,
                lesion_mask_tokens: Tensor,
                **kwargs: Any) -> ProbeResult: ...

@dataclass(frozen=True)
class ProbeResult:
    name: str
    per_token: Tensor | None
    per_query: Tensor | None
    per_block: Tensor | None          # NEW — e.g. spectral scalars, per-head LP metrics
    metadata: dict[str, Any]

# recorder.py
class ProbeRecorder:
    """Orchestrates hook registration, probe computation, and HDF5 writing."""
    def __init__(
        self,
        probes: Sequence[Probe],
        output_dir: Path,
        save_attention_maps: bool = True,   # NEW
        save_logits: bool = True,           # NEW
        save_rel_pos_bias: bool = True,     # NEW
    ) -> None: ...
    def run(self, model: nn.Module, dataloader: DataLoader,
            epoch_tag: str | int, device: torch.device) -> Path: ...

# patching.py  (NEW)
@dataclass(frozen=True)
class PatchingConfig:
    source_checkpoint: Path
    target_checkpoint: Path
    stage: int = 0
    blocks: Sequence[int] = (0, 1)
    variants: Sequence[str] = ("q", "k", "qk", "qhat_khat", "logits")
    direction: Literal["denoising", "noising"] = "denoising"

class ActivationPatcher:
    def __init__(self, cfg: PatchingConfig) -> None: ...
    def run(self, probe_loader: DataLoader, output_dir: Path) -> Path: ...

# attention_maps.py  (NEW)
def reconstruct_query_heatmap(
    attention: Tensor, query_idx: int, shift: int,
    grid_hw: tuple[int, int], window_size: int,
) -> Tensor: ...  # (H_tok, W_tok) map in token space

def attention_rollout(attentions: Sequence[Tensor]) -> Tensor: ...

def overlay_figure(
    image: Tensor, mask: Tensor, heatmap: Tensor,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure: ...

# linear_probe.py  (NEW)
@dataclass(frozen=True)
class LinearProbeMetrics:
    balanced_accuracy: float
    weight_sparsity: float
    margin: float

class LinearProbe(Probe):
    def __init__(self, n_splits: int = 5, lambda_grid: Sequence[float] = ...) -> None: ...

# spectral.py  (NEW)
class SpectralProbe(Probe):
    def __init__(self) -> None: ...
```

Each scalar probe is its own small module (~30–60 LOC):

```python
# peakiness.py          — Probe 1  (existing)
# entropy.py            — Probe 2  (existing)
# lesion_mass.py        — Probe 3  (existing)
# logit_gap.py          — Probe 4  (existing)
# attention_iou.py      — Probe 5  (existing)
# spatial_loc_error.py  — Probe 6  (NEW)
# linear_probe.py       — Probe 7  (NEW)
# spectral.py           — Probe 8  (NEW)
```

---

## Design notes

**Why store attention maps for every probe slice, not a curated subset.**
Simplicity dominates: a single storage policy across slices, probes,
checkpoints, and runs removes decision points and makes the Phase-5
analysis reproducible without reference to a curation manifest. The
budget (~40 GB for the full sweep) is comfortable.

**Why store per-token / per-query rather than aggregated statistics.** The
aggregation strategy (per-patient? per-lesion-size? conditional on
detection success?) is not committed at probe time. Raw tensors compress
well in HDF5 with blosc.

**Why HDF5 instead of parquet.** Per-token arrays are not tabular — they
have variable length per batch (depends on how many lesion tokens each
window contains), and they live alongside small scalars, per-head vectors,
and large attention tensors. HDF5 handles heterogeneous numeric groups
naturally; parquet does not.

**Why a fixed probe batch.** Comparing any probe across epochs and across
$p$ values is only meaningful if the inputs are identical. Otherwise an
epoch-over-epoch change could be "the model changed" or "this epoch
happened to see harder slices". This also allows paired bootstrap at
slice level in Phase 5.

**Why `float16` for attention and logits.** Attention $\in [0, 1]$ and
logits have dynamic range bounded by $\alpha \|q\|_p \|k\|_p + \|b^{\text{rel}}\|_\infty$,
which is $< 10$ in practice. `float16` has ~3 decimal digits of
precision — sufficient for downstream analysis. Test AT10 pins the
precision loss.

**Why Probe 7 (linear probe) instead of an SAE.** Argued at length in
§"Why not a Sparse Autoencoder". The summary: expansion factor and
activation pool are below SAE regime; the hypothesis being tested is
already binary (lesion / background) so a classifier is the correct tool.

**Handling no-lesion windows.** Not every window in a lesion-containing
slice contains lesion tokens. Per-query probes 2–7 are computed only for
windows containing at least one lesion token; Probes 3–6 only for
lesion queries within those windows. Probe 7 (linear probe) uses all
tokens including from non-lesion windows, because the classifier benefits
from the full background distribution; Probe 8 (spectral) likewise.
Token counts per stratum are exposed in the output metadata.

**Why fold activation patching into Phase 4 rather than into Phase 5.**
Patching requires model checkpoints loaded with hooks — the same
machinery as the scalar probes. Putting it in Phase 5 would duplicate the
hook infrastructure or couple Phase 5 to model code it otherwise does not
touch. It also reuses the fixed probe batch, which is Phase-4 territory.

---

## Implementation checklist

1. `probes/base.py` — `Probe` protocol, `ProbeResult` dataclass (extended
   with `per_block`), registry.
2. Probe modules:
   - `probes/peakiness.py`, `entropy.py`, `lesion_mass.py`, `logit_gap.py`,
     `attention_iou.py` — one probe each **(existing; no changes)**.
   - `probes/spatial_loc_error.py` — Probe 6 **(NEW)**.
   - `probes/linear_probe.py` — Probe 7 **(NEW)**. Depends on `scikit-learn`;
     add to `pyproject.toml`.
   - `probes/spectral.py` — Probe 8 **(NEW)**.
3. `probes/attention_maps.py` — reconstruction, rollout, overlay helpers
   **(NEW)**.
4. `probes/patching.py` — `PatchingConfig`, `ActivationPatcher`,
   `run_patching_sweep` **(NEW)**.
5. `probes/recorder.py` — extend to
   - accumulate attention and logit tensors per block,
   - call per-block probes (7, 8) after per-slice accumulation,
   - write the new HDF5 groups,
   - write relative-position-bias and its entropy.
6. `probes/tokenization.py` — existing; **extend** with
   `window_boundary_distance(flags, window_size)` for the boundary
   stratification control.
7. `training/callbacks.py::ProbeCallback` — invokes `ProbeRecorder` at the
   scheduled epochs and on best-checkpoint saves. Add `AlphaLogger`
   callback that writes `alpha_trajectory.jsonl` per step.
8. `training/callbacks.py::PatchingCallback` — at the end of training,
   invokes `ActivationPatcher` on the best-small-recall checkpoint pair
   for the current fold.
9. `cli/probe.py` — standalone post-hoc probe extraction given a
   checkpoint path.
10. `cli/patching.py` — standalone post-hoc activation patching given
    (source, target) checkpoint paths **(NEW)**.

---

## Acceptance tests

### 1. Probe correctness on synthetic attention (`test_probes_synthetic.py`)

Existing tests for Probes 1–5 (peakiness bounds, entropy bounds, lesion
mass range). Extended with:

```python
def test_spatial_loc_error_bounds():
    # Attention peaked on the lesion centroid → SLE = 0
    W = 7
    A = torch.zeros(1, W * W)
    A[0, 24] = 1.0  # centre of a 7x7 window
    lesion_mask = torch.zeros(W * W, dtype=torch.bool)
    lesion_mask[24] = True
    sle = SpatialLocalizationError().compute_per_query(A, lesion_mask, W)
    assert sle.item() == pytest.approx(0.0, abs=1e-5)

def test_participation_ratio_isotropic():
    # Isotropic features → PR = d_k
    X = torch.randn(10_000, 16)
    pr = SpectralProbe._participation_ratio(X)
    assert pr == pytest.approx(16.0, rel=0.1)

def test_participation_ratio_rank_one():
    # Rank-1 features → PR ≈ 1
    u = torch.randn(16)
    X = torch.randn(10_000, 1) * u
    pr = SpectralProbe._participation_ratio(X)
    assert pr == pytest.approx(1.0, abs=0.1)

def test_linear_probe_separable():
    # Perfectly separable features → balanced accuracy ≈ 1
    X_lesion = torch.randn(100, 16) + 3.0
    X_bg = torch.randn(100, 16) - 3.0
    metrics = LinearProbe().compute_value(X_lesion, X_bg)
    assert metrics.balanced_accuracy >= 0.95
```

### 2. Toy-model interior maximum (AT2, existing)

Run a synthetic forward pass where Q/K contain a known peaky lesion
token and a known diffuse background token, evaluate Probe 4 for
$p \in \{1.5, 2, 2.5, 3, 3.5, 4, 5\}$, assert the empirical argmax of
$\Delta(p)$ lies in $\{3, 4\}$.

### 3. Token-level lesion tagging (`test_tokenization.py`) — blocker

Existing test plus a verification that the token-index mapping is exact
for single-pixel masks at every pixel of a 16×16 grid (small but
exhaustive).

### 4. Window partition reconstruction

Existing test. Extended to verify the SW-MSA shift: a token with known
coordinate in a pre-shift image must end up in the predicted shifted
window.

### 5. End-to-end probe pipeline smoke test

Existing. Extended assertions:

```python
def test_probe_recorder_end_to_end(tmp_path):
    ...
    with h5py.File(out) as f:
        assert "block_0_wmsa" in f
        for key in [
            "peakiness_q", "entropy", "lesion_mass",
            "logit_gap", "attention_iou",
            "spatial_localization_error",
            "lp_balanced_accuracy", "lp_weight_sparsity",
            "pr_lesion", "pr_background",
            "eigenvalues_lesion", "eigenvalues_background",
            "rel_pos_bias", "rel_pos_bias_entropy",
            "attention_full", "logits_full", "alpha",
        ]:
            assert key in f["block_0_wmsa"], key
        assert "image" in f["inputs"]
        assert "mask" in f["inputs"]
```

### 6. Probe values invariant across identical runs

Two calls to `ProbeRecorder.run` on the same model weights, same fixed
loader, same device, produce bit-identical probe tensors (assert equality,
not allclose). Extended to Probes 6–8. For Probe 7 (which calls
`sklearn.linear_model.LogisticRegression`), seed the solver via
`random_state=0`.

### 7. No autograd retention

`ProbeRecorder.run` must be wrapped in `torch.inference_mode()`; test that
after running, `torch.cuda.memory_allocated()` has returned to the
pre-probe level within a small tolerance.

### 8. Attention-map reconstruction round trip (`test_attention_maps.py`, NEW)

```python
def test_reconstruct_query_heatmap_identity():
    """Identity attention (each query attends only to itself) must
    reconstruct to a single-peak heatmap at the query's own coordinate."""
    W = 7
    n_win = 4  # 2x2 grid of windows
    A = torch.eye(W * W).unsqueeze(0).unsqueeze(0).expand(n_win, 1, W*W, W*W)
    heatmap = reconstruct_query_heatmap(
        attention=A, query_idx=5, shift=0,
        grid_hw=(2 * W, 2 * W), window_size=W,
    )
    assert heatmap.shape == (2 * W, 2 * W)
    assert heatmap.sum().item() == pytest.approx(1.0, abs=1e-5)
    # Non-zero at exactly one location
    assert (heatmap > 0).sum().item() == 1

def test_attention_rollout_residual_form():
    # Two layers, identity attention → rollout = 0.5*(I+I)*0.5*(I+I) = I
    A = [torch.eye(10), torch.eye(10)]
    R = attention_rollout(A)
    torch.testing.assert_close(R, torch.eye(10), atol=1e-6, rtol=0.0)
```

### 9. Activation patching correctness (`test_patching.py`, NEW)

```python
def test_self_patching_is_identity():
    """Patching a model with its own captures must leave Dice unchanged."""
    model = build_swin_unetr_lp(..., lp_cfg=LpQKNormConfig(p=3.0))
    ckpt = save_checkpoint(model, tmp_path / "self.ckpt")
    cfg = PatchingConfig(
        source_checkpoint=ckpt, target_checkpoint=ckpt,
        variants=("q", "k", "qk"),
    )
    patcher = ActivationPatcher(cfg)
    out = patcher.run(loader, tmp_path)
    with h5py.File(out) as f:
        for variant in ("q", "k", "qk"):
            torch.testing.assert_close(
                torch.tensor(f[f"block_0/variant_{variant}/dice_patched"][()]),
                torch.tensor(f[f"block_0/dice_target"][()]),
                atol=1e-5, rtol=0.0,
            )

def test_patching_changes_output_when_checkpoints_differ():
    """Patching from a genuinely different checkpoint must change Dice."""
    ...  # two different random seeds → different weights → non-zero delta
```

### 10. Float16 precision for attention storage

Round-trip test: a `float32` attention row softmaxed to $[0,1]$, stored as
`float16`, reloaded, must still sum to 1 within $10^{-3}$ and reproduce
Probe 2 (entropy) within $10^{-3}$ relative error. If not, `float32`
storage is mandated.

### 11. Alpha trajectory append-only

Test that `AlphaLogger` produces a strictly monotonically increasing
`step` column and one line per step.

---

## Expected runtime

Per checkpoint, on the local RTX 4060, 32 slices:

| Component                             | Time     | Notes                                       |
|---------------------------------------|----------|---------------------------------------------|
| Forward pass + hook capture           | ~3 s     |                                             |
| Scalar probes 1–6                     | ~0.5 s   | vectorised                                  |
| Linear probe (Probe 7)                | ~5 s     | per-head L1-logistic with 5-fold CV         |
| Spectral probe (Probe 8)              | ~0.1 s   | SVD of 16-dim covariance                    |
| Attention-map HDF5 write              | ~2 s     | blosc:lz4                                   |
| **Total per checkpoint**              | **~10 s**|                                             |
| Activation patching (per fold, once)  | ~60 s    | 5 variants × 2 blocks × 32 slices × 2 fwd   |

Across 9 checkpoints × 18 runs: total probe cost ~30 min; patching ~18 min.

---

## References

### Probing methodology

- Alain & Bengio. *Understanding Intermediate Layers Using Linear
  Classifier Probes*. ICLR 2017 workshop. arXiv:1610.01644.
- Vig. *A Multiscale Visualization of Attention in the Transformer Model*.
  ACL 2019.
- Michel, Levy, Neubig. *Are Sixteen Heads Really Better than One?*
  NeurIPS 2019.
- Clark, Khandelwal, Levy, Manning. *What Does BERT Look At?*
  BlackboxNLP 2019. arXiv:1906.04341.
- Rogers, Kovaleva, Rumshisky. *A Primer in BERTology*. TACL 2020.
  doi:10.1162/tacl_a_00349.

### Attention-map analysis

- Abnar & Zuidema. *Quantifying Attention Flow in Transformers*. ACL 2020.
  arXiv:2005.00928.
- Chefer, Gur, Wolf. *Transformer Interpretability Beyond Attention
  Visualization*. CVPR 2021. arXiv:2012.09838.

### Causal interventions

- Meng, Bau, Andonian, Belinkov. *Locating and Editing Factual
  Associations in GPT* (ROME). NeurIPS 2022. arXiv:2202.05262.
- Heimersheim & Nanda. *How to use and interpret activation patching*.
  arXiv:2404.15255.
- Vig et al. *Investigating Gender Bias in Language Models Using Causal
  Mediation Analysis*. NeurIPS 2020.

### Dictionary learning (deferred to appendix)

- Bricken et al. *Towards Monosemanticity: Decomposing Language Models
  with Dictionary Learning*. Anthropic 2023.
- Cunningham, Ewart, Riggs, Huben, Sharkey. *Sparse Autoencoders Find
  Highly Interpretable Features in Language Models*. ICLR 2024.
  arXiv:2309.08600.
- Braun et al. *Identifying Functionally Important Features with
  End-to-End Sparse Dictionary Learning*. arXiv:2405.12241.
- Templeton et al. *Scaling Monosemanticity: Extracting Interpretable
  Features from Claude 3 Sonnet*. Anthropic 2024.

### Spectral / population geometry

- Gao, Trautmann, Yu, Santhanam, Ryu, Shenoy, Ganguli. *A theory of
  multineuronal dimensionality, dynamics and measurement*. bioRxiv 2017.
  doi:10.1101/214262.
- Kornblith, Norouzi, Lee, Hinton. *Similarity of Neural Network
  Representations Revisited*. ICML 2019. arXiv:1905.00414.

### Architecture

- Liu et al. *Swin Transformer*. ICCV 2021. arXiv:2103.14030.
- Cao et al. *Swin-Unet*. ECCVW 2022. arXiv:2105.05537.
- Hatamizadeh et al. *Swin UNETR*. BrainLes 2021. arXiv:2201.01266.
- Henry et al. *Query-Key Normalization for Transformers*. EMNLP 2020.
  arXiv:2010.04245.
- López-Rubio et al. *Enhanced QKNorm with the Lp Norm*. 2026.
  arXiv:2602.05006.

### Tooling (optional / appendix)

- Bloom, Chanin, Tigges. *SAELens*. 2024.
  <https://github.com/jbloomAus/SAELens>.
- Marks et al. *dictionary_learning*. 2024.
  <https://github.com/saprmarks/dictionary_learning>.
