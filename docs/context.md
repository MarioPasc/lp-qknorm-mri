## Model recommendation: Swin-Unet (not TransUNet), stage 1 probed

The tempting default for "simple + canonical medical transformer" is TransUNet (*Chen et al., arXiv:2102.04306, 2021*), but it is the wrong fit for FCD specifically. TransUNet feeds a 224×224 image through a ResNet-50 backbone that downsamples 32× before the ViT bottleneck, yielding a 14×14 token grid. An FCD Type II lesion is ~1–3 cm; at ~1 mm/pixel that is 10–30 pixels, which collapses to **<1 token** at the bottleneck. You cannot mechanistically measure attention concentration on a lesion that occupies a single token — there is nothing to concentrate.

**Swin-Unet** (*Cao et al., ECCVW 2022, arXiv:2105.05537*) is the right choice:

- Patch size 4×4, four hierarchical stages. Stage 1 runs attention on a **56×56** grid, so FCD covers ~4–16 tokens inside 7×7 windows. This is the resolution at which the Lp-QKNorm mechanism from the previous discussion can actually be observed.
- Windowed self-attention is local (W-MSA, 7×7 windows), which matches the small-lesion inductive bias: within a window, 1–4 lesion tokens must out-compete 45–48 background tokens — the exact "peaky-key vs. diffuse-background" regime where $p > 2$ has its maximum effect.
- It is now the de facto 2D medical-imaging transformer baseline alongside TransUNet; reviewers will not push back on architecture choice.
- Unlike SegFormer or nnFormer, it is straightforward to instrument: the Q–K projections are well-isolated inside `WindowAttention`.

Keep the architecture stock; change only the Q–K normalization inside `WindowAttention.forward`, and apply the change globally for simplicity. **Restrict mechanistic probes to stage 1** — two attention blocks at 56×56, enough tokens to get statistics, few enough modules to analyze cleanly.

## Dataset

Primary: **MELD Project** (*Spitzer et al., Brain 2022, doi:10.1093/brain/awac224*). ~600 subjects, multi-centre, publicly released with a DUA. It is the FCD benchmark. Extract axial T1w slices intersecting the lesion mask.

Fallback if MELD access is not in place this week: **ATLAS v2.0** (*Liew et al., Scientific Data 2022, doi:10.1038/s41597-022-01401-7*) — 1,271 subjects, T1w, stroke lesions with a long small-lesion tail. Publicly downloadable, no DUA delay. Stratify evaluation by lesion volume; use only the smallest quartile for the headline comparison. The mechanistic claim transfers because the story is about small lesions generally, not FCD specifically — which you can flag in the paper as a deliberate generalization test.

## The single experiment

One axis, one headline metric, one mechanistic claim.

**Design.** Swin-Unet trained on 2D slices containing lesion voxels, binary segmentation, Dice + BCE loss, AdamW, fixed everything except the Q–K normalization inside stage-1 (and for completeness, all stages) window attention. Sweep

$$
p \in \{2.0, 2.5, 3.0, 3.5, 4.0\}
$$

with 3-fold patient-level cross-validation (not slice-level — data leakage risk). The $p=2$ cell is the original QKNorm baseline. A **vanilla softmax** condition (no QKNorm, stock MONAI `WindowAttention`) is included as a lower-bound control at minimal cost (+3 runs). The primary comparison is $p^\star$ vs $p=2$ (isolating the Lp generalisation), but the vanilla column answers the prerequisite question "Does QKNorm itself help?" — if it does not, the Lp sweep is moot.

**Headline metric.** Lesion-wise recall at fixed FP rate, restricted to the *small-lesion* stratum (lesion volume below the 33rd percentile in your cohort). Dice on this stratum as secondary. Reporting Dice over the full cohort will hide the effect — the Lp advantage is small-lesion-specific by hypothesis.

**Statistical test.** Paired bootstrap over patients, per-patient small-lesion recall, $p=3$ (or argmax) vs. $p=2$. One p-value, reported cleanly.

## Mechanistic probes (all via forward hooks on stage-1 attention)

Install a single hook on `WindowAttention` in stage 1 that captures $Q$, $K$, the normalized $\hat Q^{(p)}, \hat K^{(p)}$, and the post-softmax attention matrix $A$. On a held-out validation batch, tag each token as lesion or background using the ground-truth mask downsampled to 56×56. Then compute:

**Probe 1 — feature peakiness of queries/keys.** For each token $i$,

$$
\rho_p(v_i) \;=\; \frac{\|v_i\|_\infty}{\|v_i\|_2}, \qquad v_i \in \{q_i, k_i\}.
$$

Compare $\mathbb{E}[\rho_p \mid \text{lesion}]$ vs. $\mathbb{E}[\rho_p \mid \text{background}]$ across $p$. The hypothesis from the previous message predicts lesion tokens develop higher peakiness, and that the gap widens with $p$ because training under $\ell_p$-norm encourages this geometry.

**Probe 2 — attention entropy.** For each query token $i$ inside a window of size $W$,

$$
H_i \;=\; -\sum_{j=1}^{W} A_{ij}\log A_{ij}, \qquad H_i \in [0, \log W].
$$

Report $\mathbb{E}[H_i \mid i \in \text{lesion}]$ vs. $p$. Prediction: monotonic decrease up to $p \approx 3{-}4$ (sharper attention on lesion queries), then flattening — the interior-maximum behavior derived earlier.

**Probe 3 — lesion attention mass (the money plot).** For each lesion query $i$,

$$
M_i \;=\; \sum_{j \in \mathcal{L}} A_{ij}, \qquad \mathcal{L} = \{\text{lesion tokens in window of } i\}.
$$

Plot $\mathbb{E}[M_i]$ vs. $p$. This is the direct operational statement of "higher $p$ attends better to small lesions." If this curve is flat, the hypothesis is wrong regardless of what segmentation metrics say.

**Probe 4 — logit gap.** For lesion queries,

$$
\Delta_i \;=\; \max_{j\in\mathcal{L}} s_{ij} \;-\; \operatorname{median}_{j\notin\mathcal{L}} s_{ij},
$$

averaged. This is the empirical analogue of the $\Delta(p)$ derived in the toy model and lets you verify the SNR argument, not just its downstream effect.

**Probe 5 — attention-mask IoU.** Binarize $A_{i\cdot}$ at a threshold (e.g., top-$k$ tokens with $k = |\mathcal{L}|$) and compute IoU with the ground-truth lesion mask over the window. Average across lesion queries. Most directly interpretable figure for a reviewer.

These five probes, each a single scalar as a function of $p$, give a tight mechanistic narrative: *peakier features (1) → lower entropy (2) → more mass on lesion (3) → larger logit gap (4) → better spatial alignment (5) → better small-lesion recall (headline)*. That is a chain of evidence, not a correlation.

## Week schedule

- **Day 1.** Data pipeline: slice extraction, patient-level splits, small-lesion stratum definition, sanity-check a Swin-Unet forward pass. Freeze the preprocessing.
- **Day 2.** Implement `LpWindowAttention` — a drop-in replacement, ~30 lines. Verify numerically that $p{=}2$ recovers QKNorm exactly. Train one full run to confirm convergence.
- **Day 3.** Launch the 6 × 3 = 18 training runs on Picasso (A100, Singularity): vanilla + 5 Lp conditions × 3 folds. With 2D Swin-Unet these should be $\leq 1$ h each; the sweep fits in one overnight SLURM array.
- **Day 4.** Segmentation metrics, paired bootstrap, stratified plots. Decide the winning $p^\star$.
- **Day 5.** Mechanistic probes on the held-out fold, five figures.
- **Day 6.** Draft (intro, method, results, discussion). The Lp-QKNorm math is already in your PI's paper — cite it, don't re-derive.
- **Day 7.** Revision pass, arXiv.

## Risks

1. **The effect is real but not statistically significant at small $n$.** Three folds is thin. Mitigation: report per-patient effect sizes, not just means; use paired bootstrap which is valid at small $n$.
2. **MELD access delay.** Start the ATLAS v2.0 pipeline in parallel on Day 1. Publish whichever is ready.
3. **Stage-1 attention turns out to be dominated by positional biases.** Swin-Unet uses relative position bias. If probes are swamped by it, redo Probes 2–5 on the de-biased attention $\tilde A = \operatorname{softmax}(S^{(p)})$ without the bias term.
4. **$p^\star$ is not stable across folds.** This would undermine the headline. Mitigation: report the full $p$-curve, not an argmax, and argue the mechanism (entropy/mass probes) in a $p$-monotone form — consistent with the toy-model derivation which predicts a smooth interior maximum, not a sharp peak.

**Key references to cite:**

- Cao et al. *Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation*. ECCVW 2022. arXiv:2105.05537.
- Liu et al. *Swin Transformer*. ICCV 2021. arXiv:2103.14030.
- Spitzer et al. *Interpretable surface-based detection of focal cortical dysplasias: a MELD study*. Brain 2022. doi:10.1093/brain/awac224.
- Liew et al. *A large, curated, open-source stroke neuroimaging dataset (ATLAS v2.0)*. Scientific Data 2022. doi:10.1038/s41597-022-01401-7.
- Henry et al. *Query-Key Normalization for Transformers*. Findings of EMNLP 2020. arXiv:2010.04245.