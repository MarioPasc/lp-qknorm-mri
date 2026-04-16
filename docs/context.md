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
I'll read your paper first to understand the specific claim about the Lp norm for QK attention and the "small tumor hypothesis" before giving a critical assessment.I now have the complete picture: the López-Rubio et al. paper, your toy-model prediction $\Delta(p) = s^{1-2/p}\bigl[1 - (s/d_k)^{1/p}\bigr]$, the mechanistic chain, and the context of the small-lesion hypothesis. Let me now give you an honest, critical assessment.

---

# Critical assessment of the "small-tumor ⇒ $p>2$" intuition

## 1. Re-derivation of your toy model, with the hidden assumptions made explicit

Your claim rests on a single inequality: for a *peaky* (approximately $s$-sparse) lesion representation and a *diffuse* background, the post-normalization logit gap

$$
\Delta(p) \;=\; \langle \hat q^{(p)}, \hat k_L^{(p)}\rangle \;-\; \langle \hat q^{(p)}, \hat k_B^{(p)}\rangle
$$

has an interior maximum at some $p^\star>2$. Let me reconstruct this carefully, because the rigor of the intuition depends entirely on what the model assumes.

**Setup.** Let $q, k_L \in \mathbb{R}^{d_k}$ share the same support of size $s \ll d_k$, with unit magnitude on that support and zero elsewhere:

$$
q_h = k_{L,h} = \mathbb{1}[h \le s], \qquad k_{B,h} = \tfrac{1}{\sqrt{d_k}}\;\; \forall h.
$$

Then

$$
\|q\|_p = \|k_L\|_p = s^{1/p}, \qquad \|k_B\|_p = d_k^{1/p}\cdot d_k^{-1/2} = d_k^{1/p - 1/2}.
$$

After $\ell_p$-normalization,

$$
\hat q^{(p)}_h = \hat k_{L,h}^{(p)} = s^{-1/p}\,\mathbb{1}[h \le s],\qquad 
\hat k_{B,h}^{(p)} = d_k^{-1/p + 1/2}\cdot d_k^{-1/2} = d_k^{-1/p}.
$$

The two inner products are therefore

$$
\langle \hat q^{(p)}, \hat k_L^{(p)}\rangle = s \cdot s^{-2/p} = s^{\,1 - 2/p},
$$

$$
\langle \hat q^{(p)}, \hat k_B^{(p)}\rangle = s \cdot s^{-1/p}\cdot d_k^{-1/p} = s^{\,1 - 1/p}\, d_k^{-1/p} = s^{\,1-2/p}\cdot (s/d_k)^{1/p}.
$$

Subtracting gives *exactly* the formula in your docs:

$$
\boxed{\;\Delta(p) \;=\; s^{\,1-2/p}\Bigl[1 - (s/d_k)^{1/p}\Bigr]\;}
$$

Taking logs, $\log \Delta(p) = (1-2/p)\log s + \log\bigl[1-(s/d_k)^{1/p}\bigr]$. Differentiating and setting to zero yields the stationarity condition

$$
\frac{2\log s}{p^2} \;=\; \frac{(s/d_k)^{1/p}\,\log(d_k/s)}{p^2\bigl[1-(s/d_k)^{1/p}\bigr]},
$$

i.e.

$$
2\log s\cdot\bigl[1-(s/d_k)^{1/p}\bigr] \;=\; (s/d_k)^{1/p}\,\log(d_k/s).
$$

Let $u = (s/d_k)^{1/p}\in(0,1)$. Then $p^\star$ is defined implicitly by

$$
u^\star = \frac{2\log s}{2\log s + \log(d_k/s)} = \frac{2\log s}{\log(d_k\,s)}.
$$

For $s=4, d_k=64$: $u^\star = \frac{2\log 4}{\log 256} = \frac{2\cdot 2}{4}\cdot \frac{\log 2}{\log 2} = 1.0$ — which is the degenerate boundary, i.e. the continuous argmax sits at infinity for this particular $(s,d_k)$. In practice the discrete sweep gives a finite argmax inside $\{3,4\}$ because $\Delta(p)$ is very flat beyond $p\approx 3$. **This already tells you that the "interior maximum at $p^\star\in(2,4)$" statement is sensitive to $(s,d_k)$ and is not structurally guaranteed** — for some regimes $\Delta(p)$ is monotone increasing on $[1,\infty)$, and the "optimum" is only an argmax of a *flat* tail.

## 2. Where the intuition is sound

The mathematically defensible core of your claim is the following, and only the following:

**Proposition (honest version).** *If* the learned query/key representations of lesion tokens are approximately $s$-sparse with $s \ll d_k$, and background tokens are approximately uniform on the sphere, *then* $\ell_p$-normalization with $p>2$ produces a strictly larger cosine gap between lesion-lesion and lesion-background token pairs than $\ell_2$-normalization, up to some regime-dependent $p^\star$. Consequently, after softmax, the attention mass on lesion keys from lesion queries is larger.

This is correct. The mechanism is essentially the well-known fact that $\ell_p$ norms with $p>2$ interpolate between $\ell_2$ and $\ell_\infty$, and $\ell_\infty$-normalization makes sparse vectors look "more parallel" to themselves than to dense distractors. The connection to top-$k$ / sparse attention literature is direct: Martins & Astudillo (2016), *From Softmax to Sparsemax*, ICML, arXiv:1602.02068; Correia, Niculae & Martins (2019), *Adaptively Sparse Transformers*, EMNLP, arXiv:1909.00015. Both show that sharpening the attention kernel helps when the signal is concentrated — which is precisely the small-lesion regime.

## 3. Where the intuition is weak — four serious objections

### 3.1. The sparsity premise is an assumption, not a consequence

Your entire chain peakiness $\uparrow$ $\Rightarrow$ entropy $\downarrow$ $\Rightarrow$ lesion mass $\uparrow$ $\Rightarrow$ recall $\uparrow$ starts by assuming that lesion tokens *develop* peaky $(q,k)$ representations. There is no theorem that guarantees this. In fact the opposite is plausible: small lesions are represented by a small number of tokens, each of which must encode a rich feature set (shape, boundary, contrast, anatomical context). Whether those features get packed into a few coordinates or spread across the whole head dimension is an empirical question about the loss geometry and optimizer, not a consequence of choosing $p>2$. Swin's MLP ratios and LayerNorm arguably *discourage* sparse features.

**Literature evidence runs both ways.** On one hand, Elhage et al. (2022), *Toy Models of Superposition* (Anthropic, transformer-circuits.pub/2022/toy_model), document that transformer representations *can* become sparse under certain loss regimes — but they also show superposition, i.e. many features share coordinates non-sparsely. On the other hand, Timkey & van Schijndel (2021), *All Bark and No Bite: Rogue Dimensions in Transformer LMs*, EMNLP, arXiv:2109.04404, show that a *small* number of dimensions dominate similarity scores in practice — which is your friend, but it implies the peakiness is not lesion-specific, it is global. That matters for your Probe 1: if *every* token is peaky on the same rogue dimensions, the lesion/background gap in peakiness may be negligible even though the overall attention sharpens.

### 3.2. Sharpening the softmax kernel is not uniquely tied to small objects

The rationale "higher $p$ $\Rightarrow$ sharper attention $\Rightarrow$ better for small lesions" conflates two effects:

- **Effect A (geometric).** $\ell_p$-normalization changes the cosine landscape.
- **Effect B (temperature).** Post-softmax sharpness depends on the *magnitude* of logits, which is controlled by the learnable $\alpha$ in QKNorm.

Because $\alpha$ is learned, effect B is largely absorbed: if $p=2$ and the model wanted sharper attention, it would raise $\alpha$. The *only* residual benefit of $p>2$ is the geometric reshaping in effect A — which, per §3.1, requires sparse lesion features. If sparsity does not develop, $\alpha$ will compensate and the $p$-sweep will collapse to a near-flat curve. The López-Rubio et al. paper you extend *already shows* the curve is flat for $p\in\{2.5, 3, 3.5, 4\}$ on Tiny Shakespeare — the spread is $\sim 0.016$ nats across those four values. That is the signature of an effect that is real but small, and highly sensitive to the task.

### 3.3. Small lesions might benefit from the *opposite* direction

A separate and competing hypothesis is that small objects need *more* exploratory, *less* peaky attention, because a single lesion token needs to aggregate context from many surrounding non-lesion tokens to disambiguate (partial volume, boundary effects, speckle). This is the argument implicit in entmax-$\alpha$ work (Peters, Niculae & Martins, 2019, *Sparse Sequence-to-Sequence Models*, ACL, arXiv:1905.05702), where $\alpha < 2$ (softer than softmax) sometimes wins for low-resource regimes. For segmentation specifically, Wang et al. (2022), *UCTransNet* (AAAI, arXiv:2109.04335) find that broader channel-wise attention helps small-structure segmentation. **Your intuition therefore has a credible null hypothesis pointing in the opposite direction.** The experiment may well find $p^\star < 2$ or a flat profile.

### 3.4. Relative position bias interacts with $\alpha$ in Swin windows

Swin-UNETR adds a learned relative position bias $B\in\mathbb{R}^{W^2\times W^2}$ to the logits *after* the QK product. So the final logit is

$$
s^{(p)}_{ij} = \alpha\,\langle\hat q^{(p)}_i,\hat k^{(p)}_j\rangle + B_{\pi(i),\pi(j)}.
$$

Because $\langle\hat q^{(p)},\hat k^{(p)}\rangle \in[-1,1]$ regardless of $p$, the *ratio* of QK contribution to position-bias contribution is governed entirely by $\alpha$ relative to typical $|B|$. If $\alpha$ is small — which is plausible in small $7\times 7$ or $8\times 8$ windows where position already carries most of the signal — the $p$-dependence of the QK cosine is being added to a much larger position-bias term, and the downstream effect on attention is diluted. This is not fatal, but it is a confound your Probe 4 should measure: report $\alpha/\|B\|_\infty$ per block; if that ratio is $\ll 1$, the whole $\ell_p$ machinery is operating in a regime where it cannot move attention much.

## 4. A more defensible formulation of the hypothesis

Rather than "higher $p$ helps small lesions," I would recommend reframing as a **conditional** claim, because that is what the math actually supports:

> **H1 (geometric).** If lesion tokens develop higher feature peakiness $\rho = \|\cdot\|_\infty/\|\cdot\|_2$ than background tokens, then increasing $p$ beyond 2 monotonically increases the logit gap $\Delta_i$ until a point $p^\star(\rho, d_k)$ determined by $u^\star(\rho)$.
>
> **H2 (training).** Training under $\ell_p$ with $p>2$ encourages lesion tokens to develop higher peakiness than under $p=2$ (testable via Probe 1, *before* appealing to any downstream metric).
>
> **H3 (downstream).** H1 $\land$ H2 $\Rightarrow$ higher lesion attention mass and small-lesion recall.

The published paper you extend (López-Rubio et al., arXiv:2602.05006) supports *neither* H1 nor H2 directly — it only shows that validation loss on Tiny Shakespeare improves slightly for $p>2$, without probing *why*. Your Probes 1–5 are what make this study novel relative to their work; without them you would be doing a trivial hyperparameter sweep. With them, you can actually test the mechanistic chain.

## 5. Bottom line

Your intuition is **partially solid, partially wishful**. Specifically:

- The math of the logit gap is correct, *under the sparsity premise*, and the derivation gives a legitimate toy-model argument.
- The generalization from "one sparse lesion token versus one diffuse background token" to "small lesions in a real segmentation network" is a substantial leap. It depends on (i) lesion features actually becoming sparse, (ii) $\alpha$ not absorbing the effect, (iii) relative position bias not dominating, (iv) the effect surviving the hierarchical decoder. None of these are guaranteed.
- The competing hypothesis — that small objects want *softer* attention for contextual aggregation — is also live and has literature support.
- The López-Rubio et al. result is consistent with your hypothesis but does not pre-establish it. Their effect size is small (1.4% relative in validation loss), and the flatness of their curve beyond $p=2.5$ suggests the mechanism is weak in the LM setting.

The study is well-designed **precisely because the probes adjudicate the mechanism**. I would strengthen the paper by (a) stating H1/H2/H3 explicitly and declaring in advance that finding H2 false while H1 is true is a valid negative result, (b) reporting the learned $\alpha$ per-block per-$p$ as a direct test of §3.2, and (c) including the $\alpha/\|B\|_\infty$ ratio from §3.4 as a diagnostic. Without those, a null result on recall will be ambiguous — you will not know whether the mechanism failed to trigger or whether it triggered but was confounded.

---

**Key references**

- Henry et al. (2020). *Query-Key Normalization for Transformers*. EMNLP Findings. arXiv:2010.04245.
- López-Rubio, Montes-Pérez, Palomo (2026). *Enhanced QKNorm with the Lp Norm*. arXiv:2602.05006.
- Martins & Astudillo (2016). *From Softmax to Sparsemax*. ICML. arXiv:1602.02068.
- Peters, Niculae & Martins (2019). *Sparse Sequence-to-Sequence Models*. ACL. arXiv:1905.05702.
- Correia, Niculae & Martins (2019). *Adaptively Sparse Transformers*. EMNLP. arXiv:1909.00015.
- Timkey & van Schijndel (2021). *All Bark and No Bite: Rogue Dimensions in Transformer LMs*. EMNLP. arXiv:2109.04404.
- Elhage et al. (2022). *Toy Models of Superposition*. Anthropic. transformer-circuits.pub/2022/toy_model.
- Wang et al. (2022). *UCTransNet*. AAAI. arXiv:2109.04335.