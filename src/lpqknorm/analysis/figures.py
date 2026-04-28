"""Phase 5 figure functions.

All functions take long-format dataframes in, write a single file
(format inferred from the suffix of *out*), and return the resolved
:class:`pathlib.Path`.  Internally they apply the publication style
once via :func:`set_publication_style` and pass deterministic
``metadata=`` kwargs to ``savefig`` so that two consecutive runs on
the same data produce identical PDFs (Phase 5 acceptance test §6).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from lpqknorm.analysis.bootstrap import AnalysisError
from lpqknorm.analysis.style import (
    COL_WIDTH_IN,
    DOUBLE_COL_WIDTH_IN,
    PDF_METADATA,
    color_for_p,
    set_publication_style,
)


if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd
    from matplotlib.figure import Figure


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _savefig(fig: Figure, out: Path | str) -> Path:
    out_p = Path(out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    if out_p.suffix.lower() == ".pdf":
        fig.savefig(out_p, metadata=PDF_METADATA)
    else:
        fig.savefig(out_p)
    plt.close(fig)
    return out_p


def _toy_delta(p: np.ndarray, s: int, d_k: int) -> np.ndarray:
    """``Δ(p, s, d_k) = s^{1−2/p} · [1 − (s/d_k)^{1/p}]``."""
    p = np.asarray(p, dtype=float)
    return np.power(float(s), 1.0 - 2.0 / p) * (
        1.0 - np.power(float(s) / float(d_k), 1.0 / p)
    )


# ---------------------------------------------------------------------------
# Figure 1 — toy-model prediction (no experimental data needed)
# ---------------------------------------------------------------------------


def fig_toy_model_prediction(
    sparsity_levels: Sequence[int],
    d_k: int,
    out: Path | str,
    *,
    p_range: tuple[float, float] = (1.5, 8.0),
    sweep_band: tuple[float, float] | None = (2.0, 4.0),
    n_points: int = 401,
) -> Path:
    """Plot ``Δ(p) = s^{1−2/p}[1−(s/d_k)^{1/p}]`` for several ``s``.

    Parameters
    ----------
    sparsity_levels : sequence of int
        Sparsity levels ``s`` to overlay (e.g. ``[2, 4, 8, 16]``).
    d_k : int
        Per-head dimension (e.g. 24 for the configured Swin-UNETR).
    out : Path or str
        Output file (``.pdf`` for byte-deterministic preprint output;
        ``.png`` for previews).
    p_range : (float, float), default ``(1.5, 8.0)``
        Continuous range of ``p`` to plot.
    sweep_band : (float, float) or None
        Optional shaded band marking the empirical sweep range.
    n_points : int, default 401
        Number of ``p`` samples in the line.
    """
    if d_k <= 1:
        raise AnalysisError(f"d_k must be > 1, got {d_k}")
    if any(s <= 0 or s >= d_k for s in sparsity_levels):
        raise AnalysisError(
            f"sparsity levels must satisfy 0 < s < d_k={d_k}, got {list(sparsity_levels)}"
        )
    if p_range[0] <= 1.0 or p_range[1] <= p_range[0]:
        raise AnalysisError(f"invalid p_range: {p_range}")

    set_publication_style()
    fig, ax = plt.subplots(figsize=(COL_WIDTH_IN, COL_WIDTH_IN * 0.75))
    p_grid = np.linspace(p_range[0], p_range[1], n_points)
    for s in sparsity_levels:
        delta = _toy_delta(p_grid, s, d_k)
        ax.plot(p_grid, delta, label=f"s = {s}")

    if sweep_band is not None:
        ax.axvspan(*sweep_band, color="0.85", alpha=0.6, lw=0, zorder=0)

    ax.axvline(2.0, color="black", linestyle="--", linewidth=0.7, zorder=1)
    ax.set_xlabel(r"Lp norm exponent $p$")
    ax.set_ylabel(r"Logit gap $\Delta(p)$")
    ax.set_title(rf"Toy-model prediction ($d_k = {d_k}$)")
    ax.legend(frameon=False, loc="best")
    return _savefig(fig, out)


# ---------------------------------------------------------------------------
# Figures 2–6 — empirical
# ---------------------------------------------------------------------------


def _check_per_patient(per_patient: pd.DataFrame) -> None:
    needed = {"p", "p_label", "fold", "subject_id", "volume_stratum"}
    missing = needed - set(per_patient.columns)
    if missing:
        raise AnalysisError(f"per_patient missing columns: {sorted(missing)}")


def fig_stratified_dice(per_patient: pd.DataFrame, out: Path | str) -> Path:
    """Mean Dice per ``(stratum, p)`` with std error bars."""
    _check_per_patient(per_patient)
    if "dice" not in per_patient.columns:
        raise AnalysisError("per_patient lacks 'dice' column")
    set_publication_style()

    grp = (
        per_patient.groupby(["volume_stratum", "p_label"], dropna=False)["dice"]
        .agg(["mean", "std"])
        .reset_index()
    )
    strata_order = [
        s for s in ("small", "medium", "large") if s in grp["volume_stratum"].unique()
    ]
    p_labels = sorted(
        grp["p_label"].unique(),
        key=lambda x: (x == "vanilla", _safe_float(x)),
    )
    fig, ax = plt.subplots(figsize=(DOUBLE_COL_WIDTH_IN, COL_WIDTH_IN * 0.9))
    bar_w = 0.8 / max(len(p_labels), 1)
    for j, p in enumerate(p_labels):
        sub = grp[grp["p_label"] == p].set_index("volume_stratum").reindex(strata_order)
        x = np.arange(len(strata_order)) + (j - (len(p_labels) - 1) / 2) * bar_w
        ax.bar(
            x,
            sub["mean"].to_numpy(),
            width=bar_w,
            yerr=sub["std"].to_numpy(),
            capsize=2,
            label=f"p={p}",
            color=color_for_p(str(p)),
            edgecolor="black",
            linewidth=0.4,
        )
    ax.set_xticks(np.arange(len(strata_order)))
    ax.set_xticklabels(strata_order)
    ax.set_ylabel("Dice")
    ax.set_xlabel("Volume stratum")
    ax.set_title("Stratified Dice by p")
    ax.legend(
        ncol=min(len(p_labels), 6),
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.32),
    )
    fig.subplots_adjust(bottom=0.3)
    return _savefig(fig, out)


def fig_small_recall_vs_p(per_patient: pd.DataFrame, out: Path | str) -> Path:
    """Mean lesion-recall on the small stratum vs ``p``, with error bars.

    A horizontal dashed line marks the vanilla (no-QKNorm) baseline if
    present in the data.
    """
    _check_per_patient(per_patient)
    if "lesion_recall" not in per_patient.columns:
        raise AnalysisError("per_patient lacks 'lesion_recall' column")
    set_publication_style()

    small = per_patient[per_patient["volume_stratum"] == "small"]
    if small.empty:
        raise AnalysisError("no small-stratum patients in per_patient")

    # Per-fold mean, then mean ± std across folds.
    fold_means = (
        small.groupby(["p_label", "p", "fold"])["lesion_recall"].mean().reset_index()
    )
    summary = (
        fold_means.groupby(["p_label", "p"], dropna=False)["lesion_recall"]
        .agg(["mean", "std"])
        .reset_index()
    )

    vanilla = summary[summary["p_label"] == "vanilla"]
    sweep = summary[summary["p_label"] != "vanilla"].copy()
    if not sweep.empty:
        sweep["_p"] = sweep["p_label"].astype(float)
        sweep = sweep.sort_values("_p")

    fig, ax = plt.subplots(figsize=(COL_WIDTH_IN, COL_WIDTH_IN * 0.8))
    if not sweep.empty:
        ax.errorbar(
            sweep["_p"].to_numpy(),
            sweep["mean"].to_numpy(),
            yerr=sweep["std"].to_numpy(),
            marker="o",
            capsize=2,
            color="#1b1b1b",
            label="Lp-QKNorm sweep",
        )
    if not vanilla.empty:
        v = float(vanilla["mean"].iloc[0])
        ax.axhline(
            v,
            color=color_for_p("vanilla"),
            linestyle="--",
            linewidth=0.9,
            label=f"vanilla = {v:.3f}",
        )
    ax.axvline(2.0, color="0.6", linestyle=":", linewidth=0.7)
    ax.set_xlabel(r"Lp exponent $p$")
    ax.set_ylabel("Lesion recall (small stratum)")
    ax.set_title("Small-lesion recall vs p")
    ax.legend(frameon=False, loc="best")
    return _savefig(fig, out)


def fig_probe_trajectory(probes: pd.DataFrame, out: Path | str) -> Path:
    """Per-probe trajectory across ``p``; 5 subplots in a 2×3 grid (last empty)."""
    if probes.empty:
        raise AnalysisError("probes is empty; nothing to plot")
    needed = {"p", "p_label", "fold", "probe_name", "value"}
    missing = needed - set(probes.columns)
    if missing:
        raise AnalysisError(f"probes missing columns: {sorted(missing)}")
    set_publication_style()

    probe_order = (
        "feature_peakiness_q",
        "attention_entropy",
        "lesion_attention_mass",
        "lesion_background_logit_gap",
        "attention_mask_iou",
    )
    available = [pn for pn in probe_order if pn in probes["probe_name"].unique()]
    if not available:
        raise AnalysisError("none of the canonical probes is present in 'probes'")

    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(DOUBLE_COL_WIDTH_IN, DOUBLE_COL_WIDTH_IN * 0.55),
        sharex=True,
    )
    for ax, pn in zip(axes.flat, available, strict=False):
        sub = probes[(probes["probe_name"] == pn) & (probes["p_label"] != "vanilla")]
        if sub.empty:
            ax.set_visible(False)
            continue
        per_run = (
            sub.groupby(["p_label", "fold"], dropna=False)["value"].mean().reset_index()
        )
        per_run["_p"] = per_run["p_label"].astype(float)
        per_p = (
            per_run.groupby(["p_label", "_p"], dropna=False)["value"]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values("_p")
        )
        ax.errorbar(
            per_p["_p"].to_numpy(),
            per_p["mean"].to_numpy(),
            yerr=per_p["std"].to_numpy(),
            marker="o",
            capsize=2,
            color="black",
        )
        ax.axvline(2.0, color="0.6", linestyle=":", linewidth=0.6)
        ax.set_title(pn.replace("_", " "))
        ax.set_xlabel(r"$p$")
    # Blank any unused axes.
    for ax in axes.flat[len(available) :]:
        ax.set_visible(False)
    fig.tight_layout()
    return _savefig(fig, out)


def fig_mechanism_chain(
    per_patient: pd.DataFrame, probes: pd.DataFrame, out: Path | str
) -> Path:
    """Scatter of mean probe value vs lesion recall, colour-coded by ``p``."""
    if probes.empty:
        raise AnalysisError("probes is empty; cannot draw mechanism chain")
    _check_per_patient(per_patient)

    # Reduce probes to one mean value per (p, fold) for the canonical
    # probe ordering.
    set_publication_style()
    probe_order = (
        "feature_peakiness_q",
        "attention_entropy",
        "lesion_attention_mass",
    )
    available = [pn for pn in probe_order if pn in probes["probe_name"].unique()]
    if not available:
        raise AnalysisError("no canonical probes available for mechanism chain")

    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(available),
        figsize=(DOUBLE_COL_WIDTH_IN, COL_WIDTH_IN),
        sharey=True,
    )
    if len(available) == 1:
        axes = np.array([axes])

    small = (
        per_patient[per_patient["volume_stratum"] == "small"]
        .groupby(["p", "p_label", "fold"], dropna=False)["lesion_recall"]
        .mean()
        .reset_index()
    )
    for ax, pn in zip(axes, available, strict=False):
        probe_grp = (
            probes[probes["probe_name"] == pn]
            .groupby(["p", "p_label", "fold"], dropna=False)["value"]
            .mean()
            .reset_index()
        )
        merged = small.merge(
            probe_grp, on=["p_label", "fold"], how="inner", suffixes=("_pp", "")
        )
        for _, row in merged.iterrows():
            ax.scatter(
                row["value"],
                row["lesion_recall"],
                color=color_for_p(str(row["p_label"])),
                s=28,
                edgecolor="black",
                linewidths=0.4,
            )
        if len(merged) >= 3:
            x = merged["value"].to_numpy()
            y = merged["lesion_recall"].to_numpy()
            slope, intercept = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 30)
            ax.plot(xs, slope * xs + intercept, "k-", linewidth=0.8)
        ax.set_xlabel(pn.replace("_", " "))
    axes[0].set_ylabel("Small-lesion recall")
    fig.tight_layout()
    return _savefig(fig, out)


def fig_per_patient_effect(per_patient: pd.DataFrame, out: Path | str) -> Path:
    """Forest-style plot of per-patient (lesion-recall p* − p_2) for the small stratum.

    Treatment ``p`` is the one with the highest mean small-stratum
    recall; control is ``p = 2``.  Each row is a patient sorted by
    effect.
    """
    _check_per_patient(per_patient)
    if "lesion_recall" not in per_patient.columns:
        raise AnalysisError("per_patient lacks 'lesion_recall'")
    set_publication_style()

    small = per_patient[per_patient["volume_stratum"] == "small"].copy()
    small["p_label"] = small["p_label"].astype(str)
    if "2.0" not in small["p_label"].unique():
        raise AnalysisError("p=2.0 baseline missing from per_patient")
    treat_candidates = (
        small[small["p_label"] != "vanilla"]
        .groupby("p_label")["lesion_recall"]
        .mean()
        .sort_values(ascending=False)
    )
    treat_candidates = treat_candidates.drop(index="2.0", errors="ignore")
    if treat_candidates.empty:
        raise AnalysisError("no comparison p found beyond the p=2 baseline")
    p_star = treat_candidates.index[0]

    pivot = (
        small.groupby(["subject_id", "p_label"])["lesion_recall"]
        .mean()
        .unstack("p_label")
    )
    if "2.0" not in pivot.columns or p_star not in pivot.columns:
        raise AnalysisError("pivot missing required columns")
    diff = (pivot[p_star] - pivot["2.0"]).dropna().sort_values()
    fig, ax = plt.subplots(figsize=(COL_WIDTH_IN, max(2.0, 0.06 * len(diff))))
    y = np.arange(len(diff))
    colors = ["#67000d" if v >= 0 else "#08306b" for v in diff.values]
    ax.barh(y, diff.values, color=colors, edgecolor="black", linewidth=0.3)
    ax.axvline(0.0, color="black", linewidth=0.6)
    ax.set_yticks([])
    ax.set_xlabel(rf"recall at p={p_star} - recall at p=2")
    ax.set_ylabel(f"Patients (n = {len(diff)})")
    ax.set_title(f"Per-patient effect (small stratum, p* = {p_star})")
    return _savefig(fig, out)


# ---------------------------------------------------------------------------
# Local utilities
# ---------------------------------------------------------------------------


def _safe_float(x: object) -> float:
    try:
        return float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("inf")
