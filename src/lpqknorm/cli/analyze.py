"""CLI entry point: ``lpqknorm-analyze``.

Walks a Phase 3/4 results tree, computes the stratified metrics
table, runs the patient-level paired bootstrap and Holm correction
against ``p = 2``, writes Figures 1-6 (Phase 5 spec), and emits a
``analysis_manifest.json`` describing the analysed run set.

Usage::

    lpqknorm-analyze \
        --results-root /path/to/results/p_sweep_v1 \
        --output       /path/to/paper_outputs

Or, equivalently, via the Hydra-style key=value form mentioned in
``docs/phase_05_analysis.md``::

    python -m lpqknorm.cli.analyze \
        results_root=/path/to/results/p_sweep_v1 \
        output=/path/to/paper_outputs
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from lpqknorm.analysis.aggregation import (
    load_per_patient,
    load_probe_summaries,
    load_probes,
    load_runs,
)
from lpqknorm.analysis.bootstrap import (
    BootstrapResult,
    holm_bonferroni,
    paired_patient_bootstrap,
)
from lpqknorm.analysis.effect_size import paired_cohen_d
from lpqknorm.analysis.figures import (
    fig_mechanism_chain,
    fig_per_patient_effect,
    fig_probe_trajectory,
    fig_small_recall_vs_p,
    fig_stratified_dice,
    fig_toy_model_prediction,
)


if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


logger = logging.getLogger(__name__)

CANONICAL_PROBES = (
    "feature_peakiness_q",
    "attention_entropy",
    "lesion_attention_mass",
    "lesion_background_logit_gap",
    "attention_mask_iou",
)


# ---------------------------------------------------------------------------
# Argument parsing — accept both --flag and Hydra-style key=value tokens.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnalyzeConfig:
    results_root: Path
    output: Path
    seed: int = 20260216
    n_resamples: int = 10_000
    p_baseline: str = "2.0"
    sparsity_levels: tuple[int, ...] = (2, 4, 8, 16)
    d_k: int = 24


def _split_kv_tokens(argv: Sequence[str]) -> list[str]:
    """Translate ``key=value`` Hydra tokens to ``--key value`` argparse form."""
    out: list[str] = []
    for tok in argv:
        if tok.startswith("--") or "=" not in tok or tok.startswith("="):
            out.append(tok)
            continue
        key, val = tok.split("=", 1)
        if key.replace("_", "").isalnum():
            out.extend([f"--{key}", val])
        else:
            out.append(tok)
    return out


def _parse_args(argv: Sequence[str] | None) -> AnalyzeConfig:
    parser = argparse.ArgumentParser(description="Phase 5 analysis CLI")
    parser.add_argument("--results-root", "--results_root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260216)
    parser.add_argument("--n-resamples", "--n_resamples", type=int, default=10_000)
    parser.add_argument(
        "--p-baseline",
        "--p_baseline",
        type=str,
        default="2.0",
        help="Reference p label for paired comparisons.",
    )
    parser.add_argument(
        "--sparsity-levels",
        "--sparsity_levels",
        type=str,
        default="2,4,8,16",
        help="Comma-separated integer sparsity levels for fig 1.",
    )
    parser.add_argument("--d-k", "--d_k", type=int, default=24)
    raw = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(_split_kv_tokens(raw))
    sparsity = tuple(int(x) for x in str(args.sparsity_levels).split(",") if x)
    return AnalyzeConfig(
        results_root=Path(args.results_root),
        output=Path(args.output),
        seed=int(args.seed),
        n_resamples=int(args.n_resamples),
        p_baseline=str(args.p_baseline),
        sparsity_levels=sparsity,
        d_k=int(args.d_k),
    )


# ---------------------------------------------------------------------------
# Pipeline pieces
# ---------------------------------------------------------------------------


def _stratified_table(per_patient: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std of every numeric metric, per (p, stratum, fold)."""
    metric_cols = [
        c
        for c in ("dice", "iou", "lesion_recall", "false_positives_per_slice")
        if c in per_patient.columns
    ]
    fold_means = (
        per_patient.groupby(["p_label", "p", "fold", "volume_stratum"], dropna=False)[
            metric_cols
        ]
        .mean(numeric_only=True)
        .reset_index()
    )
    summary = (
        fold_means.groupby(["p_label", "p", "volume_stratum"], dropna=False)[
            metric_cols
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join([c for c in col if c]).rstrip("_") for col in summary.columns.values
    ]
    return summary


def _headline_bootstrap(
    per_patient: pd.DataFrame,
    p_baseline: str,
    seed: int,
    n_resamples: int,
    metric: str = "lesion_recall",
    stratum: str = "small",
) -> tuple[pd.DataFrame, dict[str, BootstrapResult]]:
    """Pool patients across folds and bootstrap each ``p`` vs ``p_baseline``.

    Returns the summary table and a dict of raw BootstrapResult objects.
    """
    sub = per_patient[per_patient["volume_stratum"] == stratum].copy()
    if sub.empty:
        return pd.DataFrame(), {}
    pivot = (
        sub.groupby(["fold", "subject_id", "p_label"])[metric].mean().unstack("p_label")
    )
    if p_baseline not in pivot.columns:
        return pd.DataFrame(), {}

    rows: list[dict[str, object]] = []
    raw: dict[str, BootstrapResult] = {}
    for p in pivot.columns:
        if p == p_baseline:
            continue
        pair = pivot[[p_baseline, p]].dropna()
        if len(pair) < 2:
            continue
        res = paired_patient_bootstrap(
            pair[p].to_numpy(),
            pair[p_baseline].to_numpy(),
            n_resamples=n_resamples,
            seed=seed,
        )
        d = paired_cohen_d(pair[p].to_numpy(), pair[p_baseline].to_numpy())
        rows.append(
            {
                "p_label": p,
                "metric": metric,
                "stratum": stratum,
                "delta_mean": res.mean,
                "ci_low": res.ci_low,
                "ci_high": res.ci_high,
                "p_value_one_sided": res.p_value_one_sided,
                "cohen_d": d,
                "n_patients": res.n_patients,
            }
        )
        raw[f"{metric}_{stratum}_p={p}_vs_p={p_baseline}"] = res

    if not rows:
        return pd.DataFrame(), raw
    df = pd.DataFrame(rows)
    df["p_holm"] = holm_bonferroni(df["p_value_one_sided"].to_numpy())
    return df, raw


def _per_fold_argmax(per_patient: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        c for c in ("dice", "iou", "lesion_recall") if c in per_patient.columns
    ]
    rows: list[dict[str, object]] = []
    for fold, gfold in per_patient.groupby("fold", dropna=False):
        for stratum, gstr in gfold.groupby("volume_stratum", dropna=False):
            for metric in metric_cols:
                ranking = (
                    gstr.groupby("p_label")[metric].mean().sort_values(ascending=False)
                )
                if ranking.empty:
                    continue
                rows.append(
                    {
                        "fold": fold,
                        "stratum": stratum,
                        "metric": metric,
                        "p_star": ranking.index[0],
                        "value": float(ranking.iloc[0]),
                    }
                )
    return pd.DataFrame(rows)


def _write_probe_summary_figure(probe_summaries: pd.DataFrame, out: Path) -> Path:
    """Plot a 3x3 grid of probe scalar summaries vs ``p`` (one panel per
    summary).  Lesion vs background are overlaid for the per-token probes.
    """
    import matplotlib.pyplot as plt

    from lpqknorm.analysis.style import (
        DOUBLE_COL_WIDTH_IN,
        color_for_p,
        set_publication_style,
    )

    set_publication_style()
    df = probe_summaries.copy()
    # Average across blocks (W-MSA + SW-MSA) to reduce panel count.
    df = (
        df.groupby(["p_label", "p", "fold", "summary_name"], dropna=False)["value"]
        .mean()
        .reset_index()
    )
    summary_names = sorted(df["summary_name"].unique())
    n = len(summary_names)
    if n == 0:
        raise ValueError("no probe summaries to plot")
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(DOUBLE_COL_WIDTH_IN, DOUBLE_COL_WIDTH_IN * 0.55 * max(nrows / 2, 1)),
        squeeze=False,
    )
    for ax, name in zip(axes.flat, summary_names, strict=False):
        sub = df[df["summary_name"] == name]
        per_p = (
            sub.groupby(["p_label", "p"], dropna=False)["value"]
            .agg(["mean", "std"])
            .reset_index()
        )
        per_p["_p"] = per_p["p_label"].astype(str)
        # Drop non-numeric p (vanilla) for a continuous axis.
        per_p = per_p[per_p["_p"] != "vanilla"].copy()
        if per_p.empty:
            ax.set_visible(False)
            continue
        per_p["_p"] = per_p["_p"].astype(float)
        per_p = per_p.sort_values("_p")
        ax.errorbar(
            per_p["_p"].to_numpy(),
            per_p["mean"].to_numpy(),
            yerr=per_p["std"].to_numpy(),
            marker="o",
            capsize=2,
            color=color_for_p("2.0"),
        )
        ax.axvline(2.0, color="0.6", linestyle=":", linewidth=0.6)
        ax.set_title(name.replace("_", " "), fontsize=8)
        ax.set_xlabel(r"$p$")
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    fig.tight_layout()
    out_p = Path(out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_p, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    return out_p


def _write_table(df: pd.DataFrame, base: Path) -> Iterable[Path]:
    base.parent.mkdir(parents=True, exist_ok=True)
    pq = base.with_suffix(".parquet")
    # PyArrow refuses to serialise object columns whose values mix
    # numeric and string types (the ``p`` column is float for numeric
    # sweeps but ``"vanilla"`` for the no-QKNorm control).  Cast every
    # object column to string before writing.
    safe = df.copy()
    for col in safe.select_dtypes(include="object").columns:
        safe[col] = safe[col].astype(str)
    safe.to_parquet(pq)
    yield pq
    tex = base.with_suffix(".tex")
    try:
        safe.to_latex(tex, index=False, float_format="%.4f", escape=True)
        yield tex
    except Exception as exc:  # pragma: no cover - LaTeX export depends on jinja
        logger.warning("Skipping LaTeX export for %s: %s", base, exc)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    cfg = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    out = cfg.output
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "bootstrap").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(out / "logs" / "analyze.log", mode="w")
    fh.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logging.getLogger().addHandler(fh)
    t0 = time.perf_counter()

    runs = load_runs(cfg.results_root)
    logger.info("Discovered %d runs", len(runs))
    per_patient = load_per_patient(cfg.results_root)
    if per_patient.empty:
        logger.warning("No per_patient parquet files found under %s", cfg.results_root)
    probes = load_probes(cfg.results_root)
    logger.info("Loaded probes (raw): %d rows", len(probes))
    probe_summaries = load_probe_summaries(cfg.results_root)
    logger.info("Loaded probe summaries: %d rows", len(probe_summaries))

    # ---------- tables ----------
    if not per_patient.empty:
        strat = _stratified_table(per_patient)
        for p in _write_table(strat, out / "tables" / "stratified_metrics"):
            logger.info("Wrote %s", p)

        for stratum in ("small", "medium", "large"):
            for metric in ("lesion_recall", "dice"):
                head_df, raw = _headline_bootstrap(
                    per_patient,
                    p_baseline=cfg.p_baseline,
                    seed=cfg.seed,
                    n_resamples=cfg.n_resamples,
                    metric=metric,
                    stratum=stratum,
                )
                if head_df.empty:
                    continue
                base = out / "tables" / f"headline_{metric}_{stratum}"
                for p in _write_table(head_df, base):
                    logger.info("Wrote %s", p)
                for key, res in raw.items():
                    np.savez(
                        out / "bootstrap" / f"{key}.npz",
                        distribution=res.distribution,
                        mean=res.mean,
                        ci_low=res.ci_low,
                        ci_high=res.ci_high,
                        p_value_one_sided=res.p_value_one_sided,
                        n_patients=res.n_patients,
                    )

        argmax = _per_fold_argmax(per_patient)
        if not argmax.empty:
            for p in _write_table(argmax, out / "tables" / "per_fold_argmax"):
                logger.info("Wrote %s", p)

    # ---------- figures ----------
    fig_paths: dict[str, str] = {}
    fig_paths["fig1_toy_model_prediction"] = str(
        fig_toy_model_prediction(
            cfg.sparsity_levels,
            d_k=cfg.d_k,
            out=out / "figures" / "fig1_toy_model_prediction.pdf",
        )
    )
    if not per_patient.empty:
        fig_paths["fig2_stratified_dice"] = str(
            fig_stratified_dice(
                per_patient, out=out / "figures" / "fig2_stratified_dice.pdf"
            )
        )
        fig_paths["fig3_small_recall_vs_p"] = str(
            fig_small_recall_vs_p(
                per_patient, out=out / "figures" / "fig3_small_recall_vs_p.pdf"
            )
        )
        try:
            fig_paths["fig6_per_patient_effect"] = str(
                fig_per_patient_effect(
                    per_patient,
                    out=out / "figures" / "fig6_per_patient_effect.pdf",
                )
            )
        except Exception as exc:
            logger.warning("Skipping fig6: %s", exc)
    if not probes.empty:
        try:
            fig_paths["fig4_probe_trajectory"] = str(
                fig_probe_trajectory(
                    probes, out=out / "figures" / "fig4_probe_trajectory.pdf"
                )
            )
        except Exception as exc:
            logger.warning("Skipping fig4: %s", exc)
        if not per_patient.empty:
            try:
                fig_paths["fig5_mechanism_chain"] = str(
                    fig_mechanism_chain(
                        per_patient,
                        probes,
                        out=out / "figures" / "fig5_mechanism_chain.pdf",
                    )
                )
            except Exception as exc:
                logger.warning("Skipping fig5: %s", exc)

    # ---------- probe summaries (Phase-4 schema-aware) ----------
    if not probe_summaries.empty:
        for p in _write_table(probe_summaries, out / "tables" / "probe_summaries"):
            logger.info("Wrote %s", p)
        try:
            fig_paths["fig4b_probe_summaries"] = str(
                _write_probe_summary_figure(
                    probe_summaries,
                    out=out / "figures" / "fig4b_probe_summaries.pdf",
                )
            )
        except Exception as exc:
            logger.warning("Skipping fig4b: %s", exc)

    # ---------- manifest ----------
    manifest = {
        "results_root": str(cfg.results_root),
        "output": str(out),
        "seed": cfg.seed,
        "n_resamples": cfg.n_resamples,
        "p_baseline": cfg.p_baseline,
        "sparsity_levels": list(cfg.sparsity_levels),
        "d_k": cfg.d_k,
        "n_runs": len(runs),
        "n_per_patient_rows": len(per_patient),
        "n_probe_rows": len(probes),
        "n_probe_summary_rows": len(probe_summaries),
        "n_probe_files": int(probe_summaries["run_id"].nunique())
        if not probe_summaries.empty
        else 0,
        "figures": fig_paths,
        "elapsed_seconds": round(time.perf_counter() - t0, 3),
    }
    (out / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info("Done in %.2fs; manifest at %s", manifest["elapsed_seconds"], out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
