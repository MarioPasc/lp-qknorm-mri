"""Phase 5 — downstream analysis, effect sizes, and figures."""

from __future__ import annotations

from lpqknorm.analysis.aggregation import (
    load_per_patient,
    load_probes,
    load_runs,
)
from lpqknorm.analysis.bootstrap import (
    BootstrapResult,
    holm_bonferroni,
    paired_patient_bootstrap,
)
from lpqknorm.analysis.effect_size import hedges_g, paired_cohen_d
from lpqknorm.analysis.figures import (
    fig_mechanism_chain,
    fig_per_patient_effect,
    fig_probe_trajectory,
    fig_small_recall_vs_p,
    fig_stratified_dice,
    fig_toy_model_prediction,
)
from lpqknorm.analysis.probe_curves import probe_curve, probe_outcome_correlation
from lpqknorm.analysis.stratification import attach_strata
from lpqknorm.analysis.style import set_publication_style


__all__ = [
    "BootstrapResult",
    "attach_strata",
    "fig_mechanism_chain",
    "fig_per_patient_effect",
    "fig_probe_trajectory",
    "fig_small_recall_vs_p",
    "fig_stratified_dice",
    "fig_toy_model_prediction",
    "hedges_g",
    "holm_bonferroni",
    "load_per_patient",
    "load_probes",
    "load_runs",
    "paired_cohen_d",
    "paired_patient_bootstrap",
    "probe_curve",
    "probe_outcome_correlation",
    "set_publication_style",
]
