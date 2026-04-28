"""Walk a Phase 3/4 results tree and load it into long-format dataframes.

The expected directory layout is::

    {results_root}/p={p}/fold={fold}/seed={seed}/
        config.yaml
        manifest.json
        metrics/test_per_patient.parquet
        metrics/val_per_patient.parquet
        probes/epoch_{tag}.h5            # optional, written by Phase 4

This module is a thin convenience layer over filesystem walks.  It is
deliberately permissive: missing artefacts are reported as warnings,
not hard errors, so an analysis can run while Phase 4 is still in
flight on the cluster.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from lpqknorm.analysis.bootstrap import AnalysisError


logger = logging.getLogger(__name__)


_RUN_RE = re.compile(r"p=(?P<p>[^/]+)/fold=(?P<fold>\d+)/seed=(?P<seed>\d+)")


def _parse_p(token: str) -> float | str:
    """Parse the ``p=`` directory token: numeric or ``"vanilla"``."""
    if token.lower() == "vanilla":
        return "vanilla"
    try:
        return float(token)
    except ValueError as exc:
        raise AnalysisError(f"unparseable p token: {token!r}") from exc


def load_runs(results_root: Path | str, experiment: str | None = None) -> pd.DataFrame:
    """Discover every ``(p, fold, seed)`` run directory under *results_root*.

    Parameters
    ----------
    results_root : Path or str
        Root of the sweep tree (the directory that directly contains
        ``p=*`` subdirectories).
    experiment : str or None
        Optional experiment name; if given, must match the
        ``experiment`` field in each run's ``manifest.json`` /
        ``config.yaml`` (rows with a different value are dropped with
        a warning).

    Returns
    -------
    pd.DataFrame
        One row per discovered run with columns
        ``[run_id, p, fold, seed, run_dir, experiment, dataset_name,
        wall_time_sec, final_epoch, best_val_dice, best_small_recall]``.
        Numeric fields default to ``NaN`` when unavailable.

    Raises
    ------
    AnalysisError
        If *results_root* does not exist.
    """
    root = Path(results_root)
    if not root.exists():
        raise AnalysisError(f"results_root does not exist: {root}")

    rows: list[dict[str, object]] = []
    for run_dir in sorted(root.glob("p=*/fold=*/seed=*")):
        match = _RUN_RE.search(run_dir.as_posix())
        if not match:
            continue
        p_raw = match.group("p")
        fold = int(match.group("fold"))
        seed = int(match.group("seed"))
        try:
            p_val = _parse_p(p_raw)
        except AnalysisError as exc:
            logger.warning("Skipping unparseable run dir %s: %s", run_dir, exc)
            continue

        manifest_path = run_dir / "manifest.json"
        manifest: dict[str, object] = {}
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Could not read %s: %s", manifest_path, exc)

        exp_name = manifest.get("experiment") if manifest else None
        if experiment is not None and exp_name not in (None, experiment):
            logger.info(
                "Skipping run %s: experiment=%s != %s", run_dir, exp_name, experiment
            )
            continue

        run_id = f"p={p_raw}_f{fold}_s{seed}"
        rows.append(
            {
                "run_id": run_id,
                "p": p_val,
                "p_label": p_raw,
                "fold": fold,
                "seed": seed,
                "run_dir": str(run_dir),
                "experiment": exp_name,
                "dataset_name": manifest.get("dataset_name"),
                "wall_time_sec": manifest.get("walltime_sec", float("nan")),
                "final_epoch": manifest.get("final_epoch", float("nan")),
                "best_val_dice": manifest.get("best_val_dice", float("nan")),
                "best_small_recall": manifest.get("best_small_recall", float("nan")),
            }
        )

    return pd.DataFrame(rows)


def load_per_patient(
    results_root: Path | str,
    *,
    split: Literal["test", "val"] = "test",
    aggregate: bool = True,
) -> pd.DataFrame:
    """Load ``metrics/{split}_per_patient.parquet`` from every run.

    The parquet files are slice-level (one row per slice).  When
    ``aggregate`` is True (default), rows are collapsed to one entry
    per ``(run_id, subject_id, volume_stratum)`` by averaging numeric
    metrics — this is the level at which patient-level paired
    comparisons are computed (Nadeau & Bengio, 2003).

    Parameters
    ----------
    results_root : Path or str
        Root of the sweep tree.
    split : {"test", "val"}
        Which evaluation split to load.
    aggregate : bool, default True
        If True, return one row per patient × stratum.  If False,
        return the raw slice-level rows.

    Returns
    -------
    pd.DataFrame
        Long-format frame with at minimum the columns
        ``[run_id, p, p_label, fold, seed, subject_id, volume_stratum,
        dice, iou, lesion_recall, false_positives_per_slice]``.
        Returns an empty frame when no parquet files are found.
    """
    runs = load_runs(results_root)
    if runs.empty:
        return pd.DataFrame()

    parts: list[pd.DataFrame] = []
    metric_cols = {"dice", "iou", "lesion_recall", "false_positives_per_slice"}
    for _, row in runs.iterrows():
        path = Path(str(row["run_dir"])) / "metrics" / f"{split}_per_patient.parquet"
        if not path.exists():
            logger.warning("Missing %s; skipping run_id=%s", path, row["run_id"])
            continue
        try:
            df = pd.read_parquet(path)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed reading %s: %s", path, exc)
            continue
        df = df.copy()
        df["run_id"] = row["run_id"]
        df["p"] = row["p"]
        df["p_label"] = row["p_label"]
        df["fold"] = row["fold"]
        df["seed"] = row["seed"]
        parts.append(df)

    if not parts:
        return pd.DataFrame()
    cat = pd.concat(parts, ignore_index=True)

    if not aggregate:
        return cat

    keep = [c for c in cat.columns if c in metric_cols]
    grouped = cat.groupby(
        ["run_id", "p", "p_label", "fold", "seed", "subject_id", "volume_stratum"],
        as_index=False,
        dropna=False,
    )[keep].mean(numeric_only=True)
    return grouped


def load_probes(
    results_root: Path | str,
    *,
    checkpoint: Literal["best_dice", "final"] = "best_dice",
) -> pd.DataFrame:
    """Load every ``probes/epoch_{checkpoint}.h5`` into a long-format frame.

    Each HDF5 (Phase 4 schema) contains ``/block_0_wmsa`` and
    ``/block_1_swmsa`` groups with per-token / per-query datasets named
    after the probes (e.g. ``feature_peakiness_q``, ``attention_entropy``,
    ``lesion_attention_mass``, ``lesion_background_logit_gap``,
    ``attention_mask_iou``, ``spatial_localization_error``).  The
    ``/inputs`` group holds ``subject_id`` (one per probe slice).

    Parameters
    ----------
    results_root : Path or str
        Root of the sweep tree.
    checkpoint : {"best_dice", "final"}
        Which probe HDF5 to load.

    Returns
    -------
    pd.DataFrame
        One row per ``(run_id, block, slice_index, probe_name)`` with
        columns ``[..., subject_id, value]``.  Empty if no probe files
        are present (Phase 4 not yet executed).
    """
    import h5py  # imported lazily; analysis pkg should still import without h5py

    runs = load_runs(results_root)
    if runs.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for _, run in runs.iterrows():
        h5_path = Path(str(run["run_dir"])) / "probes" / f"epoch_{checkpoint}.h5"
        if not h5_path.exists():
            continue
        try:
            with h5py.File(h5_path, "r") as f:
                inputs = f.get("inputs")
                if inputs is None or "subject_id" not in inputs:
                    sids: list[str] = []
                else:
                    raw = inputs["subject_id"][...]
                    sids = [
                        s.decode() if isinstance(s, (bytes, bytearray)) else str(s)
                        for s in raw
                    ]
                for block_name in (k for k in f if k.startswith("block_")):
                    block_grp = f[block_name]
                    for probe_name in block_grp:
                        ds = block_grp[probe_name]
                        if not hasattr(ds, "shape"):
                            continue
                        arr = np.asarray(ds[...])
                        # Skip shapes that don't fit the legacy
                        # per-slice schema: 0-D scalars, very long
                        # flat arrays (per-token), and per-head 1D.
                        if arr.ndim == 0:
                            continue
                        if arr.ndim >= 2:
                            arr = arr.reshape(arr.shape[0], -1).mean(axis=1)
                        if arr.shape[0] == 0 or arr.shape[0] > max(1024, len(sids) * 4):
                            continue
                        for i, val in enumerate(arr.tolist()):
                            sid = sids[i] if i < len(sids) else None
                            rows.append(
                                {
                                    "run_id": run["run_id"],
                                    "p": run["p"],
                                    "p_label": run["p_label"],
                                    "fold": run["fold"],
                                    "seed": run["seed"],
                                    "block": block_name,
                                    "slice_index": i,
                                    "subject_id": sid,
                                    "probe_name": probe_name,
                                    "value": float(val),
                                }
                            )
        except (OSError, KeyError) as exc:
            logger.warning("Could not parse %s: %s", h5_path, exc)
            continue

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Phase-4 schema-aware summary loader (real recorder output)
# ---------------------------------------------------------------------------


# Per-token probes carry a parallel ``*_is_lesion`` flag and are flat
# (n_slices × n_windows × n_heads × n_tokens).  Reducing to per-block
# scalars (mean over lesion / background tokens) yields the Probe-1/2
# summary used by Figure 4.
_PER_TOKEN_PROBES: tuple[tuple[str, str], ...] = (
    ("peakiness_q", "peakiness_q_is_lesion"),
    ("peakiness_k", "peakiness_k_is_lesion"),
    ("entropy", "peakiness_q_is_lesion"),
)
# Per-query probes are 1D arrays whose entries are already restricted to
# lesion queries; the mean is the per-block summary.
_PER_QUERY_PROBES: tuple[str, ...] = (
    "lesion_mass",
    "logit_gap",
    "attention_iou",
    "spatial_localization_error",
)
# Per-block scalar summaries.
_SCALAR_PROBES: tuple[str, ...] = (
    "alpha",
    "pr_lesion",
    "pr_background",
    "stable_rank_lesion",
    "stable_rank_background",
)


def _decode(arr: np.ndarray) -> np.ndarray:
    """Decode a byte-string ndarray to UTF-8 strings."""
    if arr.dtype.kind in ("S", "O"):
        return np.array(
            [s.decode() if isinstance(s, (bytes, bytearray)) else str(s) for s in arr]
        )
    return arr


def load_probe_summaries(
    results_root: Path | str,
    *,
    checkpoint: Literal["best_dice", "final"] = "best_dice",
) -> pd.DataFrame:
    """Load every probe HDF5 and reduce each block to scalar summaries.

    Schema (Phase 4 ``ProbeRecorder`` output)::

        block_{0,1}_(s)wmsa/
            peakiness_q (float32, flat)             + peakiness_q_is_lesion (bool, flat)
            peakiness_k (float32, flat)             + peakiness_k_is_lesion (bool, flat)
            entropy (float32, flat)
            lesion_mass, logit_gap, attention_iou,
            spatial_localization_error (float32, 1D, lesion-only)
            alpha, pr_lesion, pr_background,
            stable_rank_lesion, stable_rank_background (scalar)

    Per-token quantities are stratified by ``*_is_lesion`` to obtain a
    lesion-mean and background-mean.  Per-query quantities are
    summarised as their plain mean (already restricted to lesion
    queries).  Per-block scalars are returned as-is.

    Parameters
    ----------
    results_root : Path or str
    checkpoint : {"best_dice", "final"}

    Returns
    -------
    pd.DataFrame
        One row per ``(run_id, block, summary_name)`` with the canonical
        columns ``[run_id, p, p_label, fold, seed, block, summary_name,
        value, n]`` (``n`` = number of underlying tokens / queries used
        in the reduction).  Empty if no probe files are present.
    """
    import h5py

    runs = load_runs(results_root)
    if runs.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for _, run in runs.iterrows():
        h5_path = Path(str(run["run_dir"])) / "probes" / f"epoch_{checkpoint}.h5"
        if not h5_path.exists():
            continue
        try:
            with h5py.File(h5_path, "r") as f:
                for block_name in (k for k in f if k.startswith("block_")):
                    g = f[block_name]
                    summaries: dict[str, tuple[float, int]] = {}

                    for probe, flag in _PER_TOKEN_PROBES:
                        if probe not in g or flag not in g:
                            continue
                        vals = np.asarray(g[probe][...], dtype=np.float64)
                        is_lesion = np.asarray(g[flag][...], dtype=bool)
                        if vals.size == 0:
                            continue
                        if is_lesion.any():
                            summaries[f"{probe}_lesion_mean"] = (
                                float(vals[is_lesion].mean()),
                                int(is_lesion.sum()),
                            )
                        if (~is_lesion).any():
                            summaries[f"{probe}_background_mean"] = (
                                float(vals[~is_lesion].mean()),
                                int((~is_lesion).sum()),
                            )

                    for probe in _PER_QUERY_PROBES:
                        if probe not in g:
                            continue
                        vals = np.asarray(g[probe][...], dtype=np.float64)
                        if vals.size == 0:
                            continue
                        summaries[f"{probe}_mean"] = (
                            float(np.nanmean(vals)),
                            int(vals.size),
                        )

                    for probe in _SCALAR_PROBES:
                        if probe not in g:
                            continue
                        vals = np.asarray(g[probe][...], dtype=np.float64).reshape(-1)
                        if vals.size == 0:
                            continue
                        summaries[probe] = (float(vals.mean()), int(vals.size))

                    for name, (val, n) in summaries.items():
                        rows.append(
                            {
                                "run_id": run["run_id"],
                                "p": run["p"],
                                "p_label": run["p_label"],
                                "fold": run["fold"],
                                "seed": run["seed"],
                                "block": block_name,
                                "summary_name": name,
                                "value": val,
                                "n": n,
                            }
                        )
        except (OSError, KeyError) as exc:
            logger.warning("Could not parse %s: %s", h5_path, exc)
            continue

    return pd.DataFrame(rows)
