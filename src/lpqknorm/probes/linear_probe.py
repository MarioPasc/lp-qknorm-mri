"""Probe 7 — L1-regularised linear probe (per-head logistic regression).

Substitute for sparse-autoencoder dictionary learning at Swin-UNETR
stage-0 scale (d_head in [4, 48]).  For each head ``h`` we fit

    L_LP(w, b) = mean_i log(1 + exp(-y_i (w^T x_i + b))) + lambda * ||w||_1

with ``y_i in {-1, +1}`` indicating lesion vs background, and ``x_i``
being the pre-norm query feature for token ``i`` of head ``h``.  Report
three scalars per head:

- Balanced accuracy ``BA = 0.5 * (TPR + TNR)`` under 5-fold CV.
- Decision-boundary sparsity ``sigma(w) = ||w||_1 / (||w||_2 + eps)``.
- Mean signed margin ``mbar = (1/N) sum_i y_i (w^T x_i + b)``.

References
----------
- Alain & Bengio.  *Understanding Intermediate Layers Using Linear
  Classifier Probes*.  ICLR 2017 workshop.  arXiv:1610.01644.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch


with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

from lpqknorm.probes.base import ProbeResult


if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor

    from lpqknorm.models.hooks import AttentionCapture


logger = logging.getLogger(__name__)

_DEFAULT_LAMBDA_GRID: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0)


@dataclass(frozen=True)
class LinearProbeMetrics:
    """Per-head metrics from one fitted linear probe.

    Parameters
    ----------
    balanced_accuracy : float
        Mean of sensitivity and specificity, averaged across CV folds.
    weight_sparsity : float
        ``||w||_1 / (||w||_2 + eps)``; larger means coefficients are
        more concentrated in a handful of dimensions.
    margin : float
        Mean signed functional margin ``y_i * (w^T x_i + b)`` on the
        full feature pool, using the full-data refit.
    """

    balanced_accuracy: float
    weight_sparsity: float
    margin: float


class LinearProbe:
    """Probe 7 — per-head L1-logistic regression linear probe."""

    name = "linear_probe"

    def __init__(
        self,
        n_splits: int = 5,
        lambda_grid: Sequence[float] = _DEFAULT_LAMBDA_GRID,
        random_state: int = 0,
        min_samples_per_class: int = 10,
    ) -> None:
        self._n_splits = n_splits
        self._lambda_grid = tuple(lambda_grid)
        self._random_state = random_state
        self._min_per_class = min_samples_per_class

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    @staticmethod
    def _sparsity(weight: np.ndarray, eps: float = 1e-12) -> float:
        l1 = float(np.abs(weight).sum())
        l2 = float(np.linalg.norm(weight))
        return l1 / (l2 + eps)

    def _fit_single(
        self,
        x_lesion: np.ndarray,
        x_bg: np.ndarray,
    ) -> LinearProbeMetrics:
        """Fit one linear probe on pooled features for a single head.

        Parameters
        ----------
        x_lesion : np.ndarray
            Shape ``(N_L, d_k)``.
        x_bg : np.ndarray
            Shape ``(N_B, d_k)``.

        Returns
        -------
        LinearProbeMetrics
        """
        n_l = x_lesion.shape[0]
        n_b = x_bg.shape[0]
        if n_l < self._min_per_class or n_b < self._min_per_class:
            logger.debug(
                "LinearProbe: insufficient samples (lesion=%d, bg=%d); "
                "returning NaN metrics.",
                n_l,
                n_b,
            )
            return LinearProbeMetrics(
                balanced_accuracy=float("nan"),
                weight_sparsity=float("nan"),
                margin=float("nan"),
            )

        x = np.concatenate([x_lesion, x_bg], axis=0).astype(np.float64)
        y = np.concatenate(
            [np.ones(n_l, dtype=np.int64), np.zeros(n_b, dtype=np.int64)]
        )

        # 5-fold stratified CV for balanced accuracy across lambda grid.
        # Pick the lambda with the highest mean BA.
        n_splits = min(self._n_splits, n_l, n_b)
        n_splits = max(2, n_splits)
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=self._random_state
        )

        best_lambda = self._lambda_grid[0]
        best_ba = -1.0
        for lam in self._lambda_grid:
            c = 1.0 / float(lam)
            ba_folds: list[float] = []
            for tr_idx, te_idx in skf.split(x, y):
                clf = LogisticRegression(
                    penalty="l1",
                    solver="liblinear",
                    C=c,
                    random_state=self._random_state,
                    max_iter=200,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=FutureWarning)
                    warnings.simplefilter("ignore", category=UserWarning)
                    clf.fit(x[tr_idx], y[tr_idx])
                yhat = clf.predict(x[te_idx])
                ytrue = y[te_idx]
                tp = float(((yhat == 1) & (ytrue == 1)).sum())
                tn = float(((yhat == 0) & (ytrue == 0)).sum())
                fp = float(((yhat == 1) & (ytrue == 0)).sum())
                fn = float(((yhat == 0) & (ytrue == 1)).sum())
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                ba_folds.append(0.5 * (tpr + tnr))
            mean_ba = float(np.mean(ba_folds))
            if mean_ba > best_ba:
                best_ba = mean_ba
                best_lambda = lam

        # Refit on full data with best lambda for sparsity and margin.
        c = 1.0 / float(best_lambda)
        final_clf = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=c,
            random_state=self._random_state,
            max_iter=500,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            warnings.simplefilter("ignore", category=UserWarning)
            final_clf.fit(x, y)
        w = final_clf.coef_.ravel()
        b = float(final_clf.intercept_.ravel()[0])

        # Signed margin with y_signed in {-1, +1}.
        y_signed = np.where(y == 1, 1.0, -1.0)
        decisions = x @ w + b
        margin = float((y_signed * decisions).mean())

        return LinearProbeMetrics(
            balanced_accuracy=best_ba,
            weight_sparsity=self._sparsity(w),
            margin=margin,
        )

    def compute_value(
        self,
        x_lesion: Tensor,
        x_bg: Tensor,
    ) -> LinearProbeMetrics:
        """Fit a single probe on already-pooled lesion/background features.

        Parameters
        ----------
        x_lesion : Tensor
            Shape ``(N_L, d_k)``.
        x_bg : Tensor
            Shape ``(N_B, d_k)``.

        Returns
        -------
        LinearProbeMetrics
        """
        return self._fit_single(
            x_lesion.detach().cpu().numpy(),
            x_bg.detach().cpu().numpy(),
        )

    # ------------------------------------------------------------------
    # Probe protocol
    # ------------------------------------------------------------------

    def compute(
        self,
        capture: AttentionCapture,
        lesion_flags: Tensor,
    ) -> ProbeResult:
        """Fit one probe per head on pre-norm ``q`` features.

        Parameters
        ----------
        capture : AttentionCapture
            Must have non-None ``q``.
        lesion_flags : Tensor
            Shape ``(B*nW, n)`` bool.

        Returns
        -------
        ProbeResult
            ``per_block`` has keys ``lp_balanced_accuracy``,
            ``lp_weight_sparsity``, ``lp_margin``, each shape ``(n_heads,)``.
        """
        q = capture.q
        assert q is not None
        # q: (B*nW, nh, n, d_head)
        _bnw, nh, _n, d_head = q.shape
        # Expand lesion flags: (B*nW, nh, n)
        is_lesion = lesion_flags.unsqueeze(1).expand(-1, nh, -1)

        ba = torch.full((nh,), float("nan"), dtype=torch.float32)
        sparsity = torch.full((nh,), float("nan"), dtype=torch.float32)
        margin = torch.full((nh,), float("nan"), dtype=torch.float32)

        # Reshape per head: (N_tokens_per_head, d_head).
        q_per_head = q.permute(1, 0, 2, 3).reshape(nh, -1, d_head)
        flags_per_head = is_lesion.permute(1, 0, 2).reshape(nh, -1)

        for h in range(nh):
            xh = q_per_head[h]
            fh = flags_per_head[h]
            xl = xh[fh]
            xb = xh[~fh]
            m = self._fit_single(
                xl.detach().cpu().numpy().astype(np.float64),
                xb.detach().cpu().numpy().astype(np.float64),
            )
            ba[h] = m.balanced_accuracy
            sparsity[h] = m.weight_sparsity
            margin[h] = m.margin

        return ProbeResult(
            name=self.name,
            per_block={
                "lp_balanced_accuracy": ba,
                "lp_weight_sparsity": sparsity,
                "lp_margin": margin,
            },
            metadata={"n_heads": nh},
        )
