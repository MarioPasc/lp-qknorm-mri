"""Probe 8 — Spectral / participation-ratio probe.

Population-level complement to Probe 1 (per-vector peakiness).  Pools
pre-norm ``q`` features (across heads) into a ``(N, d_k)`` matrix and
reports on the spectrum of its empirical covariance:

    Sigma = (1 / (N - 1)) (X - X̄)^T (X - X̄)
    lambda_1 ≥ ... ≥ lambda_{d_k}
    PR = (Σ lambda_i)^2 / Σ lambda_i^2          ∈ [1, d_k]
    SR = ||X||_F^2 / ||X||_2^2
    ~lambda_i = lambda_i / Σ_j lambda_j

Reported for the lesion and background token pools separately.

References
----------
- Gao et al.  *A theory of multineuronal dimensionality, dynamics and
  measurement*.  bioRxiv 2017.  doi:10.1101/214262.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from lpqknorm.probes.base import ProbeResult


if TYPE_CHECKING:
    from torch import Tensor

    from lpqknorm.models.hooks import AttentionCapture


logger = logging.getLogger(__name__)


class SpectralProbe:
    """Probe 8 — participation ratio, stable rank, eigenvalue spectrum."""

    name = "spectral"

    def __init__(self, min_samples: int = 8) -> None:
        self._min_samples = min_samples

    # ------------------------------------------------------------------
    # Primitives (also used by unit tests)
    # ------------------------------------------------------------------

    @staticmethod
    def _participation_ratio(x: Tensor, eps: float = 1e-12) -> float:
        """Participation ratio of the centred feature matrix ``x``.

        Parameters
        ----------
        x : Tensor
            Shape ``(N, d)``.  Need not be centred — this function centres
            internally.
        """
        if x.shape[0] < 2:
            return float("nan")
        xc = x - x.mean(dim=0, keepdim=True)
        cov = xc.T @ xc / (xc.shape[0] - 1)
        eigvals = torch.linalg.eigvalsh(cov).clamp(min=0.0)
        num = float(eigvals.sum().item()) ** 2
        den = float((eigvals * eigvals).sum().item()) + eps
        return num / den

    @staticmethod
    def _stable_rank(x: Tensor, eps: float = 1e-12) -> float:
        """Stable rank ``||X||_F^2 / ||X||_2^2``."""
        if x.shape[0] < 2:
            return float("nan")
        fro_sq = float((x * x).sum().item())
        spec = float(torch.linalg.matrix_norm(x, ord=2).item())
        return fro_sq / (spec * spec + eps)

    @staticmethod
    def _eigenvalues(x: Tensor) -> Tensor:
        """Sorted (descending) non-negative eigenvalues of the covariance.

        Returns zeros-of-length-d_k if ``x`` has fewer than 2 rows.
        """
        d = x.shape[1]
        if x.shape[0] < 2:
            return torch.zeros(d, dtype=torch.float32)
        xc = x - x.mean(dim=0, keepdim=True)
        cov = xc.T @ xc / (xc.shape[0] - 1)
        eigvals = torch.linalg.eigvalsh(cov).clamp(min=0.0)
        return torch.flip(eigvals, dims=[0]).to(torch.float32)

    # ------------------------------------------------------------------
    # Probe protocol
    # ------------------------------------------------------------------

    def compute(
        self,
        capture: AttentionCapture,
        lesion_flags: Tensor,
    ) -> ProbeResult:
        """Compute spectral statistics on pooled pre-norm ``q`` features.

        Parameters
        ----------
        capture : AttentionCapture
            Must have non-None ``q``.
        lesion_flags : Tensor
            Shape ``(B*nW, n)`` bool.

        Returns
        -------
        ProbeResult
            ``per_block`` has keys:
              - ``pr_lesion`` (scalar), ``pr_background`` (scalar)
              - ``stable_rank_lesion`` (scalar), ``stable_rank_background``
              - ``eigenvalues_lesion`` ``(d_k,)``,
                ``eigenvalues_background`` ``(d_k,)``
        """
        q = capture.q
        assert q is not None
        _bnw, nh, _n, d_head = q.shape
        is_lesion = lesion_flags.unsqueeze(1).expand(-1, nh, -1)

        # Pool all tokens across windows and heads into (N, d_head).
        x_all = q.reshape(-1, d_head).to(torch.float32)
        flags_all = is_lesion.reshape(-1)

        x_lesion = x_all[flags_all]
        x_bg = x_all[~flags_all]

        if x_lesion.shape[0] < self._min_samples:
            pr_l = float("nan")
            sr_l = float("nan")
            ev_l = torch.zeros(d_head, dtype=torch.float32)
        else:
            pr_l = self._participation_ratio(x_lesion)
            sr_l = self._stable_rank(x_lesion)
            ev_l = self._eigenvalues(x_lesion)

        if x_bg.shape[0] < self._min_samples:
            pr_b = float("nan")
            sr_b = float("nan")
            ev_b = torch.zeros(d_head, dtype=torch.float32)
        else:
            pr_b = self._participation_ratio(x_bg)
            sr_b = self._stable_rank(x_bg)
            ev_b = self._eigenvalues(x_bg)

        return ProbeResult(
            name=self.name,
            per_block={
                "pr_lesion": torch.tensor(pr_l, dtype=torch.float32),
                "pr_background": torch.tensor(pr_b, dtype=torch.float32),
                "stable_rank_lesion": torch.tensor(sr_l, dtype=torch.float32),
                "stable_rank_background": torch.tensor(sr_b, dtype=torch.float32),
                "eigenvalues_lesion": ev_l,
                "eigenvalues_background": ev_b,
            },
            metadata={
                "n_lesion": int(x_lesion.shape[0]),
                "n_background": int(x_bg.shape[0]),
                "d_k": int(d_head),
            },
        )
