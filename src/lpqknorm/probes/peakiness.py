"""Probe 1 — Feature peakiness.

Measures the coordinate concentration of pre-normalisation Q/K vectors::

    rho(v) = ||v||_inf / (||v||_2 + eps)

Range ``[1/sqrt(d_k), 1]``.  A larger rho means a peakier coordinate distribution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from lpqknorm.probes.base import ProbeResult


if TYPE_CHECKING:
    from torch import Tensor

    from lpqknorm.models.hooks import AttentionCapture


class FeaturePeakiness:
    """Probe 1 — Feature peakiness rho(v) = ||v||_inf / (||v||_2 + eps).

    Parameters
    ----------
    target : ``"q"`` or ``"k"``
        Which pre-norm vector to probe.
    eps : float
        Numerical safety constant.
    """

    def __init__(self, target: Literal["q", "k"], eps: float = 1e-6) -> None:
        self._target = target
        self._eps = eps
        self.name = f"peakiness_{target}"

    @staticmethod
    def compute_value(v: Tensor, eps: float = 1e-6) -> Tensor:
        """Compute rho per token.

        Parameters
        ----------
        v : Tensor
            Shape ``(..., d_head)``.

        Returns
        -------
        Tensor
            Shape ``(...)``.
        """
        return v.abs().amax(dim=-1) / (v.norm(p=2, dim=-1) + eps)  # type: ignore[no-any-return]

    def compute(
        self,
        capture: AttentionCapture,
        lesion_flags: Tensor,
    ) -> ProbeResult:
        """Compute peakiness for all tokens.

        Parameters
        ----------
        capture : AttentionCapture
            Must have non-None ``q`` and ``k``.
        lesion_flags : Tensor
            Shape ``(B*nW, n)`` bool.

        Returns
        -------
        ProbeResult
            ``per_token`` has shape ``(B*nW * nh * n,)``.
        """
        v = capture.q if self._target == "q" else capture.k
        assert v is not None
        # v: (B*nW, nh, n, d_head) → rho: (B*nW, nh, n)
        rho = self.compute_value(v, self._eps)
        # Expand lesion_flags to match (B*nW, nh, n)
        nh = rho.shape[1]
        is_lesion = lesion_flags.unsqueeze(1).expand(-1, nh, -1)
        return ProbeResult(
            name=self.name,
            per_token=rho.reshape(-1),
            metadata={"is_lesion": is_lesion.reshape(-1)},
        )
