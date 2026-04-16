"""Probe 2 — Per-query attention entropy.

Measures the sharpness of attention distributions::

    H_i = -Σ_{j=1}^{W²} A_{ij} log A_{ij}

Range ``[0, log(W²)]``.  Lower entropy means sharper (more concentrated)
attention.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lpqknorm.probes.base import ProbeResult


if TYPE_CHECKING:
    from torch import Tensor

    from lpqknorm.models.hooks import AttentionCapture


class AttentionEntropy:
    """Probe 2 — Per-query attention entropy.

    Parameters
    ----------
    eps : float
        Clamp floor for log stability.
    """

    name = "entropy"

    def __init__(self, eps: float = 1e-9) -> None:
        self._eps = eps

    @staticmethod
    def compute_value(attn_row: Tensor, eps: float = 1e-9) -> Tensor:
        """Compute entropy of attention rows.

        Parameters
        ----------
        attn_row : Tensor
            Shape ``(..., n)`` — attention distribution (sums to 1).

        Returns
        -------
        Tensor
            Shape ``(...)``.
        """
        safe = attn_row.clamp(min=eps)
        return -(safe * safe.log()).sum(dim=-1)

    def compute(
        self,
        capture: AttentionCapture,
        lesion_flags: Tensor,
    ) -> ProbeResult:
        """Compute entropy for all queries.

        Parameters
        ----------
        capture : AttentionCapture
            Must have non-None ``attention``.
        lesion_flags : Tensor
            Shape ``(B*nW, n)`` bool.

        Returns
        -------
        ProbeResult
            ``per_query`` has shape ``(B*nW * nh * n,)``.
        """
        attn = capture.attention
        assert attn is not None
        # attn: (B*nW, nh, n, n) → H: (B*nW, nh, n)
        h_vals = self.compute_value(attn, self._eps)
        nh = h_vals.shape[1]
        is_lesion = lesion_flags.unsqueeze(1).expand(-1, nh, -1)
        return ProbeResult(
            name=self.name,
            per_query=h_vals.reshape(-1),
            metadata={"is_lesion": is_lesion.reshape(-1)},
        )
