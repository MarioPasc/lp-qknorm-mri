"""Probe 3 — Lesion attention mass.

For lesion queries ``i`` in a window containing lesion tokens ``L_win``::

    M_i = Σ_{j ∈ L_win(i)} A_{ij}

Range ``[0, 1]``.  Predicted to increase with ``p`` on the small-lesion
stratum.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from lpqknorm.probes.base import ProbeResult


if TYPE_CHECKING:
    from torch import Tensor

    from lpqknorm.models.hooks import AttentionCapture


class LesionAttentionMass:
    """Probe 3 — Lesion attention mass on lesion queries.

    Only computed for lesion queries in windows that contain at least one
    lesion key token.
    """

    name = "lesion_mass"

    @staticmethod
    def compute_per_query(
        attn: Tensor,
        lesion_mask: Tensor,
    ) -> Tensor:
        """Compute M_i for lesion queries in one attention matrix.

        Parameters
        ----------
        attn : Tensor
            Shape ``(n, n)`` — attention matrix for one head/window.
        lesion_mask : Tensor
            Shape ``(n,)`` bool — per-token lesion flags for this window.

        Returns
        -------
        Tensor
            Shape ``(n_lesion,)`` — mass values for lesion queries.
            Empty if no lesion tokens.
        """
        if not lesion_mask.any():
            return torch.empty(0, device=attn.device)
        lesion_rows = attn[lesion_mask]  # (n_lesion, n)
        return lesion_rows[:, lesion_mask].sum(dim=-1)  # (n_lesion,)

    def compute(
        self,
        capture: AttentionCapture,
        lesion_flags: Tensor,
    ) -> ProbeResult:
        """Compute lesion mass for all lesion queries across windows/heads.

        Parameters
        ----------
        capture : AttentionCapture
            Must have non-None ``attention``.
        lesion_flags : Tensor
            Shape ``(B*nW, n)`` bool.

        Returns
        -------
        ProbeResult
            ``per_query`` has shape ``(N_lesion_queries,)`` — only lesion
            queries in lesion-containing windows.
        """
        attn = capture.attention
        assert attn is not None
        # attn: (B*nW, nh, n, n)
        bnw, nh, _n, _ = attn.shape
        results: list[Tensor] = []

        for w in range(bnw):
            lf = lesion_flags[w]  # (n,) bool
            if not lf.any():
                continue
            for h in range(nh):
                m = self.compute_per_query(attn[w, h], lf)
                results.append(m)

        per_query = torch.cat(results) if results else torch.empty(0)
        return ProbeResult(
            name=self.name,
            per_query=per_query,
            metadata={"n_lesion_queries": int(per_query.numel())},
        )
