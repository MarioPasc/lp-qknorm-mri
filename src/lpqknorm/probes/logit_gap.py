"""Probe 4 — Lesion-background logit gap.

For each lesion query ``i`` in a window containing lesion tokens ``L_win``
and background tokens ``B_win``::

    Δ_i = max_{j ∈ L_win} s_{ij}  -  median_{j ∈ B_win} s_{ij}

Uses the **full** pre-softmax logits (including relative position bias),
because that is what the softmax sees.

The empirical analogue of the toy-model ``Delta(p) = s^{1-2/p}[1-(s/d_k)^{1/p}]``
with an interior maximum at ``p* > 2``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from lpqknorm.probes.base import ProbeResult
from lpqknorm.probes.tokenization import compute_logits_with_bias


if TYPE_CHECKING:
    from torch import Tensor

    from lpqknorm.models.hooks import AttentionCapture


class LesionBackgroundLogitGap:
    """Probe 4 — Logit gap between best lesion key and median background key.

    Only computed for lesion queries in windows with both lesion and
    background tokens.
    """

    name = "logit_gap"

    def compute(
        self,
        capture: AttentionCapture,
        lesion_flags: Tensor,
    ) -> ProbeResult:
        """Compute logit gap for all lesion queries across windows/heads.

        Parameters
        ----------
        capture : AttentionCapture
            Must have non-None ``logits`` and ``relative_position_bias``.
        lesion_flags : Tensor
            Shape ``(B*nW, n)`` bool.

        Returns
        -------
        ProbeResult
            ``per_query`` has shape ``(N_lesion_queries,)``.
        """
        full_logits = compute_logits_with_bias(capture)
        # full_logits: (B*nW, nh, n, n)
        bnw, nh, _n, _ = full_logits.shape
        results: list[Tensor] = []

        for w in range(bnw):
            lf = lesion_flags[w]  # (n,) bool
            bg = ~lf
            if not lf.any() or not bg.any():
                continue
            for h in range(nh):
                s = full_logits[w, h]  # (n, n)
                lesion_rows = s[lf]  # (n_lesion, n)
                # max over lesion keys
                max_lesion = lesion_rows[:, lf].amax(dim=-1)  # (n_lesion,)
                # median over background keys
                med_bg = lesion_rows[:, bg].median(dim=-1).values  # (n_lesion,)
                results.append(max_lesion - med_bg)

        per_query = torch.cat(results) if results else torch.empty(0)
        return ProbeResult(
            name=self.name,
            per_query=per_query,
            metadata={"n_lesion_queries": int(per_query.numel())},
        )
