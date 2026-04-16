"""Probe 5 — Attention-mask IoU.

For each lesion query ``i``, binarise attention by the top-``k`` tokens
where ``k = |L_win(i)|``, then compute IoU with the lesion window mask::

    T_i = top-k binary attention mask
    IoU_i = |T_i ∩ M| / |T_i U M|

Range ``[0, 1]``.  Predicted to increase with ``p``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from lpqknorm.probes.base import ProbeResult


if TYPE_CHECKING:
    from torch import Tensor

    from lpqknorm.models.hooks import AttentionCapture


class AttentionMaskIoU:
    """Probe 5 — IoU between top-k binarised attention and lesion mask.

    Only computed for lesion queries in lesion-containing windows.
    """

    name = "attention_iou"

    def compute(
        self,
        capture: AttentionCapture,
        lesion_flags: Tensor,
    ) -> ProbeResult:
        """Compute attention-mask IoU for all lesion queries.

        Parameters
        ----------
        capture : AttentionCapture
            Must have non-None ``attention``.
        lesion_flags : Tensor
            Shape ``(B*nW, n)`` bool.

        Returns
        -------
        ProbeResult
            ``per_query`` has shape ``(N_lesion_queries,)``.
        """
        attn = capture.attention
        assert attn is not None
        # attn: (B*nW, nh, n, n)
        bnw, nh, _n, _ = attn.shape
        results: list[Tensor] = []

        for w in range(bnw):
            lf = lesion_flags[w]  # (n,) bool
            k = int(lf.sum().item())
            if k == 0:
                continue
            for h in range(nh):
                a = attn[w, h]  # (n, n)
                lesion_rows = a[lf]  # (k, n)
                # Top-k binarisation per lesion query
                _, topk_idx = lesion_rows.topk(k, dim=-1)  # (k, k)
                t_mask = torch.zeros_like(lesion_rows, dtype=torch.bool)
                t_mask.scatter_(1, topk_idx, True)  # (k, n) binary
                m_mask = lf.unsqueeze(0).expand(k, -1)  # (k, n) binary
                inter = (t_mask & m_mask).sum(dim=-1).float()
                union = (t_mask | m_mask).sum(dim=-1).float()
                iou = inter / union.clamp(min=1.0)
                results.append(iou)

        per_query = torch.cat(results) if results else torch.empty(0)
        return ProbeResult(
            name=self.name,
            per_query=per_query,
            metadata={"n_lesion_queries": int(per_query.numel())},
        )
