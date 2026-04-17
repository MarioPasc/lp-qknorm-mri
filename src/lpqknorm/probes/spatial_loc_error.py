"""Probe 6 — Spatial localisation error.

Peak-based complement to Probe 5 (attention-mask IoU).  For each lesion
query ``i`` let

    j*_i   = argmax_j A_{ij}                       (attention argmax)
    mu_i   = centroid of lesion tokens in the same window
    SLE_i  = || coord(j*_i) - mu_i ||_2

with ``coord(j)`` in intra-window ``(y, x)`` token coordinates.

Range: ``[0, W * sqrt(2)]``.  Lower values mean the attention peak sits
on the lesion cluster.  Probe 5 measures *coverage* (set overlap); Probe
6 measures *peakedness location*.  The two can disagree (e.g., diffuse
attention centred on the lesion yields high IoU and high SLE).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from lpqknorm.probes.base import ProbeResult


if TYPE_CHECKING:
    from torch import Tensor

    from lpqknorm.models.hooks import AttentionCapture


class SpatialLocalizationError:
    """Probe 6 — distance between attention argmax and lesion centroid."""

    name = "spatial_localization_error"

    def __init__(self, window_size: int = 7) -> None:
        self._window_size = window_size

    @staticmethod
    def compute_per_query(
        attn_row: Tensor,
        lesion_mask: Tensor,
        window_size: int,
    ) -> Tensor:
        """Compute SLE for a single-head attention row and lesion mask.

        Parameters
        ----------
        attn_row : Tensor
            Shape ``(k, W²)`` — attention rows for ``k`` lesion queries.
            May also be ``(1, W²)`` or ``(W²,)`` for a single query.
        lesion_mask : Tensor
            Shape ``(W²,)`` bool — per-token lesion flags in the window.
        window_size : int
            Window side length ``W``.

        Returns
        -------
        Tensor
            Shape ``(k,)`` — SLE values (float32).
        """
        if attn_row.ndim == 1:
            attn_row = attn_row.unsqueeze(0)

        w = window_size
        device = attn_row.device

        # Build (W², 2) table of intra-window (y, x) coordinates.
        ys = torch.arange(w, device=device).view(w, 1).expand(w, w).reshape(-1)
        xs = torch.arange(w, device=device).view(1, w).expand(w, w).reshape(-1)
        coords = torch.stack([ys, xs], dim=-1).float()  # (W², 2)

        # Centroid of lesion tokens in this window.
        lesion_coords = coords[lesion_mask]  # (n_lesion, 2)
        centroid = lesion_coords.mean(dim=0)  # (2,)

        # Argmax coordinate per query.
        argmax = attn_row.argmax(dim=-1)  # (k,)
        peak_coords = coords[argmax]  # (k, 2)

        sle: Tensor = (peak_coords - centroid).norm(p=2, dim=-1)  # (k,)
        return sle

    def compute(
        self,
        capture: AttentionCapture,
        lesion_flags: Tensor,
    ) -> ProbeResult:
        """Compute SLE for all lesion queries.

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
        bnw, nh, n, _ = attn.shape
        w = self._window_size
        assert n == w * w, f"attention n={n} does not match window_size²={w * w}"

        results: list[Tensor] = []
        for wi in range(bnw):
            lf = lesion_flags[wi]
            if not lf.any():
                continue
            for h in range(nh):
                a = attn[wi, h]  # (n, n)
                lesion_rows = a[lf]
                sle = self.compute_per_query(lesion_rows, lf, w)
                results.append(sle)

        per_query = torch.cat(results) if results else torch.empty(0)
        return ProbeResult(
            name=self.name,
            per_query=per_query,
            metadata={"n_lesion_queries": int(per_query.numel())},
        )
