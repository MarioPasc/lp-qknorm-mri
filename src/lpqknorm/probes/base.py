"""Base types for mechanistic probes.

Defines the :class:`Probe` protocol and :class:`ProbeResult` dataclass that
every probe module must satisfy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable


if TYPE_CHECKING:
    from torch import Tensor

    from lpqknorm.models.hooks import AttentionCapture


@dataclass(frozen=True)
class ProbeResult:
    """Output of one probe call for one block.

    Parameters
    ----------
    name : str
        Probe identifier (e.g. ``"peakiness_q"``).
    per_token : Tensor or None
        Shape ``(N_tokens,)`` — metrics defined at every token.
    per_query : Tensor or None
        Shape ``(N_queries,)`` — metrics defined per query row.
    per_block : dict[str, Tensor] or None
        Block-level outputs keyed by name.  Used by Probes 7 (per-head
        vectors) and 8 (spectral scalars and eigenvalue vectors), which
        do not reduce to a single per-token or per-query tensor.
    metadata : dict
        Scalar summaries, counters, and auxiliary arrays.
    """

    name: str
    per_token: Tensor | None = None
    per_query: Tensor | None = None
    per_block: dict[str, Tensor] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Probe(Protocol):
    """Protocol every mechanistic probe must satisfy."""

    name: str

    def compute(
        self,
        capture: AttentionCapture,
        lesion_flags: Tensor,
    ) -> ProbeResult:
        """Compute probe values from one block's capture.

        Parameters
        ----------
        capture : AttentionCapture
            Detached, cloned capture from ``AttentionHookRegistry``.
        lesion_flags : Tensor
            Boolean tensor of shape ``(B*nW, n)`` where ``True`` means the
            token is a lesion token.  Aligned with the batch x window layout
            of capture tensors.

        Returns
        -------
        ProbeResult
        """
        ...
