"""Phase 4 mechanistic probes for Lp-QKNorm attention analysis."""

from __future__ import annotations

from lpqknorm.probes.attention_iou import AttentionMaskIoU
from lpqknorm.probes.base import Probe, ProbeResult
from lpqknorm.probes.entropy import AttentionEntropy
from lpqknorm.probes.lesion_mass import LesionAttentionMass
from lpqknorm.probes.logit_gap import LesionBackgroundLogitGap
from lpqknorm.probes.peakiness import FeaturePeakiness
from lpqknorm.probes.recorder import ProbeRecorder
from lpqknorm.probes.tokenization import (
    compute_logits_with_bias,
    mask_to_token_flags,
    window_partition_flags,
)


__all__ = [
    "AttentionEntropy",
    "AttentionMaskIoU",
    "FeaturePeakiness",
    "LesionAttentionMass",
    "LesionBackgroundLogitGap",
    "Probe",
    "ProbeRecorder",
    "ProbeResult",
    "compute_logits_with_bias",
    "mask_to_token_flags",
    "window_partition_flags",
]
