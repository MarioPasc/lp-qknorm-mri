"""Phase 4 mechanistic probes for Lp-QKNorm attention analysis."""

from __future__ import annotations

from lpqknorm.probes.attention_iou import AttentionMaskIoU
from lpqknorm.probes.attention_maps import (
    attention_rollout,
    overlay_figure,
    reconstruct_query_heatmap,
)
from lpqknorm.probes.base import Probe, ProbeResult
from lpqknorm.probes.entropy import AttentionEntropy
from lpqknorm.probes.lesion_mass import LesionAttentionMass
from lpqknorm.probes.linear_probe import LinearProbe, LinearProbeMetrics
from lpqknorm.probes.logit_gap import LesionBackgroundLogitGap
from lpqknorm.probes.patching import (
    ActivationPatcher,
    PatchingConfig,
    run_patching_sweep,
)
from lpqknorm.probes.peakiness import FeaturePeakiness
from lpqknorm.probes.recorder import ProbeRecorder
from lpqknorm.probes.spatial_loc_error import SpatialLocalizationError
from lpqknorm.probes.spectral import SpectralProbe
from lpqknorm.probes.tokenization import (
    compute_logits_with_bias,
    mask_to_token_flags,
    window_boundary_distance,
    window_partition_flags,
)


__all__ = [
    "ActivationPatcher",
    "AttentionEntropy",
    "AttentionMaskIoU",
    "FeaturePeakiness",
    "LesionAttentionMass",
    "LesionBackgroundLogitGap",
    "LinearProbe",
    "LinearProbeMetrics",
    "PatchingConfig",
    "Probe",
    "ProbeRecorder",
    "ProbeResult",
    "SpatialLocalizationError",
    "SpectralProbe",
    "attention_rollout",
    "compute_logits_with_bias",
    "mask_to_token_flags",
    "overlay_figure",
    "reconstruct_query_heatmap",
    "run_patching_sweep",
    "window_boundary_distance",
    "window_partition_flags",
]
