"""Phase 3 training components: module, losses, metrics, callbacks, logging."""

from __future__ import annotations

from lpqknorm.training.callbacks import (
    ArtefactDirectoryCallback,
    AttentionSummaryCallback,
    GradientNormCallback,
    ManifestCallback,
    PerPatientMetricsCallback,
    ProbeCallback,
    RunManifest,
)
from lpqknorm.training.logging import StructuredLogger
from lpqknorm.training.losses import CompoundSegLoss
from lpqknorm.training.metrics import (
    LesionDetectionResult,
    dice_score,
    hd95,
    iou_score,
    lesion_wise_detection,
)
from lpqknorm.training.module import (
    LpSegmentationModule,
    ModelConfig,
    TrainingConfig,
)


__all__ = [
    "ArtefactDirectoryCallback",
    "AttentionSummaryCallback",
    "CompoundSegLoss",
    "GradientNormCallback",
    "LesionDetectionResult",
    "LpSegmentationModule",
    "ManifestCallback",
    "ModelConfig",
    "PerPatientMetricsCallback",
    "ProbeCallback",
    "RunManifest",
    "StructuredLogger",
    "TrainingConfig",
    "dice_score",
    "hd95",
    "iou_score",
    "lesion_wise_detection",
]
