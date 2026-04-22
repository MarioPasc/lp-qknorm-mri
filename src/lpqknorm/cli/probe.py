"""CLI entry point for post-hoc probe extraction.

Runs all five mechanistic probes on a saved checkpoint, writing HDF5
output to the run's ``probes/`` directory.

Usage::

    lpqknorm-probe --checkpoint path/to/best_val_dice.ckpt \\
                   --output-dir path/to/probes/ \\
                   --epoch-tag best_dice
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch.serialization as _serialization

from lpqknorm.data.datamodule import MockAtlasDataModule, MockDataConfig
from lpqknorm.models.lp_qknorm import LpQKNormConfig
from lpqknorm.probes import (
    AttentionEntropy,
    AttentionMaskIoU,
    FeaturePeakiness,
    LesionAttentionMass,
    LesionBackgroundLogitGap,
    LinearProbe,
    ProbeRecorder,
    SpatialLocalizationError,
    SpectralProbe,
)
from lpqknorm.training.module import LpSegmentationModule, ModelConfig, TrainingConfig

# PyTorch >= 2.6 defaults torch.load(weights_only=True); LightningModule
# hparams for LpSegmentationModule include these dataclasses. Allow-list
# them so load_from_checkpoint succeeds without insecure weights_only=False.
_serialization.add_safe_globals([ModelConfig, TrainingConfig, LpQKNormConfig])


logger = logging.getLogger(__name__)


def main() -> None:
    """Entry point for ``lpqknorm-probe``."""
    parser = argparse.ArgumentParser(description="Post-hoc probe extraction")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to .ckpt file"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Probe output directory"
    )
    parser.add_argument(
        "--epoch-tag",
        type=str,
        default="posthoc",
        help="Label for the HDF5 file",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device (cpu or cuda)"
    )
    parser.add_argument(
        "--n-probe-samples",
        type=int,
        default=32,
        help="Max validation samples",
    )
    args = parser.parse_args()

    # Load model from checkpoint. strict=False tolerates the optional
    # loss_fn.bce.pos_weight buffer, which is present only when training
    # received a DataModule-derived pos_weight and is reconstructed as
    # None here (probe runs on the model weights alone; the loss fn is
    # irrelevant).
    module = LpSegmentationModule.load_from_checkpoint(
        args.checkpoint, map_location=args.device, strict=False
    )
    model = module.model
    model.eval()

    # Build fixed probe loader (mock for now; real ATLAS in Phase 1)
    dm = MockAtlasDataModule(MockDataConfig(n_val=args.n_probe_samples))
    dm.setup("fit")

    recorder = ProbeRecorder(
        probes=[
            FeaturePeakiness("q"),
            FeaturePeakiness("k"),
            AttentionEntropy(),
            LesionAttentionMass(),
            LesionBackgroundLogitGap(),
            AttentionMaskIoU(),
            SpatialLocalizationError(),
            LinearProbe(),
            SpectralProbe(),
        ],
        output_dir=Path(args.output_dir),
        n_probe_samples=args.n_probe_samples,
    )

    out_path = recorder.run(
        model=model,
        dataloader=dm.val_dataloader(),
        epoch_tag=args.epoch_tag,
        device=args.device,
    )
    logger.info("Probes written to %s", out_path)


if __name__ == "__main__":
    main()
