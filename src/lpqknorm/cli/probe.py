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

from lpqknorm.data.datamodule import MockAtlasDataModule, MockDataConfig
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
from lpqknorm.training.module import LpSegmentationModule


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

    # Load model from checkpoint
    module = LpSegmentationModule.load_from_checkpoint(
        args.checkpoint, map_location=args.device
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
