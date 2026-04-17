"""CLI entry point for post-hoc activation patching.

Usage::

    lpqknorm-patching \
        --source-ckpt path/to/p_star/best_small_recall.ckpt \
        --target-ckpt path/to/p2/best_small_recall.ckpt \
        --output-dir  path/to/probes/ \
        [--variants q k qk qhat_khat logits] \
        [--blocks 0 1] \
        [--direction denoising] \
        [--n-probe-samples 32]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from lpqknorm.data.datamodule import MockAtlasDataModule, MockDataConfig
from lpqknorm.probes.patching import (
    VALID_VARIANTS,
    ActivationPatcher,
    PatchingConfig,
)


logger = logging.getLogger(__name__)


def main() -> None:
    """Entry point for ``lpqknorm-patching``."""
    parser = argparse.ArgumentParser(description="Post-hoc activation patching")
    parser.add_argument("--source-ckpt", type=str, required=True)
    parser.add_argument("--target-ckpt", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--variants",
        type=str,
        nargs="+",
        default=list(VALID_VARIANTS),
        choices=list(VALID_VARIANTS),
    )
    parser.add_argument("--blocks", type=int, nargs="+", default=[0, 1])
    parser.add_argument(
        "--direction",
        type=str,
        choices=["denoising", "noising"],
        default="denoising",
    )
    parser.add_argument("--stage", type=int, default=0)
    parser.add_argument("--n-probe-samples", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    cfg = PatchingConfig(
        source_checkpoint=Path(args.source_ckpt),
        target_checkpoint=Path(args.target_ckpt),
        stage=args.stage,
        blocks=tuple(args.blocks),
        variants=tuple(args.variants),
        direction=args.direction,
        n_probe_samples=args.n_probe_samples,
    )

    # Mock loader for now — the training scripts will pass the real
    # fixed probe loader when this CLI is invoked from a run orchestrator.
    dm = MockAtlasDataModule(MockDataConfig(n_val=args.n_probe_samples))
    dm.setup("fit")

    patcher = ActivationPatcher(cfg)
    out = patcher.run(
        dm.val_dataloader(), output_dir=Path(args.output_dir), device=args.device
    )
    logger.info("Activation patching written to %s", out)


if __name__ == "__main__":
    main()
