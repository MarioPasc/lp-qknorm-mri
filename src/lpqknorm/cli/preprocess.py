"""CLI entry point: preprocess raw datasets into standardized HDF5 format."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from lpqknorm.data.converter import (
    PreprocessConfig,
    write_standardized_h5,
)


if TYPE_CHECKING:
    from collections.abc import Iterator

    from lpqknorm.data.converter import SubjectVolume
from lpqknorm.data.converters import get_converter
from lpqknorm.data.converters.brats_men import extract_patient_id
from lpqknorm.utils.seeding import set_global_seed


logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess a raw dataset into standardized HDF5 format.",
        prog="lpqknorm-preprocess",
    )
    parser.add_argument(
        "--converter",
        type=str,
        default="brats_men",
        help="Converter name from the registry (default: brats_men)",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        required=True,
        help="Path to raw dataset directory",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/media/mpascual/Sandisk2TB/research/lpqknorm_mri/data"),
        help="Output directory for HDF5 file and QC artifacts",
    )
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260216)
    parser.add_argument(
        "--target-spacing",
        type=float,
        nargs=3,
        default=[1.0, 1.0, 1.0],
        help="Target spacing in mm (D H W)",
    )
    parser.add_argument(
        "--in-plane-size",
        type=int,
        nargs=2,
        default=[224, 224],
        help="Target in-plane size (H W)",
    )
    parser.add_argument(
        "--min-lesion-voxels",
        type=int,
        default=10,
        help="Minimum foreground voxels for has_lesion=True",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only discover subjects, do not write HDF5",
    )
    return parser


def main() -> None:
    """Entry point for lpqknorm-preprocess."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    args = _build_parser().parse_args()
    set_global_seed(args.seed)

    converter = get_converter(args.converter)
    logger.info("Using converter: %s", args.converter)

    records = converter.discover_subjects(args.raw_root)
    logger.info("Discovered %d subjects", len(records))

    if args.dry_run:
        logger.info("Dry run — stopping after discovery")
        for r in records[:5]:
            logger.info("  %s", r.subject_id)
        logger.info("  ... (%d total)", len(records))
        return

    cfg = PreprocessConfig(
        target_spacing_mm=tuple(args.target_spacing),
        in_plane_size=tuple(args.in_plane_size),
        min_lesion_voxels_per_slice=args.min_lesion_voxels,
        seed=args.seed,
    )

    # Determine patient ID extractor based on converter
    pid_extractor = extract_patient_id if args.converter == "brats_men" else None

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{converter.info.name}.h5"

    t0 = time.perf_counter()

    # Stream subjects one at a time (memory-efficient)
    def _subject_generator() -> Iterator[SubjectVolume]:
        n_ok, n_skip = 0, 0
        for i, record in enumerate(records):
            logger.info(
                "[%d/%d] Loading %s ...",
                i + 1,
                len(records),
                record.subject_id,
            )
            try:
                sv = converter.load_subject(record, cfg)
            except Exception:
                logger.exception("Failed to load %s", record.subject_id)
                n_skip += 1
                continue

            if sv is None:
                logger.warning("Excluded %s (empty mask)", record.subject_id)
                n_skip += 1
                continue

            n_ok += 1
            yield sv

        logger.info("Loading complete: %d ok, %d skipped", n_ok, n_skip)

    write_standardized_h5(
        subjects=_subject_generator(),
        info=converter.info,
        cfg=cfg,
        out_path=out_path,
        n_folds=args.n_folds,
        seed=args.seed,
        expected_n_subjects=len(records),
        patient_id_extractor=pid_extractor,
    )

    elapsed = time.perf_counter() - t0
    logger.info("Preprocessing complete in %.1f s", elapsed)

    # Write success sentinel
    (out_dir / "_SUCCESS").touch()
    logger.info("Wrote _SUCCESS sentinel to %s", out_dir)


if __name__ == "__main__":
    main()
