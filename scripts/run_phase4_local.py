"""Run Phase 4 (mechanistic probes + activation patching) over the
``p_sweep_v1`` results, in-process, on the local RTX 4060.

The Phase-4 CLI scripts ship with ``MockAtlasDataModule`` wiring (Phase-2
test convenience).  This driver instead loads each run's checkpoint
together with the production :class:`SegmentationDataModule` pointed at
the local BraTS-MEN HDF5, runs :class:`ProbeRecorder` on the fixed probe
loader, and then runs :class:`ActivationPatcher` between
``p_star → p = 2`` (one swap per fold) at stage 0.

Resource envelope (RTX 4060 8 GB, bf16-mixed forward, no_grad):
- 18 checkpoints × ~10 s probe ≈ 3 min
- 3 folds × ~60 s patching ≈ 3 min
- HDF5 first-touch + worker startup overhead bumps the total to ~10–15 min.

Usage::

    ~/.conda/envs/lpqknorm/bin/python scripts/run_phase4_local.py \\
        --results-root /media/mpascual/Sandisk2TB/research/lpqknorm_mri/results \\
        --h5-path     /media/mpascual/Sandisk2TB/research/lpqknorm_mri/data/brats_men.h5 \\
        --p-star 4.0 --p-baseline 2.0 \\
        --device cuda --n-probe-samples 32

Outputs (mirrors the schema documented in ``docs/phase_04_probes.md``)::

    p=<p>/fold=<f>/seed=<seed>/probes/epoch_best_dice.h5
    p=<p_star>/fold=<f>/seed=<seed>/probes/patching_best_dice.h5

The driver is idempotent: it skips a (run, artefact) pair whose output
file already exists unless ``--overwrite`` is passed.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.serialization as _serialization
from torch.utils.data import DataLoader

from lpqknorm.data.datamodule import SegmentationDataModule


class _BinaryMaskLoader:
    """Iterate ``base_loader`` yielding batches with whole-tumor binary mask.

    The BraTS-MEN HDF5 stores the mask as ``(B, 3, H, W)`` (NET_NCR / SNFH /
    ET).  The probe recorder and ``mask_to_token_flags`` expect a binary
    mask shaped ``(B, 1, H, W)``.  We OR the three classes together to
    obtain the whole-tumor mask, which is the lesion definition used by
    the headline ``lesion_recall`` metric.
    """

    def __init__(self, base_loader: DataLoader) -> None:
        self._base = base_loader

    def __iter__(self):
        for batch in self._base:
            mask = batch["mask"]
            if mask.dim() == 4 and mask.shape[1] > 1:
                mask = (mask.sum(dim=1, keepdim=True) > 0).to(mask.dtype)
            yield {**batch, "mask": mask}

    def __len__(self) -> int:
        return len(self._base)


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
from lpqknorm.probes.patching import ActivationPatcher, PatchingConfig
from lpqknorm.training.module import (
    LpSegmentationModule,
    ModelConfig,
    TrainingConfig,
)


# Allow LightningModule hparams under torch.load(weights_only=True).
_serialization.add_safe_globals([ModelConfig, TrainingConfig, LpQKNormConfig])

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("phase4")


P_VALUES = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
FOLDS = (0, 1, 2)


@dataclass(frozen=True)
class RunSpec:
    p: float
    fold: int
    run_dir: Path

    @property
    def ckpt_path(self) -> Path:
        return self.run_dir / "checkpoints" / "best_val_dice.ckpt"

    @property
    def small_recall_ckpt(self) -> Path:
        return self.run_dir / "checkpoints" / "best_small_recall.ckpt"

    @property
    def probes_dir(self) -> Path:
        return self.run_dir / "probes"


def discover_runs(results_root: Path, seed: int) -> list[RunSpec]:
    """Enumerate every (p, fold) run directory under *results_root*."""
    out: list[RunSpec] = []
    for p in P_VALUES:
        for f in FOLDS:
            run = results_root / f"p={p}" / f"fold={f}" / f"seed={seed}"
            if not run.exists():
                logger.warning("Missing run directory: %s", run)
                continue
            out.append(RunSpec(p=p, fold=f, run_dir=run))
    return out


def build_probe_loader(
    h5_path: Path,
    fold: int,
    n_probe_samples: int,
    num_workers: int,
) -> SegmentationDataModule:
    """Build a deterministic probe loader from the validation split.

    The DataModule's val loader is ``shuffle=False`` and lesion-only; we
    take the first ``n_probe_samples`` slices via the recorder's own
    truncation logic.
    """
    dm = SegmentationDataModule(
        h5_path=h5_path,
        fold=fold,
        spatial_mode="2d",
        batch_size=4,
        num_workers=num_workers,
        augment=False,
        lesion_only=True,
    )
    dm.setup("fit")
    return dm


def run_probes_on_run(
    run: RunSpec,
    h5_path: Path,
    n_probe_samples: int,
    device: str,
    num_workers: int,
    overwrite: bool,
) -> Path | None:
    """Run probes 1–8 on one (p, fold) checkpoint."""
    out_path = run.probes_dir / "epoch_best_dice.h5"
    if out_path.exists() and not overwrite:
        logger.info("[probe] SKIP existing %s", out_path)
        return out_path
    if not run.ckpt_path.exists():
        logger.error("[probe] missing ckpt: %s", run.ckpt_path)
        return None

    t0 = time.perf_counter()
    module = LpSegmentationModule.load_from_checkpoint(
        run.ckpt_path, map_location=device, strict=False
    )
    model = module.model.eval().to(device)

    dm = build_probe_loader(h5_path, run.fold, n_probe_samples, num_workers)

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
        output_dir=run.probes_dir,
        n_probe_samples=n_probe_samples,
    )
    out = recorder.run(
        model=model,
        dataloader=_BinaryMaskLoader(dm.val_dataloader()),
        epoch_tag="best_dice",
        device=device,
    )
    dt = time.perf_counter() - t0
    logger.info("[probe] OK p=%s fold=%d in %.1fs -> %s", run.p, run.fold, dt, out)
    # Free GPU memory before next run.
    del module, model, dm
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return out


def run_patching_for_fold(
    fold: int,
    run_index: dict[tuple[float, int], RunSpec],
    p_star: float,
    p_baseline: float,
    h5_path: Path,
    n_probe_samples: int,
    device: str,
    num_workers: int,
    overwrite: bool,
) -> Path | None:
    """Run activation patching p_star → p_baseline for one fold."""
    src = run_index[(p_star, fold)]
    tgt = run_index[(p_baseline, fold)]
    src_ckpt = src.small_recall_ckpt
    tgt_ckpt = tgt.small_recall_ckpt
    out_path = src.probes_dir / "patching_best_dice.h5"
    if out_path.exists() and not overwrite:
        logger.info("[patching] SKIP existing %s", out_path)
        return out_path
    if not (src_ckpt.exists() and tgt_ckpt.exists()):
        logger.error("[patching] missing ckpt(s): src=%s tgt=%s", src_ckpt, tgt_ckpt)
        return None

    t0 = time.perf_counter()
    cfg = PatchingConfig(
        source_checkpoint=src_ckpt,
        target_checkpoint=tgt_ckpt,
        stage=0,
        blocks=(0, 1),
        variants=("q", "k", "qk", "qhat_khat", "logits"),
        direction="denoising",
        n_probe_samples=n_probe_samples,
    )
    dm = build_probe_loader(h5_path, fold, n_probe_samples, num_workers)
    patcher = ActivationPatcher(cfg)
    out = patcher.run(
        _BinaryMaskLoader(dm.val_dataloader()),
        output_dir=src.probes_dir,
        device=device,
    )
    dt = time.perf_counter() - t0
    logger.info(
        "[patching] OK fold=%d p=%s->%s in %.1fs -> %s",
        fold,
        p_star,
        p_baseline,
        dt,
        out,
    )
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root", type=Path, required=True, help="p_sweep_v1 results root"
    )
    parser.add_argument("--h5-path", type=Path, required=True, help="BraTS-MEN HDF5")
    parser.add_argument("--seed", type=int, default=20260216)
    parser.add_argument("--n-probe-samples", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--p-star",
        type=float,
        default=4.0,
        help="Source p for activation patching (best small-recall in p_sweep_v1).",
    )
    parser.add_argument(
        "--p-baseline",
        type=float,
        default=2.0,
        help="Target p for activation patching (Henry et al. QKNorm baseline).",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--probes-only", action="store_true", help="Skip the patching pass."
    )
    parser.add_argument(
        "--patching-only", action="store_true", help="Skip the probe pass."
    )
    args = parser.parse_args()

    if not args.h5_path.exists():
        logger.error("HDF5 not found: %s", args.h5_path)
        return 1
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU.")
        args.device = "cpu"

    runs = discover_runs(args.results_root, args.seed)
    if len(runs) != len(P_VALUES) * len(FOLDS):
        logger.warning(
            "Expected %d runs, found %d", len(P_VALUES) * len(FOLDS), len(runs)
        )
    run_index = {(r.p, r.fold): r for r in runs}

    sweep_t0 = time.perf_counter()

    # ----- Probes -----
    if not args.patching_only:
        logger.info("=== Probe pass: %d runs ===", len(runs))
        for run in runs:
            try:
                run_probes_on_run(
                    run,
                    h5_path=args.h5_path,
                    n_probe_samples=args.n_probe_samples,
                    device=args.device,
                    num_workers=args.num_workers,
                    overwrite=args.overwrite,
                )
            except Exception:
                logger.exception(
                    "Probe failed for p=%s fold=%d (continuing)", run.p, run.fold
                )

    # ----- Patching -----
    if not args.probes_only:
        logger.info(
            "=== Patching pass: p=%s -> p=%s, %d folds ===",
            args.p_star,
            args.p_baseline,
            len(FOLDS),
        )
        for fold in FOLDS:
            if (args.p_star, fold) not in run_index or (
                args.p_baseline,
                fold,
            ) not in run_index:
                logger.warning("Skipping fold %d (missing run)", fold)
                continue
            try:
                run_patching_for_fold(
                    fold=fold,
                    run_index=run_index,
                    p_star=args.p_star,
                    p_baseline=args.p_baseline,
                    h5_path=args.h5_path,
                    n_probe_samples=args.n_probe_samples,
                    device=args.device,
                    num_workers=args.num_workers,
                    overwrite=args.overwrite,
                )
            except Exception:
                logger.exception("Patching failed for fold %d (continuing)", fold)

    logger.info("Phase 4 sweep finished in %.1fs", time.perf_counter() - sweep_t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
