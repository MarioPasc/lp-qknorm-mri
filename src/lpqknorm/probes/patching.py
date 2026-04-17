"""Activation patching for causal analysis of the Lp effect.

Swap stage-0 attention intermediates from a source checkpoint (typically
the best-small-recall ``p = p*`` run) into the forward pass of a target
checkpoint (typically the ``p = 2`` baseline), then measure the Dice
delta per slice.  Five variants isolate the locus of the effect:

- ``q``       — swap raw pre-norm queries.
- ``k``       — swap raw pre-norm keys.
- ``qk``      — swap both.
- ``qhat_khat`` — swap the post-Lp-norm representations (target's
  ``alpha`` is kept).
- ``logits``  — swap the full pre-bias logits ``alpha * <q_hat, k_hat>``.

The patching-effect score is

    PE = (D_patched - D_target) / (D_source - D_target + eps)
       ∈ (-inf, 1].

``PE ≈ 1`` → mechanism is fully localised at the patched stage/block.
``PE ≈ 0`` → mechanism is elsewhere.
``PE < 0``  → the patch destroys the target.

References
----------
- Meng et al.  *Locating and Editing Factual Associations in GPT* (ROME).
  NeurIPS 2022.  arXiv:2202.05262.
- Heimersheim & Nanda.  *How to use and interpret activation patching*.
  arXiv:2404.15255.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import h5py
import numpy as np
import torch
import torch.nn as nn

from lpqknorm.models.hooks import AttentionCapture, AttentionHookRegistry
from lpqknorm.training.metrics import dice_score


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence
    from pathlib import Path

    from torch.utils.data import DataLoader

    from lpqknorm.models.attention import LpWindowAttention


logger = logging.getLogger(__name__)

VALID_VARIANTS: tuple[str, ...] = ("q", "k", "qk", "qhat_khat", "logits")


@dataclass(frozen=True)
class PatchingConfig:
    """Configuration for one activation-patching sweep.

    Parameters
    ----------
    source_checkpoint : Path
        Checkpoint from which intermediate captures are drawn.  In the
        "denoising" direction this is the ``p = p*`` (high-performing) run.
    target_checkpoint : Path
        Checkpoint whose forward pass receives the patches.
    stage : int
        Swin stage at which to patch.  Default ``0``.
    blocks : Sequence[int]
        Block indices within the stage to patch.  Default ``(0, 1)``.
    variants : Sequence[str]
        Subset of :data:`VALID_VARIANTS` to evaluate.
    direction : ``"denoising"`` or ``"noising"``
        Labelling only; the source/target roles are already fixed by the
        ``*_checkpoint`` fields.  Stored in the output metadata so Phase 5
        can separate the two analyses.
    n_probe_samples : int
        Maximum samples from ``probe_loader`` to use.  Default ``32``.
    """

    source_checkpoint: Path
    target_checkpoint: Path
    stage: int = 0
    blocks: Sequence[int] = field(default_factory=lambda: (0, 1))
    variants: Sequence[str] = field(default_factory=lambda: VALID_VARIANTS)
    direction: Literal["denoising", "noising"] = "denoising"
    n_probe_samples: int = 32


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_block_attention(model: nn.Module, stage: int, block_idx: int) -> nn.Module:
    """Return ``model.swinViT.layers{stage+1}[0].blocks[block_idx].attn``."""
    attr = f"layers{stage + 1}"
    layer = getattr(model.swinViT, attr)[0]
    return layer.blocks[block_idx].attn  # type: ignore[no-any-return]


def _pred_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Binary prediction via sigmoid + 0.5 threshold."""
    return (torch.sigmoid(logits) > 0.5).to(logits.dtype)


@contextmanager
def _monkey_patch_forward(
    module: nn.Module, new_forward: Callable[..., torch.Tensor]
) -> Iterator[None]:
    """Temporarily override ``module.forward`` with ``new_forward``."""
    original = module.__dict__.get("forward", None)
    module.forward = new_forward
    try:
        yield
    finally:
        if original is None:
            # Instance had no override before; fall back to class forward.
            if "forward" in module.__dict__:
                del module.__dict__["forward"]
        else:
            module.forward = original


def _make_patched_forward(
    attn_module: LpWindowAttention,
    source_cap: AttentionCapture,
    variant: str,
) -> Callable[..., torch.Tensor]:
    """Return a replacement forward for ``attn_module`` honouring ``variant``.

    The returned closure re-implements :meth:`LpWindowAttention.forward`
    verbatim, with variant-specific substitutions of ``q``, ``k``,
    ``q_hat``, ``k_hat``, or the full pre-bias logits.
    """
    if variant not in VALID_VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; expected {VALID_VARIANTS}")

    qkv = attn_module.qkv
    proj = attn_module.proj
    attn_drop = attn_module.attn_drop
    proj_drop = attn_module.proj_drop
    softmax = attn_module.softmax
    lp_qknorm = attn_module.lp_qknorm
    rpb_table = attn_module.relative_position_bias_table
    rpb_index = attn_module.relative_position_index
    num_heads = attn_module.num_heads

    src_q = source_cap.q
    src_k = source_cap.k
    src_qhat = source_cap.q_hat
    src_khat = source_cap.k_hat
    src_logits = source_cap.logits

    def patched(x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b, n, c = x.shape
        qkv_out = (
            qkv(x).reshape(b, n, 3, num_heads, c // num_heads).permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv_out[0], qkv_out[1], qkv_out[2]

        # --- variant overrides ---
        if variant in ("q", "qk"):
            if src_q is None:
                raise ValueError(f"Variant {variant!r} requires source q")
            q = src_q.to(q.device, q.dtype)
        if variant in ("k", "qk"):
            if src_k is None:
                raise ValueError(f"Variant {variant!r} requires source k")
            k = src_k.to(k.device, k.dtype)

        if variant == "qhat_khat":
            if src_qhat is None or src_khat is None:
                raise ValueError(
                    "qhat_khat variant requires source q_hat/k_hat; vanilla "
                    "sources do not produce them."
                )
            q_hat = src_qhat.to(q.device, q.dtype)
            k_hat = src_khat.to(k.device, k.dtype)
            _, _, alpha = lp_qknorm(q, k)  # use target's learned alpha
            logits_pre_bias = alpha * (q_hat @ k_hat.transpose(-2, -1))
        elif variant == "logits":
            if src_logits is None:
                raise ValueError("logits variant requires source logits")
            logits_pre_bias = src_logits.to(q.device, q.dtype)
        else:
            q_hat, k_hat, alpha = lp_qknorm(q, k)
            logits_pre_bias = alpha * (q_hat @ k_hat.transpose(-2, -1))

        # --- relative position bias ---
        rpb_idx_buf = torch.as_tensor(rpb_index)
        rpb = rpb_table[rpb_idx_buf.clone()[:n, :n].reshape(-1)].reshape(n, n, -1)
        rpb = rpb.permute(2, 0, 1).contiguous()
        attn = logits_pre_bias + rpb.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b // nw, nw, num_heads, n, n) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, num_heads, n, n)
        attn = softmax(attn)
        attn = attn_drop(attn).to(v.dtype)
        x_out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x_out = proj(x_out)
        x_out = proj_drop(x_out)
        return x_out  # type: ignore[no-any-return]

    return patched


# ---------------------------------------------------------------------------
# ActivationPatcher
# ---------------------------------------------------------------------------


def _default_model_loader(path: Path) -> nn.Module:
    """Load a Lightning-checkpointed :class:`LpSegmentationModule` model."""
    from lpqknorm.training.module import LpSegmentationModule

    module = LpSegmentationModule.load_from_checkpoint(path, map_location="cpu")
    return module.model  # the underlying SwinUNETR-LP nn.Module


class ActivationPatcher:
    """Run an activation-patching sweep for one (source, target) pair.

    Parameters
    ----------
    cfg : PatchingConfig
    model_loader : Callable[[Path], nn.Module], optional
        Factory that maps a checkpoint path to a ready-to-eval
        :class:`~lpqknorm.models.swin_unetr_lp.build_swin_unetr_lp` model.
        Default loads a Lightning checkpoint and returns ``module.model``.
        Tests pass a simpler factory that loads a raw ``state_dict`` into
        a freshly built model.
    source_model, target_model : nn.Module, optional
        Pre-loaded models.  When both are provided, ``model_loader`` and
        the ``*_checkpoint`` fields of ``cfg`` are ignored.  Used by tests
        to bypass checkpoint I/O.
    """

    def __init__(
        self,
        cfg: PatchingConfig,
        model_loader: Callable[[Path], nn.Module] | None = None,
        source_model: nn.Module | None = None,
        target_model: nn.Module | None = None,
    ) -> None:
        self._cfg = cfg
        self._loader = model_loader or _default_model_loader
        self._preloaded_source = source_model
        self._preloaded_target = target_model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        probe_loader: DataLoader[Any],
        output_dir: Path,
        device: torch.device | str = "cpu",
    ) -> Path:
        """Execute the patching sweep and write the HDF5 output.

        Parameters
        ----------
        probe_loader : DataLoader
            Fixed probe loader (same one used by :class:`ProbeRecorder`).
        output_dir : Path
            Directory for the output file.  Created if needed.
        device : torch.device or str
            Device for both source and target forward passes.

        Returns
        -------
        Path
            Path of the written HDF5 file.
        """
        cfg = self._cfg
        device = torch.device(device) if isinstance(device, str) else device

        source_model = (
            (
                self._preloaded_source
                if self._preloaded_source is not None
                else self._loader(cfg.source_checkpoint)
            )
            .to(device)
            .eval()
        )
        target_model = (
            (
                self._preloaded_target
                if self._preloaded_target is not None
                else self._loader(cfg.target_checkpoint)
            )
            .to(device)
            .eval()
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "patching_best_dice.h5"

        # Per-block, per-variant accumulators.
        dice_patched: dict[int, dict[str, list[float]]] = {
            b: {v: [] for v in cfg.variants} for b in cfg.blocks
        }
        pe_vals: dict[int, dict[str, list[float]]] = {
            b: {v: [] for v in cfg.variants} for b in cfg.blocks
        }
        preds_patched: dict[int, dict[str, list[np.ndarray]]] = {
            b: {v: [] for v in cfg.variants} for b in cfg.blocks
        }
        dice_source_all: list[float] = []
        dice_target_all: list[float] = []

        samples_seen = 0
        with torch.inference_mode():
            for batch in probe_loader:
                if samples_seen >= cfg.n_probe_samples:
                    break
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                b = images.shape[0]
                remaining = cfg.n_probe_samples - samples_seen
                if b > remaining:
                    images = images[:remaining]
                    masks = masks[:remaining]
                    b = remaining

                # Capture source intermediates.
                source_caps = self._capture_source(
                    source_model, images, stage=cfg.stage
                )

                # Reference forward passes.
                src_logits_out = source_model(images)
                tgt_logits_out = target_model(images)
                d_src = dice_score(_pred_from_logits(src_logits_out), masks)
                d_tgt = dice_score(_pred_from_logits(tgt_logits_out), masks)
                dice_source_all.extend([float(x) for x in d_src.tolist()])
                dice_target_all.extend([float(x) for x in d_tgt.tolist()])

                # Patch sweep.
                for block_idx in cfg.blocks:
                    cap = source_caps.get(block_idx)
                    if cap is None:
                        logger.warning(
                            "No source capture for block %d; skipping.",
                            block_idx,
                        )
                        continue
                    target_attn = _get_block_attention(
                        target_model, cfg.stage, block_idx
                    )
                    for variant in cfg.variants:
                        patched_fn = _make_patched_forward(target_attn, cap, variant)  # type: ignore[arg-type]
                        with _monkey_patch_forward(target_attn, patched_fn):
                            out = target_model(images)
                        d_pat = dice_score(_pred_from_logits(out), masks)
                        pe = (d_pat - d_tgt) / (d_src - d_tgt + 1e-8)
                        dice_patched[block_idx][variant].extend(
                            [float(x) for x in d_pat.tolist()]
                        )
                        pe_vals[block_idx][variant].extend(
                            [float(x) for x in pe.tolist()]
                        )
                        pred_np = (
                            _pred_from_logits(out)
                            .detach()
                            .cpu()
                            .squeeze(1)
                            .to(torch.uint8)
                            .numpy()
                        )
                        preds_patched[block_idx][variant].append(pred_np)

                samples_seen += b

        # Write HDF5.
        self._write_h5(
            out_path,
            dice_patched=dice_patched,
            pe_vals=pe_vals,
            preds_patched=preds_patched,
            dice_source_all=dice_source_all,
            dice_target_all=dice_target_all,
        )
        logger.info(
            "ActivationPatcher: wrote %s (variants=%s, blocks=%s, n=%d)",
            out_path,
            list(cfg.variants),
            list(cfg.blocks),
            samples_seen,
        )
        return out_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _capture_source(
        self, source_model: nn.Module, images: torch.Tensor, stage: int
    ) -> dict[int, AttentionCapture]:
        registry = AttentionHookRegistry()
        registry.register(source_model, stages=[stage])
        try:
            _ = source_model(images)
            caps = registry.captures()
        finally:
            registry.remove()
        return {cap.block_index: cap for cap in caps}

    def _write_h5(
        self,
        path: Path,
        *,
        dice_patched: dict[int, dict[str, list[float]]],
        pe_vals: dict[int, dict[str, list[float]]],
        preds_patched: dict[int, dict[str, list[np.ndarray]]],
        dice_source_all: list[float],
        dice_target_all: list[float],
    ) -> None:
        cfg = self._cfg
        with h5py.File(path, "w") as f:
            meta = f.create_group("metadata")
            meta.attrs["source_checkpoint"] = str(cfg.source_checkpoint)
            meta.attrs["target_checkpoint"] = str(cfg.target_checkpoint)
            meta.attrs["direction"] = cfg.direction
            meta.attrs["stage_index"] = cfg.stage
            meta.attrs["variants"] = list(cfg.variants)

            for block_idx in cfg.blocks:
                grp = f.create_group(f"block_{block_idx}")
                grp.create_dataset(
                    "dice_source",
                    data=np.asarray(dice_source_all, dtype=np.float32),
                )
                grp.create_dataset(
                    "dice_target",
                    data=np.asarray(dice_target_all, dtype=np.float32),
                )
                for variant in cfg.variants:
                    vgrp = grp.create_group(f"variant_{variant}")
                    vgrp.create_dataset(
                        "dice_patched",
                        data=np.asarray(
                            dice_patched[block_idx][variant], dtype=np.float32
                        ),
                    )
                    vgrp.create_dataset(
                        "pe",
                        data=np.asarray(pe_vals[block_idx][variant], dtype=np.float32),
                    )
                    if preds_patched[block_idx][variant]:
                        arr = np.concatenate(preds_patched[block_idx][variant], axis=0)
                        vgrp.create_dataset(
                            "prediction",
                            data=arr,
                            compression="gzip",
                            compression_opts=4,
                        )


def run_patching_sweep(
    probe_loader: DataLoader[Any],
    cfg: PatchingConfig,
    output_dir: Path,
    device: torch.device | str = "cpu",
    model_loader: Callable[[Path], nn.Module] | None = None,
) -> Path:
    """Thin wrapper that instantiates :class:`ActivationPatcher` and runs it."""
    patcher = ActivationPatcher(cfg, model_loader=model_loader)
    return patcher.run(probe_loader, output_dir=output_dir, device=device)


__all__ = [
    "VALID_VARIANTS",
    "ActivationPatcher",
    "PatchingConfig",
    "run_patching_sweep",
]
