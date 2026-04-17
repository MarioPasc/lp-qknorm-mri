"""Probe recorder — orchestrates hook registration, probe computation, and HDF5 output.

The :class:`ProbeRecorder` is the central Phase-4 entry point.  It:

1. Registers forward hooks via :class:`AttentionHookRegistry`.
2. Iterates the fixed probe DataLoader (first ``n_probe_samples`` slices).
3. For each batch, runs per-capture probes (Probes 1–6) on both blocks.
4. Accumulates per-slice attention, logits, rel-pos-bias, input images,
   masks, and provenance metadata.
5. Runs per-block probes (Probes 7–8) on the full pooled feature tensor
   after all slices are seen.
6. Writes a single HDF5 file with groups ``/metadata``, ``/inputs``,
   ``/block_0_wmsa``, ``/block_1_swmsa`` matching the Phase-4 spec.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np
import torch
import torch.nn as nn

from lpqknorm.models.hooks import AttentionHookRegistry
from lpqknorm.probes.tokenization import (
    mask_to_token_flags,
    window_boundary_distance,
    window_partition_flags,
)


if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from torch.utils.data import DataLoader

    from lpqknorm.probes.base import Probe, ProbeResult


logger = logging.getLogger(__name__)

_BLOCK_NAMES = {0: "block_0_wmsa", 1: "block_1_swmsa"}

_PER_QUERY_NAMES = frozenset(
    {
        "entropy",
        "lesion_mass",
        "logit_gap",
        "attention_iou",
        "spatial_localization_error",
    }
)


class ProbeRecorder:
    """Orchestrate hook registration, probe computation, and HDF5 output.

    Parameters
    ----------
    probes : Sequence[Probe]
        Probe instances to run.  Per-capture probes run each batch;
        per-block probes (those returning ``per_block``) also run once
        per block on the pooled data.
    output_dir : Path
        Directory where HDF5 files are written.
    stage : int
        Swin stage to probe.  Default ``0`` (finest resolution).
    patch_stride : tuple[int, int]
        Mask downsampling stride.  Default ``(2, 2)``.
    window_size : int
        Window side length ``W``.  Default ``7``.
    n_probe_samples : int
        Maximum number of validation samples to process.  Default ``32``.
    save_attention_maps : bool
        If ``True`` (default), persist per-sample ``attention_full``
        tensors as ``float16`` for each block.
    save_logits : bool
        If ``True`` (default), persist per-sample ``logits_full``
        (bias-inclusive, pre-softmax) as ``float16``.
    save_rel_pos_bias : bool
        If ``True`` (default), persist the block's relative-position-bias
        tensor and its per-head softmax entropy.
    compression : str
        HDF5 compression.  ``"gzip"`` (default) is portable; use
        ``"lzf"`` for faster, less dense storage.
    rng_seed : int
        Seed recorded in ``/metadata/rng_seed`` for provenance.

    Notes
    -----
    All tensor work runs inside ``torch.inference_mode()`` so no autograd
    graph is kept alive.  Captures are cloned and detached before any
    probe receives them.
    """

    def __init__(
        self,
        probes: Sequence[Probe],
        output_dir: Path,
        stage: int = 0,
        patch_stride: tuple[int, int] = (2, 2),
        window_size: int = 7,
        n_probe_samples: int = 32,
        save_attention_maps: bool = True,
        save_logits: bool = True,
        save_rel_pos_bias: bool = True,
        compression: str = "gzip",
        rng_seed: int = 0,
    ) -> None:
        self._probes = list(probes)
        self._output_dir = output_dir
        self._stage = stage
        self._patch_stride = patch_stride
        self._window_size = window_size
        self._n_probe_samples = n_probe_samples
        self._save_attention_maps = save_attention_maps
        self._save_logits = save_logits
        self._save_rel_pos_bias = save_rel_pos_bias
        self._compression = compression
        self._rng_seed = rng_seed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        model: nn.Module,
        dataloader: DataLoader[Any],
        epoch_tag: str | int,
        device: torch.device | str = "cpu",
    ) -> Path:
        """Run all probes on the fixed probe batch and write HDF5.

        Parameters
        ----------
        model : nn.Module
            The SwinUNETR model (with Lp-QKNorm patching).
        dataloader : DataLoader
            Fixed probe batch loader (no shuffle, no augmentation).
        epoch_tag : str or int
            Label for the output file (e.g. ``"0"``, ``"final"``).
        device : torch.device or str
            Device to run the model on.

        Returns
        -------
        Path
            Path to the written HDF5 file.
        """
        device = torch.device(device) if isinstance(device, str) else device
        self._output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self._output_dir / f"epoch_{epoch_tag}.h5"

        registry = AttentionHookRegistry()
        registry.register(model, stages=[self._stage])

        # Per-block accumulators.
        per_token_accum: dict[str, dict[str, list[torch.Tensor]]] = {
            bn: defaultdict(list) for bn in _BLOCK_NAMES.values()
        }
        per_query_accum: dict[str, dict[str, list[torch.Tensor]]] = {
            bn: defaultdict(list) for bn in _BLOCK_NAMES.values()
        }
        alpha_vals: dict[str, float] = {}
        rel_pos_bias: dict[str, torch.Tensor] = {}

        # Per-slice heavy tensors.
        attn_maps: dict[str, list[torch.Tensor]] = defaultdict(list)
        logits_maps: dict[str, list[torch.Tensor]] = defaultdict(list)

        # Pooled features for per-block probes (Probes 7 & 8).
        feature_pool: dict[str, list[torch.Tensor]] = defaultdict(list)
        lesion_pool: dict[str, list[torch.Tensor]] = defaultdict(list)

        # Provenance.
        input_images: list[torch.Tensor] = []
        input_masks: list[torch.Tensor] = []
        input_subject_ids: list[str] = []
        input_slice_indices: list[int] = []

        # Per-lesion-query provenance (shared across blocks via shift convention).
        wbd_per_block: dict[str, list[torch.Tensor]] = defaultdict(list)

        samples_seen = 0
        model.eval()

        with torch.inference_mode():
            for batch in dataloader:
                if samples_seen >= self._n_probe_samples:
                    break

                images: torch.Tensor = batch["image"].to(device)
                masks: torch.Tensor = batch["mask"].to(device)
                b = images.shape[0]

                # Clamp to n_probe_samples.
                remaining = self._n_probe_samples - samples_seen
                if b > remaining:
                    images = images[:remaining]
                    masks = masks[:remaining]
                    b = remaining

                # Slice provenance fields may or may not be collated; fall back to
                # generic synthetic values when the DataLoader does not provide them.
                subj = batch.get("subject_id")
                slc = batch.get("slice_idx")
                if isinstance(subj, list):
                    subj_list = [str(s) for s in subj[:b]]
                elif isinstance(subj, (str, bytes)):
                    subj_list = [str(subj)] * b
                else:
                    subj_list = [f"probe-{samples_seen + i:04d}" for i in range(b)]
                if isinstance(slc, torch.Tensor):
                    slc_list = [int(x) for x in slc[:b].tolist()]
                elif isinstance(slc, list):
                    slc_list = [int(x) for x in slc[:b]]
                else:
                    slc_list = [samples_seen + i for i in range(b)]

                input_images.append(images.detach().cpu().to(torch.float16))
                input_masks.append(masks.detach().cpu().squeeze(1).to(torch.uint8))
                input_subject_ids.extend(subj_list)
                input_slice_indices.extend(slc_list)

                registry.clear()
                _ = model(images)
                captures = registry.captures()

                # Token flags once per batch.
                token_flags = mask_to_token_flags(
                    masks, patch_stride=self._patch_stride
                )
                h_tok = images.shape[2] // self._patch_stride[0]
                w_tok = images.shape[3] // self._patch_stride[1]

                for cap in captures:
                    block_idx = cap.block_index
                    block_name = _BLOCK_NAMES.get(block_idx)
                    if block_name is None:
                        continue

                    shift = 0 if block_idx == 0 else self._window_size // 2
                    lesion_flags = window_partition_flags(
                        token_flags,
                        img_hw_tok=(h_tok, w_tok),
                        window_size=self._window_size,
                        shift_size=shift,
                    )

                    # Alpha, rel-pos-bias.
                    if cap.alpha is not None:
                        alpha_vals[block_name] = float(cap.alpha.item())
                    if (
                        cap.relative_position_bias is not None
                        and block_name not in rel_pos_bias
                    ):
                        # rel_pos_bias shape in LpWindowAttention: (1, nh, n, n)
                        # We also want the canonical (W²·W², nh) flat form for
                        # the HDF5.  Reconstruct it from the per-head (n, n) maps.
                        rpb = cap.relative_position_bias.detach().cpu()
                        rel_pos_bias[block_name] = rpb.to(torch.float32)

                    # Save heavy attention/logits maps if requested.
                    if self._save_attention_maps and cap.attention is not None:
                        # Reshape (B*nW, nh, n, n) -> (B, nW, nh, n, n)
                        a = cap.attention.detach().cpu().to(torch.float16)
                        _, nh, n_q, _ = a.shape
                        n_win = a.shape[0] // b
                        a = a.view(b, n_win, nh, n_q, n_q)
                        attn_maps[block_name].append(a)

                    if self._save_logits and cap.logits is not None:
                        logs = cap.logits.detach().cpu()
                        if cap.relative_position_bias is not None:
                            logs = logs + cap.relative_position_bias.detach().cpu()
                        logs = logs.to(torch.float16)
                        _, nh, n_q, _ = logs.shape
                        n_win = logs.shape[0] // b
                        logs = logs.view(b, n_win, nh, n_q, n_q)
                        logits_maps[block_name].append(logs)

                    # Pool features for per-block probes (Probes 7 & 8).
                    if cap.q is not None:
                        feature_pool[block_name].append(cap.q.detach().cpu())
                        lesion_pool[block_name].append(lesion_flags.detach().cpu())

                    # Window-boundary distance (W-MSA coords; computed on
                    # the **unshifted** grid per spec).
                    if block_idx == 0:
                        wbd = window_boundary_distance(lesion_flags, self._window_size)
                        wbd_per_block[block_name].append(wbd.detach().cpu())
                    else:
                        # For SW-MSA, compute d_wb on the un-shifted tokens
                        # so comparisons with W-MSA remain meaningful.
                        unshifted = window_partition_flags(
                            token_flags,
                            img_hw_tok=(h_tok, w_tok),
                            window_size=self._window_size,
                            shift_size=0,
                        )
                        wbd = window_boundary_distance(unshifted, self._window_size)
                        wbd_per_block[block_name].append(wbd.detach().cpu())

                    # Run per-capture probes (Probes 1–6).
                    for probe in self._probes:
                        # Skip per-block probes here; they run once later.
                        if _is_per_block_only(probe):
                            continue
                        result = probe.compute(cap, lesion_flags)
                        self._stash_result(
                            result,
                            per_token_accum[block_name],
                            per_query_accum[block_name],
                        )

                samples_seen += b

        registry.remove()

        # Run per-block probes on the pooled features.
        block_level: dict[str, dict[str, torch.Tensor]] = {
            bn: {} for bn in _BLOCK_NAMES.values()
        }
        for block_name in _BLOCK_NAMES.values():
            qs = feature_pool.get(block_name)
            lfs = lesion_pool.get(block_name)
            if not qs or not lfs:
                continue
            cap_fake = _make_pooled_capture(qs, lfs)
            lesion_flags_pooled = cap_fake[1]
            from lpqknorm.models.hooks import AttentionCapture  # local import

            pooled_cap = AttentionCapture(q=cap_fake[0])
            for probe in self._probes:
                if not _is_per_block_only(probe):
                    continue
                result = probe.compute(pooled_cap, lesion_flags_pooled)
                if result.per_block is not None:
                    block_level[block_name].update(result.per_block)

        # Write HDF5.
        self._write_h5(
            out_path,
            per_token_accum=per_token_accum,
            per_query_accum=per_query_accum,
            alpha_vals=alpha_vals,
            rel_pos_bias=rel_pos_bias,
            attn_maps=attn_maps,
            logits_maps=logits_maps,
            block_level=block_level,
            wbd_per_block=wbd_per_block,
            input_images=input_images,
            input_masks=input_masks,
            input_subject_ids=input_subject_ids,
            input_slice_indices=input_slice_indices,
            epoch_tag=str(epoch_tag),
        )
        logger.info("ProbeRecorder: wrote %s (%d samples)", out_path, samples_seen)
        return out_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _stash_result(
        result: ProbeResult,
        per_token: dict[str, list[torch.Tensor]],
        per_query: dict[str, list[torch.Tensor]],
    ) -> None:
        if result.per_token is not None:
            per_token[result.name].append(result.per_token.detach().cpu())
            il = result.metadata.get("is_lesion")
            if isinstance(il, torch.Tensor):
                per_token[f"{result.name}_is_lesion"].append(il.detach().cpu())
        if result.per_query is not None:
            per_query[result.name].append(result.per_query.detach().cpu())

    def _write_h5(
        self,
        path: Path,
        *,
        per_token_accum: dict[str, dict[str, list[torch.Tensor]]],
        per_query_accum: dict[str, dict[str, list[torch.Tensor]]],
        alpha_vals: dict[str, float],
        rel_pos_bias: dict[str, torch.Tensor],
        attn_maps: dict[str, list[torch.Tensor]],
        logits_maps: dict[str, list[torch.Tensor]],
        block_level: dict[str, dict[str, torch.Tensor]],
        wbd_per_block: dict[str, list[torch.Tensor]],
        input_images: list[torch.Tensor],
        input_masks: list[torch.Tensor],
        input_subject_ids: list[str],
        input_slice_indices: list[int],
        epoch_tag: str,
    ) -> None:
        """Write every accumulator to HDF5 under the Phase-4 schema."""
        comp = self._compression

        with h5py.File(path, "w") as f:
            # /metadata
            meta = f.create_group("metadata")
            meta.attrs["epoch_tag"] = epoch_tag
            meta.attrs["n_probe_samples"] = self._n_probe_samples
            meta.attrs["patch_stride"] = list(self._patch_stride)
            meta.attrs["window_size"] = self._window_size
            meta.attrs["stage_index"] = self._stage
            meta.attrs["rng_seed"] = self._rng_seed
            meta.attrs["save_attention_maps"] = self._save_attention_maps
            meta.attrs["save_logits"] = self._save_logits
            meta.attrs["save_rel_pos_bias"] = self._save_rel_pos_bias

            # /inputs
            if input_images:
                ing = f.create_group("inputs")
                imgs = torch.cat(input_images, dim=0).numpy()
                mks = torch.cat(input_masks, dim=0).numpy()
                ing.create_dataset("image", data=imgs, compression=comp)
                ing.create_dataset("mask", data=mks, compression=comp)
                ing.create_dataset(
                    "subject_id",
                    data=np.array(input_subject_ids, dtype="S16"),
                )
                ing.create_dataset(
                    "slice_index",
                    data=np.asarray(input_slice_indices, dtype=np.int32),
                )

            # /block_*
            for block_name in _BLOCK_NAMES.values():
                grp = f.create_group(block_name)

                # Per-token / per-query arrays (Probes 1–6).
                self._write_cat(grp, per_token_accum.get(block_name, {}), comp)
                self._write_cat(grp, per_query_accum.get(block_name, {}), comp)

                # Per-block tensors (Probes 7 & 8).
                for key, tensor in block_level.get(block_name, {}).items():
                    arr = tensor.detach().cpu().numpy().astype(np.float32)
                    # Scalar datasets cannot carry chunk/filter options.
                    if arr.ndim > 0:
                        grp.create_dataset(key, data=arr, compression=comp)
                    else:
                        grp.create_dataset(key, data=arr)

                # Alpha scalar.
                if block_name in alpha_vals:
                    grp.create_dataset(
                        "alpha",
                        data=np.float32(alpha_vals[block_name]),
                    )

                # Relative position bias tensor + softmax entropy.
                if self._save_rel_pos_bias and block_name in rel_pos_bias:
                    rpb = rel_pos_bias[block_name]
                    grp.create_dataset(
                        "rel_pos_bias",
                        data=rpb.numpy(),
                        compression=comp,
                    )
                    # Softmax-normalised entropy per head.  rpb may be shape
                    # (1, nh, n, n) or (nh, n, n); flatten the spatial dims.
                    rpb_flat = rpb.reshape(rpb.shape[0], rpb.shape[-3], -1)[0]
                    # rpb_flat: (nh, n*n); softmax over last dim, then entropy.
                    probs = torch.softmax(rpb_flat, dim=-1).clamp(min=1e-12)
                    ent = -(probs * probs.log()).sum(dim=-1)  # (nh,)
                    grp.create_dataset(
                        "rel_pos_bias_entropy",
                        data=ent.numpy().astype(np.float32),
                    )

                # Heavy attention/logits maps.
                if self._save_attention_maps and block_name in attn_maps:
                    arr = torch.cat(attn_maps[block_name], dim=0).numpy()
                    grp.create_dataset("attention_full", data=arr, compression=comp)
                if self._save_logits and block_name in logits_maps:
                    arr = torch.cat(logits_maps[block_name], dim=0).numpy()
                    grp.create_dataset("logits_full", data=arr, compression=comp)

                # Window-boundary distance per lesion query.
                if wbd_per_block.get(block_name):
                    wbd = torch.cat(wbd_per_block[block_name]).numpy().astype(np.int8)
                    grp.create_dataset(
                        "window_boundary_distance",
                        data=wbd,
                        compression=comp,
                    )

    @staticmethod
    def _write_cat(
        group: h5py.Group,
        mapping: dict[str, list[torch.Tensor]],
        compression: str,
    ) -> None:
        """Concatenate and write each tensor list to the HDF5 group."""
        for key, tensors in mapping.items():
            if not tensors:
                continue
            arr = torch.cat(tensors).numpy()
            group.create_dataset(key, data=arr, compression=compression)


def _is_per_block_only(probe: Probe) -> bool:
    """Return True if the probe is one of the per-block probes (7 or 8).

    Heuristic: check the probe's class name.  Per-block probes are
    :class:`LinearProbe` and :class:`SpectralProbe`.
    """
    cls_name = type(probe).__name__
    return cls_name in {"LinearProbe", "SpectralProbe"}


def _make_pooled_capture(
    qs: list[torch.Tensor],
    lfs: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool ``q`` and lesion-flag tensors from multiple captures.

    Each item in ``qs`` has shape ``(B*nW, nh, W², d_head)``; each item
    in ``lfs`` has shape ``(B*nW, W²)``.  Returns a pooled fake capture
    with matching leading dimensions so per-block probes see a single
    consolidated batch.
    """
    q_cat = torch.cat(qs, dim=0)  # (sum_B*nW, nh, n, d_head)
    lf_cat = torch.cat(lfs, dim=0)  # (sum_B*nW, n)
    return q_cat, lf_cat


__all__ = ["ProbeRecorder"]
