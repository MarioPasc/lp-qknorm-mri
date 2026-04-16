"""Probe recorder — orchestrates hook registration, probe computation, and HDF5 output.

The :class:`ProbeRecorder` is the central entry point for Phase 4.  It:

1. Registers forward hooks via :class:`AttentionHookRegistry`.
2. Iterates the fixed probe DataLoader (first ``n_probe_samples`` slices).
3. For each batch, runs all probes on both blocks (W-MSA and SW-MSA).
4. Writes a single HDF5 file with groups ``/block_0_wmsa/`` and
   ``/block_1_swmsa/``.
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
from lpqknorm.probes.tokenization import mask_to_token_flags, window_partition_flags


if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from torch.utils.data import DataLoader

    from lpqknorm.probes.base import Probe


logger = logging.getLogger(__name__)

_BLOCK_NAMES = {0: "block_0_wmsa", 1: "block_1_swmsa"}


class ProbeRecorder:
    """Orchestrate hook registration, probe computation, and HDF5 output.

    Parameters
    ----------
    probes : Sequence[Probe]
        Probe instances to run (all five, or a subset).
    output_dir : Path
        Directory where HDF5 files are written.
    stage : int
        Swin stage to probe.  Default ``0`` (finest resolution).
    patch_stride : tuple[int, int]
        Mask downsampling stride.  Default ``(2, 2)``.
    window_size : int
        Window size.  Default ``7``.
    n_probe_samples : int
        Maximum number of validation samples to process.  Default ``32``.
    """

    def __init__(
        self,
        probes: Sequence[Probe],
        output_dir: Path,
        stage: int = 0,
        patch_stride: tuple[int, int] = (2, 2),
        window_size: int = 7,
        n_probe_samples: int = 32,
    ) -> None:
        self._probes = list(probes)
        self._output_dir = output_dir
        self._stage = stage
        self._patch_stride = patch_stride
        self._window_size = window_size
        self._n_probe_samples = n_probe_samples

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

        # Accumulators: block_name → probe_name → list[Tensor]
        accum: dict[str, dict[str, list[torch.Tensor]]] = {
            bn: defaultdict(list) for bn in _BLOCK_NAMES.values()
        }
        # Alpha per block (scalar, last seen)
        alpha_vals: dict[str, float] = {}

        samples_seen = 0
        model.eval()

        with torch.inference_mode():
            for batch in dataloader:
                if samples_seen >= self._n_probe_samples:
                    break

                images: torch.Tensor = batch["image"].to(device)
                masks: torch.Tensor = batch["mask"].to(device)
                b = images.shape[0]

                # Clamp to n_probe_samples
                remaining = self._n_probe_samples - samples_seen
                if b > remaining:
                    images = images[:remaining]
                    masks = masks[:remaining]
                    b = remaining

                registry.clear()
                _ = model(images)
                captures = registry.captures()

                # Compute token flags once
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

                    # Store alpha
                    if cap.alpha is not None:
                        alpha_vals[block_name] = float(cap.alpha.item())

                    # Run each probe
                    for probe in self._probes:
                        result = probe.compute(cap, lesion_flags)
                        if result.per_token is not None:
                            accum[block_name][result.name].append(
                                result.per_token.cpu()
                            )
                        elif result.per_query is not None:
                            accum[block_name][result.name].append(
                                result.per_query.cpu()
                            )
                        # Store is_lesion metadata under a special key
                        if "is_lesion" in result.metadata:
                            il = result.metadata["is_lesion"]
                            key = f"{result.name}_is_lesion"
                            if isinstance(il, torch.Tensor):
                                accum[block_name][key].append(il.cpu())

                samples_seen += b

        registry.remove()

        # Write HDF5
        self._write_h5(out_path, accum, alpha_vals, str(epoch_tag))
        logger.info("ProbeRecorder: wrote %s (%d samples)", out_path, samples_seen)
        return out_path

    def _write_h5(
        self,
        path: Path,
        accum: dict[str, dict[str, list[torch.Tensor]]],
        alpha_vals: dict[str, float],
        epoch_tag: str,
    ) -> None:
        """Write accumulated probe results to HDF5."""
        with h5py.File(path, "w") as f:
            for block_name, probe_data in accum.items():
                grp = f.create_group(block_name)
                for key, tensors in probe_data.items():
                    if not tensors:
                        continue
                    arr = torch.cat(tensors).numpy()
                    grp.create_dataset(
                        key,
                        data=arr,
                        compression="gzip",
                        compression_opts=4,
                    )
                # Alpha scalar
                if block_name in alpha_vals:
                    grp.create_dataset(
                        "alpha",
                        data=np.float32(alpha_vals[block_name]),
                    )

            # Metadata attributes
            meta = f.create_group("metadata")
            meta.attrs["epoch_tag"] = epoch_tag
            meta.attrs["n_probe_samples"] = self._n_probe_samples
            meta.attrs["patch_stride"] = list(self._patch_stride)
            meta.attrs["window_size"] = self._window_size
