"""Forward-hook registry for capturing attention intermediates from LpWindowAttention.

This module provides the infrastructure used by Phase 4 probes to extract
per-forward-pass tensors (q, k, q_hat, k_hat, logits, attention weights, and
alpha) from :class:`~lpqknorm.models.attention.LpWindowAttention` modules
located at specific Swin stages.

Design rationale
----------------
Hooks store *references* (not copies) to tensors during the forward pass and
only perform ``.detach().clone()`` when :meth:`AttentionHookRegistry.captures`
is called.  This avoids two pitfalls simultaneously:

1. Keeping the autograd graph alive unnecessarily when probing under
   ``torch.inference_mode()``.
2. Paying the memory cost of cloning inside the time-critical forward pass.

The ``_capture`` dict is populated by ``LpWindowAttention.forward`` before the
hook fires.  The hook merely reads the already-populated dict and appends it to
the raw capture list, annotated with stage and block indices.

Stage mapping
-------------
Stage ``i`` maps to ``model.swinViT.layers{i+1}[0]``:

- Stage 0  →  ``model.swinViT.layers1[0]``  (finest resolution)
- Stage 1  →  ``model.swinViT.layers2[0]``
- Stage 2  →  ``model.swinViT.layers3[0]``
- Stage 3  →  ``model.swinViT.layers4[0]``  (coarsest resolution)

Each ``BasicLayer`` contains one or more ``SwinTransformerBlock`` instances,
each of which owns one :class:`~lpqknorm.models.attention.LpWindowAttention`.
The block index within the layer is recorded verbatim.

References
----------
- Liu et al. *Swin Transformer*. ICCV 2021. arXiv:2103.14030.
- Hatamizadeh et al. *Swin UNETR*. BrainLes 2021. arXiv:2201.01266.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch.nn as nn
from torch import Tensor

from lpqknorm.utils.exceptions import HookError


if TYPE_CHECKING:
    from collections.abc import Iterable

    from torch.utils.hooks import RemovableHandle

    from lpqknorm.models.attention import LpWindowAttention

# Type alias for the forward-hook callable returned by _make_hook.
_HookCallable = Callable[..., None]

logger = logging.getLogger(__name__)

# Stage-to-attribute mapping for MONAI SwinTransformer.
# Verified against monai.networks.nets.swin_unetr.SwinTransformer (MONAI ≥ 1.3).
_STAGE_ATTR: dict[int, str] = {
    0: "layers1",
    1: "layers2",
    2: "layers3",
    3: "layers4",
}


@dataclass
class AttentionCapture:
    """Snapshot of tensors produced by one :class:`LpWindowAttention` forward call.

    All fields are ``None`` until populated by :meth:`AttentionHookRegistry.captures`.
    Index fields default to ``-1`` as sentinel values.

    Attributes
    ----------
    q : Tensor or None
        Raw (pre-normalisation) query tensor of shape
        ``(B * n_windows, n_heads, W², head_dim)``.
    k : Tensor or None
        Raw (pre-normalisation) key tensor, same shape as ``q``.
    q_hat : Tensor or None
        Lp-normalised query tensor, same shape as ``q``.
    k_hat : Tensor or None
        Lp-normalised key tensor, same shape as ``k``.
    logits : Tensor or None
        Pre-softmax attention scores ``alpha * q_hat @ k_hat^T`` **before**
        relative position bias is added, shape
        ``(B * n_windows, n_heads, W^2, W^2)``.  Probe 4 uses this to
        compute the logit gap isolated from positional effects.
    attention : Tensor or None
        Post-softmax attention weights in [0, 1], same shape as ``logits``.
    alpha : Tensor or None
        Scalar learnable temperature ``alpha = softplus(alpha_raw)``, shape ``()``.
    stage_index : int
        Index of the Swin stage (0 = finest resolution).
    block_index : int
        Index of the ``SwinTransformerBlock`` within the stage.
    """

    q: Tensor | None = None
    k: Tensor | None = None
    q_hat: Tensor | None = None
    k_hat: Tensor | None = None
    logits: Tensor | None = None
    attention: Tensor | None = None
    alpha: Tensor | None = None
    stage_index: int = -1
    block_index: int = -1


class AttentionHookRegistry:
    """Registers forward hooks on :class:`~lpqknorm.models.attention.LpWindowAttention`
    modules and aggregates their captured tensors.

    Usage
    -----
    >>> registry = AttentionHookRegistry()
    >>> registry.register(model, stages=[0])
    >>> with torch.inference_mode():
    ...     _ = model(x)
    >>> captures = registry.captures()  # list[AttentionCapture]
    >>> registry.remove()

    The registry holds *raw* references during the forward pass and only
    materialises detached clones in :meth:`captures`.  This is safe because
    ``LpWindowAttention`` populates ``module._capture`` with live tensors
    before the hook fires; the hook stores a *shallow copy* of that dict so
    that the module can overwrite ``_capture`` on the next forward without
    corrupting already-stored captures.
    """

    def __init__(self) -> None:
        self._hooks: list[RemovableHandle] = []
        # Each element is a dict with keys matching AttentionCapture fields
        # plus 'stage_index' and 'block_index'.
        self._raw_captures: list[dict[str, Tensor | int]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self, model: nn.Module, stages: Iterable[int]) -> None:
        """Register forward hooks on all LpWindowAttention modules at the given stages.

        Walks ``model.swinViT.layers{i+1}[0].blocks`` for each ``i`` in
        ``stages`` and attaches a ``register_forward_hook`` to every
        :class:`~lpqknorm.models.attention.LpWindowAttention` found there.

        Parameters
        ----------
        model : nn.Module
            A model built by :func:`~lpqknorm.models.swin_unetr_lp.build_swin_unetr_lp`.
            Must expose a ``swinViT`` attribute with ``layers1`` … ``layers4``
            ``ModuleList`` attributes.
        stages : Iterable[int]
            Zero-based stage indices (0 = finest resolution).  Valid range: 0-3.

        Raises
        ------
        HookError
            If ``model`` does not have a ``swinViT`` attribute, if any requested
            stage index is out of range, or if no :class:`LpWindowAttention`
            modules are found after walking all requested stages (which likely
            means ``model`` is a vanilla MONAI SwinUNETR without Lp patching).
        """
        from lpqknorm.models.attention import (
            LpWindowAttention,  # local import to avoid circular
        )

        stages_list = list(stages)
        if not stages_list:
            raise HookError(
                "No stages specified; provide at least one stage index.",
                details={"stages": stages_list},
            )

        if not hasattr(model, "swinViT"):
            raise HookError(
                "model does not have a 'swinViT' attribute. "
                "Expected a model built by build_swin_unetr_lp().",
                details={"model_type": type(model).__name__},
            )

        swin_vit = model.swinViT
        hooks_registered = 0

        for stage_idx in stages_list:
            if stage_idx not in _STAGE_ATTR:
                raise HookError(
                    f"Stage index {stage_idx} is out of range. Valid indices: {sorted(_STAGE_ATTR)}.",
                    details={"stage_index": stage_idx},
                )
            attr = _STAGE_ATTR[stage_idx]
            if not hasattr(swin_vit, attr):
                raise HookError(
                    f"swinViT does not have attribute '{attr}' (stage {stage_idx}).",
                    details={"stage_index": stage_idx, "attr": attr},
                )

            basic_layer = getattr(swin_vit, attr)[0]
            if not hasattr(basic_layer, "blocks"):
                raise HookError(
                    f"BasicLayer at stage {stage_idx} has no 'blocks' attribute.",
                    details={
                        "stage_index": stage_idx,
                        "layer_type": type(basic_layer).__name__,
                    },
                )

            for block_idx, block in enumerate(basic_layer.blocks):
                attn_module = getattr(block, "attn", None)
                if attn_module is None:
                    logger.debug(
                        "Block %d at stage %d has no 'attn' attribute; skipping.",
                        block_idx,
                        stage_idx,
                    )
                    continue

                if not isinstance(attn_module, LpWindowAttention):
                    logger.debug(
                        "Block %d at stage %d: attn is %s, not LpWindowAttention; skipping.",
                        block_idx,
                        stage_idx,
                        type(attn_module).__name__,
                    )
                    continue

                # Capture stage/block indices in closure-safe variables.
                _stage = stage_idx
                _block = block_idx

                def _make_hook(
                    stage: int,
                    block: int,
                ) -> _HookCallable:
                    def _hook(
                        module: LpWindowAttention,
                        inputs: tuple[object, ...],
                        output: Tensor,
                    ) -> None:
                        capture_dict = getattr(module, "_capture", {})
                        # Shallow-copy the dict so the module can reset _capture
                        # on the next forward without corrupting this snapshot.
                        raw: dict[str, Tensor | int] = dict(capture_dict)
                        raw["stage_index"] = stage
                        raw["block_index"] = block
                        self._raw_captures.append(raw)

                    return _hook

                handle = attn_module.register_forward_hook(_make_hook(_stage, _block))
                self._hooks.append(handle)
                hooks_registered += 1
                logger.debug(
                    "Registered hook on LpWindowAttention at stage=%d, block=%d.",
                    stage_idx,
                    block_idx,
                )

        if hooks_registered == 0:
            raise HookError(
                "No LpWindowAttention modules found at the requested stages. "
                "Verify that build_swin_unetr_lp() was used with a non-None lp_cfg.",
                details={"stages": stages_list, "model_type": type(model).__name__},
            )

        logger.info(
            "AttentionHookRegistry: registered %d hook(s) across stage(s) %s.",
            hooks_registered,
            stages_list,
        )

    def captures(self) -> list[AttentionCapture]:
        """Return a list of :class:`AttentionCapture` objects, one per hook firing.

        Every tensor in the returned captures is ``.detach().clone()``-ed to
        prevent keeping the autograd graph alive and to give the caller
        independent copies safe for long-lived storage.

        This method is idempotent with respect to the underlying raw captures;
        calling it twice returns two independent sets of clones.

        Returns
        -------
        list[AttentionCapture]
            Ordered by the sequence in which hooks fired during the forward
            pass.  Each element corresponds to one ``LpWindowAttention``
            module invocation.
        """
        result: list[AttentionCapture] = []
        for raw in self._raw_captures:
            capture = AttentionCapture(
                q=_safe_detach_clone(raw.get("q")),
                k=_safe_detach_clone(raw.get("k")),
                q_hat=_safe_detach_clone(raw.get("q_hat")),
                k_hat=_safe_detach_clone(raw.get("k_hat")),
                logits=_safe_detach_clone(raw.get("logits")),
                attention=_safe_detach_clone(raw.get("attention")),
                alpha=_safe_detach_clone(raw.get("alpha")),
                stage_index=int(raw.get("stage_index", -1)),
                block_index=int(raw.get("block_index", -1)),
            )
            result.append(capture)
        return result

    def clear(self) -> None:
        """Discard all accumulated raw captures without removing the hooks.

        Call this between forward passes when you want to reuse the same
        registry for multiple batches but only keep the most recent captures.
        """
        self._raw_captures.clear()
        logger.debug("AttentionHookRegistry: raw captures cleared.")

    def remove(self) -> None:
        """Remove all registered hooks and clear accumulated captures.

        After calling this method the registry is in a clean initial state and
        can be re-used by calling :meth:`register` again.
        """
        for handle in self._hooks:
            handle.remove()
        n_removed = len(self._hooks)
        self._hooks.clear()
        self._raw_captures.clear()
        logger.debug("AttentionHookRegistry: removed %d hook(s).", n_removed)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_detach_clone(value: object) -> Tensor | None:
    """Return a detached clone of ``value`` if it is a :class:`Tensor`, else ``None``.

    Parameters
    ----------
    value : object
        Any value retrieved from a raw capture dict.

    Returns
    -------
    Tensor or None
        A detached, cloned copy of ``value`` if it is a ``Tensor``; ``None``
        otherwise (including if ``value`` is already ``None``).
    """
    if isinstance(value, Tensor):
        return value.detach().clone()
    return None
