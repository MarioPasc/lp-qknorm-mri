"""Custom exception hierarchy for the lpqknorm package.

All domain-specific exceptions inherit from :class:`LpQKNormError`
so callers can catch broad or narrow as needed.
"""

from __future__ import annotations


class LpQKNormError(Exception):
    """Base exception for all lpqknorm errors."""

    def __init__(self, message: str, details: dict[str, object] | None = None) -> None:
        self.details = details or {}
        super().__init__(message)

    def __str__(self) -> str:
        base = super().__str__()
        if self.details:
            return f"{base} | details={self.details}"
        return base


# --- Model exceptions ---


class ModelConfigError(LpQKNormError):
    """Invalid model configuration (e.g., p < 1, negative eps)."""


class PatchingError(LpQKNormError):
    """Failure during SwinUNETR module-tree patching."""


class WeightTransferError(LpQKNormError):
    """Weight shape mismatch or missing key during attention patching."""


class HookError(LpQKNormError):
    """Attention hook registration or capture failure."""


class LpInitError(LpQKNormError):
    """Weight initialization failure (invalid scheme, missing target, etc.)."""


# --- Data exceptions (Phase 1 placeholders) ---


class DataIntegrityError(LpQKNormError):
    """Data pipeline integrity violation."""


class SplitLeakageError(LpQKNormError):
    """Patient ID appears in multiple dataset splits."""


class StratificationError(LpQKNormError):
    """Stratification failed (empty stratum, impossible allocation)."""


# --- Schema / converter exceptions ---


class SchemaValidationError(LpQKNormError):
    """HDF5 file does not conform to the v1.0 schema."""


class ConverterError(LpQKNormError):
    """Dataset converter encountered an unrecoverable error."""
