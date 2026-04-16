"""Shared pytest fixtures for the lpqknorm test suite."""

from __future__ import annotations

import pytest
import torch

from lpqknorm.data.datamodule import MockAtlasDataModule, MockDataConfig
from lpqknorm.models.lp_qknorm import LpQKNormConfig
from lpqknorm.training.module import ModelConfig, TrainingConfig


@pytest.fixture(params=[1.5, 2.0, 2.5, 3.0, 4.0, 8.0])
def p_value(request: pytest.FixtureRequest) -> float:
    """Parametrized p values covering the full test range."""
    return request.param  # type: ignore[return-value]


@pytest.fixture
def lp_cfg_p2() -> LpQKNormConfig:
    """LpQKNormConfig with p=2 (Henry et al. baseline)."""
    return LpQKNormConfig(p=2.0)


@pytest.fixture
def lp_cfg_p3() -> LpQKNormConfig:
    """LpQKNormConfig with p=3."""
    return LpQKNormConfig(p=3.0)


@pytest.fixture
def small_qk() -> tuple[torch.Tensor, torch.Tensor]:
    """Small Q and K tensors for fast unit tests. Shape (4, 16, 64)."""
    g = torch.Generator().manual_seed(42)
    q = torch.randn(4, 16, 64, generator=g)
    k = torch.randn(4, 16, 64, generator=g)
    return q, k


@pytest.fixture
def model_kwargs() -> dict[str, object]:
    """Standard kwargs for build_swin_unetr_lp in tests."""
    return {
        "img_size": (224, 224),
        "in_channels": 1,
        "out_channels": 2,
        "feature_size": 24,
    }


# ---------------------------------------------------------------------------
# Phase 3 fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_datamodule() -> MockAtlasDataModule:
    """Tiny MockAtlasDataModule for fast integration tests."""
    cfg = MockDataConfig(
        n_train=8,
        n_val=8,
        n_test=4,
        img_size=(224, 224),
        n_subjects=4,
        seed=0,
        batch_size=2,
    )
    dm = MockAtlasDataModule(cfg)
    dm.setup("fit")
    return dm


@pytest.fixture
def tiny_model_cfg() -> ModelConfig:
    """Minimal ModelConfig (feature_size=12) for fast tests."""
    return ModelConfig(feature_size=12, out_channels=1)


@pytest.fixture
def tiny_training_cfg() -> TrainingConfig:
    """TrainingConfig suitable for CPU-based tests."""
    return TrainingConfig(lr=1e-3, max_epochs=2, patience=5, precision="32")
