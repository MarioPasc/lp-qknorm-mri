#!/usr/bin/env python
"""Pre-training environment verification.

Run before any training job to confirm all dependencies are present,
versions match expectations, and deterministic algorithms can be enabled.
Exits with code 1 on any failure.

Usage::

    python scripts/verify_env.py
"""

from __future__ import annotations

import importlib.metadata
import json
import sys


def _check_package(name: str, min_version: str | None = None) -> str:
    """Return installed version of *name*, or exit on failure."""
    try:
        ver = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        print(f"FAIL: package '{name}' not installed.", file=sys.stderr)
        sys.exit(1)
    if min_version is not None:
        from packaging.version import Version

        if Version(ver) < Version(min_version):
            print(
                f"FAIL: {name}=={ver} < required {min_version}",
                file=sys.stderr,
            )
            sys.exit(1)
    return ver


def main() -> None:
    """Run all environment checks and print a summary."""
    env: dict[str, str] = {}

    # --- Required packages ---
    env["torch"] = _check_package("torch", "2.1.0")
    env["monai"] = _check_package("monai", "1.3.0")
    env["pytorch-lightning"] = _check_package("pytorch-lightning", "2.1.0")
    env["lpqknorm"] = _check_package("lpqknorm")
    env["hydra-core"] = _check_package("hydra-core", "1.3.0")
    env["h5py"] = _check_package("h5py")
    env["pandas"] = _check_package("pandas")
    env["scipy"] = _check_package("scipy")
    env["torchmetrics"] = _check_package("torchmetrics")

    # --- PyTorch CUDA ---
    import torch

    env["torch.cuda.is_available"] = str(torch.cuda.is_available())
    if torch.cuda.is_available():
        env["torch.cuda.device_name"] = torch.cuda.get_device_name(0)
        env["torch.version.cuda"] = torch.version.cuda or "N/A"
    else:
        print(
            "WARN: CUDA not available. Training will run on CPU.",
            file=sys.stderr,
        )

    # --- Deterministic algorithms ---
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        env["deterministic_algorithms"] = "ok (warn_only=True)"
    except Exception as e:
        print(f"FAIL: deterministic algorithms: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Summary ---
    print(json.dumps(env, indent=2))
    print("\nAll checks passed.", file=sys.stderr)


if __name__ == "__main__":
    main()
