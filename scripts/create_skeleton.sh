#!/usr/bin/env bash
# Create the project directory skeleton for lp-qknorm-mri.
# Run from the project root: bash scripts/create_skeleton.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "Project root: $ROOT"

# --- Source package directories ---
dirs=(
    src/lpqknorm/data
    src/lpqknorm/models
    src/lpqknorm/training
    src/lpqknorm/probes
    src/lpqknorm/analysis
    src/lpqknorm/utils
    src/lpqknorm/cli
    tests/unit
    tests/integration
    tests/fixtures
    configs
    scripts
)

for d in "${dirs[@]}"; do
    mkdir -p "$ROOT/$d"
    echo "  created $d/"
done

# --- PEP 561 typed marker ---
touch "$ROOT/src/lpqknorm/py.typed"

# --- __init__.py files ---
init_files=(
    src/lpqknorm/__init__.py
    src/lpqknorm/data/__init__.py
    src/lpqknorm/models/__init__.py
    src/lpqknorm/training/__init__.py
    src/lpqknorm/probes/__init__.py
    src/lpqknorm/analysis/__init__.py
    src/lpqknorm/utils/__init__.py
    src/lpqknorm/cli/__init__.py
    tests/__init__.py
    tests/unit/__init__.py
    tests/integration/__init__.py
    tests/fixtures/__init__.py
)

for f in "${init_files[@]}"; do
    touch "$ROOT/$f"
    echo "  created $f"
done

# --- Package __init__.py with version ---
cat > "$ROOT/src/lpqknorm/__init__.py" << 'PYEOF'
"""Lp-QKNorm MRI: Mechanistic study of generalized Lp query-key normalization."""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Mario Pascual González"
__all__: list[str] = []
PYEOF

# --- CLI stubs (needed for entry points in pyproject.toml) ---
for cli_mod in train preprocess probe analyze; do
    cat > "$ROOT/src/lpqknorm/cli/${cli_mod}.py" << PYEOF
"""CLI entry point: ${cli_mod}."""

from __future__ import annotations


def main() -> None:
    """Entry point for lpqknorm-${cli_mod}."""
    raise NotImplementedError("${cli_mod} CLI not yet implemented")
PYEOF
    echo "  created src/lpqknorm/cli/${cli_mod}.py"
done

# --- Empty placeholder modules for future phases ---
touch "$ROOT/src/lpqknorm/utils/exceptions.py"
touch "$ROOT/src/lpqknorm/models/lp_qknorm.py"
touch "$ROOT/src/lpqknorm/models/attention.py"
touch "$ROOT/src/lpqknorm/models/swin_unetr_lp.py"
touch "$ROOT/src/lpqknorm/models/hooks.py"
touch "$ROOT/tests/conftest.py"
touch "$ROOT/tests/unit/test_lp_qknorm.py"
touch "$ROOT/tests/unit/test_attention_equivalence.py"
touch "$ROOT/tests/integration/test_forward_pass.py"

echo ""
echo "Skeleton created. Next: ~/.conda/envs/lpqknorm/bin/pip install -e '.[dev]'"
