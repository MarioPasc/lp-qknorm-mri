"""Matplotlib publication-style helpers.

Activates the rcParams once via :func:`set_publication_style` and
guarantees deterministic PDF output (no embedded timestamp) so that
the figure-determinism acceptance test (Phase 5 §6) can compare
output bytes across consecutive invocations.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt


# Single-column / double-column figure widths in inches (preprint defaults).
COL_WIDTH_IN: float = 3.5
DOUBLE_COL_WIDTH_IN: float = 7.0


def set_publication_style() -> None:
    """Apply publication rcParams in-place.

    Use a non-interactive backend, set DPI 300, font sizes consistent
    with two-column journal templates, and disable timestamp metadata
    in the PDF backend so two consecutive runs produce byte-identical
    PDFs (matplotlib >= 3.5).
    """
    mpl.use("Agg", force=True)
    plt.rcParams.update(
        {
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "savefig.facecolor": "white",
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.2,
            "lines.markersize": 4.0,
            "pdf.compression": 9,
            # Avoid font subsetting differences across runs.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


# Colour palette tied to the p sweep — keep stable across all figures.
P_PALETTE: dict[str, str] = {
    "vanilla": "#888888",
    "1.0": "#08306b",
    "1.5": "#4292c6",
    "2.0": "#000000",  # baseline emphasised in black
    "2.5": "#fdae6b",
    "3.0": "#e6550d",
    "3.5": "#a63603",
    "4.0": "#67000d",
}


def color_for_p(p_label: str) -> str:
    """Return the canonical colour for a ``p`` label, with a grey fallback."""
    return P_PALETTE.get(p_label, "#999999")


# Deterministic PDF metadata — kwargs to pass to ``plt.savefig``.
PDF_METADATA: dict[str, str | None] = {
    "Creator": "lpqknorm",
    "Producer": "lpqknorm",
    "CreationDate": None,
    "ModDate": None,
}
