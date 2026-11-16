"""Shared figure style: palette and matplotlib defaults (light mode, white background)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent

SURFACE = "#ffffff"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8985"
GRID_C = "#e4e3df"
COLOR = {"octree": "#2a78d6", "leaf": "#1baf7a"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID_C, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
    "axes.titleweight": "semibold", "axes.titlelocation": "left", "lines.linewidth": 2,
    "legend.frameon": False, "axes.axisbelow": True,
})
