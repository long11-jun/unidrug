"""
Unified Nature/Science-style plotting configuration for DP visualization figures.
Use: from _style import apply_style, PALETTE, ...
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Nature-style muted palette
PALETTE = {
    "blue": "#4C72B0",
    "orange": "#DD8452",
    "green": "#55A868",
    "red": "#C44E52",
    "purple": "#8172B3",
    "yellow": "#CCB974",
    "cyan": "#64B5CD",
}
PALETTE_LIST = [PALETTE["blue"], PALETTE["orange"], PALETTE["green"], PALETTE["red"], PALETTE["purple"]]

# Figure dimensions (inches)
SINGLE_COL = 3.5
DOUBLE_COL = 7.0
TRIPLE_COL = 10.0


def apply_style():
    """Set matplotlib rcParams for publication figures."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 12,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,
        "grid.alpha": 0.3,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


def auto_annotate(ax, values, fmt=".2f", offset=0.02, vertical=True):
    """Add text annotations above/beside bars or points."""
    if vertical:
        for i, v in enumerate(values):
            if np.isfinite(v):
                ax.text(i, v + offset, format(v, fmt), ha="center", va="bottom", fontsize=7, rotation=0)
    else:
        for i, v in enumerate(values):
            if np.isfinite(v):
                ax.text(v + offset, i, format(v, fmt), ha="left", va="center", fontsize=7)


def gini_coefficient(weights):
    """Gini coefficient for attention concentration (0 = uniform, 1 = max concentration)."""
    w = np.asarray(weights).flatten()
    w = np.sort(w[w >= 0])
    n = len(w)
    if n == 0:
        return 0.0
    return (2 * np.sum((np.arange(1, n + 1)) * w) - (n + 1) * np.sum(w)) / (n * np.sum(w) + 1e-12)


def lorenz_curve(weights):
    """Cumulative share of attention (sorted ascending). Returns (x, y) for plotting."""
    w = np.asarray(weights).flatten()
    w = w[w >= 0]
    n = len(w)
    if n == 0:
        return np.array([0, 1]), np.array([0, 1])
    w = np.sort(w)
    cum = np.cumsum(w)
    cum = cum / (cum[-1] + 1e-12)
    x = np.linspace(0, 1, n + 1)
    y = np.concatenate([[0], cum])
    return x, y
