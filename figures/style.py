"""Shared plotting style for perturbation-benchmark figures.

Usage:
    from figures.style import apply_style, COLORS, CL_COLORS
    apply_style()
"""
import matplotlib.pyplot as plt

# Text and grid
C_TEXT = "#1F2937"
C_GRID = "#E5E7EB"

# Model colors
COLORS = {
    "Ridge": "#149E9A",
    "MLP": "#0D7377",
    "MORPH": "#6F5CC4",
    "STATE": "#8A94A6",
    "scGPT": "#D9534F",
    "BioLORD": "#E8923E",
    "scGen": "#B34D8F",
    "Direct": "#7AA6C2",
    "Source": "#2F5EAA",
    "SALT": "#C9CED6",
    "Add+Resid": "#D26A3A",
    "Morgan": "#E8923E",
    "CC": "#3572B0",
}

# Cell line colors
CL_COLORS = {
    "k562": "#4A90D9", "K562": "#4A90D9",
    "rpe1": "#D9534F", "RPE1": "#D9534F",
    "hepg2": "#5CB85C", "HepG2": "#5CB85C",
    "jurkat": "#F0AD4E", "Jurkat": "#F0AD4E",
    "C32": "#4A90D9", "HOP62": "#D9534F",
    "HepG2/C3A": "#5CB85C", "Hs 766T": "#F0AD4E",
    "PANC-1": "#6F5CC4",
}

# Model markers
MARKERS = {
    "Ridge": "s", "MLP": "^", "MORPH": "D", "STATE": "o", "scGPT": "v",
    "BioLORD": "P", "scGen": "X",
}

# Decomposition colors
C_TEMPLATE = "#3B5BA9"
C_RESIDUAL = "#149E9A"
C_CONSERVED = "#6F5CC4"
C_INTERACTION = "#D26A3A"


def apply_style():
    """Apply the standard figure style."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.linewidth": 0.6,
        "axes.edgecolor": C_TEXT,
        "xtick.color": C_TEXT,
        "ytick.color": C_TEXT,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "text.color": C_TEXT,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def clean_axes(ax, grid_y=True, grid_x=False):
    """Remove top/right spines, add light grid."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_y:
        ax.yaxis.grid(True, color=C_GRID, linewidth=0.5, zorder=0)
    if grid_x:
        ax.xaxis.grid(True, color=C_GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
