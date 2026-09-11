"""
Draw Supplementary Figure S1, the window-convergence analysis.

(a) Spearman rho of eight pre-window statistics against the 20 min window versus window
length, with the 0.9 criterion line; density statistics in dark, the rest in light grey.
(b) Between-batch coefficient of variation versus window length. Frozen tables only.
Inputs: results/tables/convergence_rho_pre.csv, results/tables/convergence_cv_pre.csv
Outputs: results/figures/figS1.{pdf,png}
Run: python code/figures/figS1.py
"""

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
TAB = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "cm", "axes.unicode_minus": False,
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "axes.edgecolor": "#333333",
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.major.size": 3, "ytick.major.size": 3, "legend.frameon": False,
    "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.dpi": 150, "savefig.bbox": None, "savefig.pad_inches": 0.02,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})
P = {"neutral_dark": "#4D4D4D", "neutral_mid": "#767676", "neutral_light": "#CFCECE",
     "neutral_black": "#272727", "signal_dark": "#0B3954", "signal_mid": "#087E8B",
     "accent": "#FF5A5F"}
FIG_W, FIG_H = 4.80, 2.30          # same width as Fig5
TKF, LBF, NTF = 7.0, 8.0, 6.2
COLS = ["density_P", "density_Z", "clustering_P", "clustering_Z",
        "betweenness_P", "betweenness_Z", "lambda2_P", "lambda2_Z"]
LAB = {"density_P": "density (player)", "density_Z": "density (zone)"}


def draw(ax, d, ylab, hline=None, hlab=None):
    x = [w for w in d.index if w <= 20]
    for c in COLS:
        dens = c.startswith("density")
        ax.plot(x, d.loc[x, c], "-", lw=1.1 if dens else 0.7,
                color=P["signal_dark"] if dens else P["neutral_light"],
                alpha=1.0 if dens else 0.9, zorder=4 if dens else 2)
        ax.plot(x, d.loc[x, c], "o", ms=2.4 if dens else 1.6,
                color=P["signal_dark"] if dens else P["neutral_light"],
                zorder=4 if dens else 2)
    if hline is not None:
        ax.axhline(hline, color=P["accent"], lw=0.9, ls=(0, (4, 2.5)), zorder=3)
        ax.text(21.3, hline + 0.055, hlab, fontsize=NTF, color=P["accent"],
                va="center", ha="right")   # label above the line
    ax.axvline(15, color=P["neutral_mid"], lw=0.7, ls=(0, (2, 2)), zorder=1)
    ax.set_xlim(4, 21.5)
    ax.set_xticks([5, 10, 15, 20])
    ax.tick_params(labelsize=TKF)
    ax.set_xlabel("window length (min)", fontsize=LBF, labelpad=2)
    ax.set_ylabel(ylab, fontsize=LBF, labelpad=2)


def main():
    rho = pd.read_csv(TAB / "convergence_rho_pre.csv", index_col=0)
    cv = pd.read_csv(TAB / "convergence_cv_pre.csv", index_col=0)

    # consistency checks
    assert set(COLS) <= set(rho.columns), "convergence table lacks one of the eight statistics"
    assert abs(rho.loc[15, "density_P"] - 0.892) < 0.002, "W15 density_P differs from the text"
    assert abs(rho.loc[15, "density_Z"] - 0.927) < 0.002, "W15 density_Z differs from the text"
    # at W15 only zone density (0.927) reaches 0.9; player density is 0.892
    n_pass = int((rho.loc[15, COLS] >= 0.9).sum())
    assert n_pass == 1, f"expected exactly 1/8 statistics >= 0.9 at W15, got {n_pass}"
    assert (rho.loc[[5, 8, 10, 12], COLS] >= 0.9).sum().sum() == 0, "no statistic should reach 0.9 at shorter windows"
    print(f"  checks passed (eight statistics, W15 density {rho.loc[15,'density_P']:.3f}/"
          f"{rho.loc[15,'density_Z']:.3f}, W15 {n_pass}/8 >= 0.9, shorter windows 0/8)")

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax1 = fig.add_axes([0.115, 0.215, 0.355, 0.640])
    ax2 = fig.add_axes([0.625, 0.215, 0.345, 0.640])
    draw(ax1, rho, "Spearman $\\rho$ vs 20 min", hline=0.9, hlab="criterion 0.9")
    ax1.set_ylim(-0.05, 1.06)
    ax1.set_yticks([0, 0.5, 1.0])
    draw(ax2, cv, "between-batch CV")
    ax2.set_ylim(0, 2.6)
    ax2.set_yticks([0, 1, 2])
    for ax, x_, y_ in [(ax1, 15.4, 0.30), (ax2, 15.4, 2.30)]:
        ax.text(x_, y_, "15 min", fontsize=NTF - 0.6, color=P["neutral_mid"],
                ha="left", va="center")
    ax1.text(5.2, 0.98, "density (zone)", fontsize=NTF, color=P["signal_dark"],
             ha="left", va="center")
    ax1.text(5.2, 0.86, "density (player)", fontsize=NTF, color=P["signal_dark"],
             ha="left", va="center", alpha=0.75)
    ax1.text(5.2, 0.14, "six further statistics", fontsize=NTF, color=P["neutral_mid"],
             ha="left", va="center")
    fig.text(0.006, 0.945, "(a)", fontsize=10, weight="bold", va="center")
    fig.text(0.520, 0.945, "(b)", fontsize=10, weight="bold", va="center")

    for ext, kw in [("pdf", {}), ("png", dict(dpi=300))]:
        fig.savefig(FIG / f"figS1.{ext}", **kw)
    plt.close(fig)
    print("saved figS1.{pdf,png}")


if __name__ == "__main__":
    main()
