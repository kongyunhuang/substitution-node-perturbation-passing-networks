"""
Draw Figure 5, the shared-activity bias and its correction.

(a) Per-competition dumbbells of cross-layer alignment: random-teammate null, naive, and with
the substituted players excluded (11 competitions). (b) Dose-response on La Liga: alignment
versus the substituted players' activity share, naive and corrected, quintile means over the
per-batch cloud. (c) Synthetic generator at lambda = 0 and 0.4, closed form vs simulation inset.
Inputs: results/tables/{bias_replication,bias_predictability,synthetic_validation,closedform_full,mvp_closedform}.csv
Outputs: results/figures/fig5.{pdf,png}
Run: python code/figures/fig5.py
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
# canvas; keep the same width across the main figures
FIG_W, FIG_H = 4.80, 3.97
TKF, LBF, NTF = 7.0, 8.0, 6.2
LABEL = {"WC22": "World Cup 2022", "Euro24": "Euro 2024", "Euro20": "Euro 2020",
         "ISL": "Indian Super L. 21/22", "WSL": "FA WSL 20/21", "Bund23": "Bundesliga 23/24",
         "Bund15": "Bundesliga 15/16", "LaLiga15": "La Liga 15/16", "PL15": "Premier L. 15/16",
         "SerieA15": "Serie A 15/16", "LaLiga23": "La Liga 23/24"}


def panel_a(fig, R):
    """(a) Per-competition dumbbells: null / naive / perturber excluded."""
    ax = fig.add_axes([0.245, 0.128, 0.245, 0.790])
    R = R.sort_values("naive").reset_index(drop=True)
    ys = np.arange(len(R))
    ax.plot([0, 0], [-0.45, len(R) - 0.75], color=P["neutral_mid"], lw=0.7,
            ls=(0, (4, 3)), zorder=1)
    for y, r in zip(ys, R.itertuples()):
        ax.plot([r.corrected, r.naive], [y, y], "-", lw=0.8, color=P["neutral_light"], zorder=2)
        if np.isfinite(r.null):
            ax.plot([r.null], [y], "|", ms=4.5, mew=1.0, color=P["neutral_mid"], zorder=3)
        ax.plot([r.naive], [y], "o", ms=3.4, color=P["neutral_dark"], zorder=4)
        ax.plot([r.corrected], [y], "o", ms=3.6, color=P["accent"], zorder=5)
    main = R.comp == "LaLiga23"
    ax.set_yticks(ys)
    ax.set_yticklabels([LABEL[c] for c in R.comp], fontsize=NTF)
    for t, m in zip(ax.get_yticklabels(), main):
        if m:
            t.set_weight("bold")
    ax.tick_params(axis="y", length=0, pad=1.5)
    ax.tick_params(axis="x", labelsize=TKF)
    ax.set_xlim(-0.06, 0.375)
    ax.set_xticks([0, 0.1, 0.2, 0.3])
    ax.set_ylim(-0.9, len(R) - 0.15)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("cross-layer alignment  cos($\\Delta B$, $\\Delta z$)", fontsize=LBF - 0.6,
                  labelpad=2)
    ax.text(0.345, len(R) - 0.52, "naive", fontsize=NTF, color=P["neutral_dark"],
            ha="center", va="center")
    ax.text(0.014, len(R) - 0.52, "perturber excluded", fontsize=NTF,
            color=P["accent"], ha="left", va="center")
    ax.text(R.null.mean(), -0.72, "random-teammate null", fontsize=NTF - 0.8,
            color=P["neutral_mid"], ha="center", va="center")


def panel_b(fig, bp):
    """(b) Dose-response: alignment vs substituted players' activity share (La Liga, 1,461 batches)."""
    ax = fig.add_axes([0.640, 0.605, 0.330, 0.313])
    ax.plot(bp.perturber_share, bp.naive, ".", ms=1.2, color=P["neutral_light"], alpha=0.5,
            mec="none", zorder=2)
    q = pd.qcut(bp.perturber_share, 5, labels=False)
    for col, c, lab in [("naive", P["neutral_dark"], "naive"),
                        ("lpo", P["accent"], "perturber excluded")]:
        g = bp.groupby(q).agg(x=("perturber_share", "mean"), m=(col, "mean"),
                              s=(col, lambda v: v.std(ddof=1) / np.sqrt(len(v))))
        ax.errorbar(g.x, g.m, yerr=1.96 * g.s, fmt="o-", ms=3.0, lw=1.1, color=c,
                    elinewidth=0.8, capsize=1.5, zorder=4)
        ax.text(g.x.iloc[-1] + 0.012, g.m.iloc[-1] + (0.075 if col == "lpo" else 0.0),
                lab, fontsize=NTF, color=c, va="center", ha="left")
    ax.plot([0.02, 0.30], [0, 0], color=P["neutral_mid"], lw=0.7, ls=(0, (4, 3)), zorder=1)
    ax.set_xlim(0.02, 0.40)
    ax.set_ylim(-0.55, 0.85)
    ax.set_xticks([0.1, 0.2, 0.3])
    ax.set_yticks([0, 0.4, 0.8])
    ax.tick_params(labelsize=TKF)
    ax.set_xlabel("substituted players' share of window activity", fontsize=LBF - 0.6,
                  labelpad=2)
    ax.set_ylabel("alignment", fontsize=LBF - 0.6, labelpad=2)


def panel_c(fig, syn, cf, mvp):
    """(c) Synthetic truth at lambda = 0 and 0.4, closed form vs simulation inset."""
    ax = fig.add_axes([0.640, 0.128, 0.330, 0.313])
    ax.axhline(0, color=P["neutral_mid"], lw=0.7, ls=(0, (4, 3)), zorder=1)
    for lam, ls_ in [(0.0, (0, (2.5, 1.5))), (0.4, "-")]:
        d = syn[(syn.lam == lam) & (syn.exp.isin(["A", "C"]))].copy()
        if not len(d):
            continue
        d["q"] = pd.qcut(d.share, 6, labels=False, duplicates="drop")
        for col, c in [("naive", P["neutral_dark"]), ("lpo", P["accent"])]:
            g = d.groupby("q").agg(x=("share", "mean"), m=(col, "mean"))
            ax.plot(g.x, g.m, ls=ls_, lw=1.1, color=c, zorder=3)
            ax.plot(g.x, g.m, "o", ms=2.2, color=c, zorder=4)
    ax.text(0.435, 0.90, "naive", fontsize=NTF, color=P["neutral_dark"], ha="left", va="center")
    ax.text(0.435, 0.40, "$\\lambda$ = 0.4", fontsize=NTF - 0.8, color=P["accent"],
            ha="left", va="center")
    ax.text(0.435, 0.03, "$\\lambda$ = 0", fontsize=NTF - 0.8, color=P["accent"],
            ha="left", va="center")
    ax.set_xlim(0.02, 0.50)
    ax.set_ylim(-0.12, 1.05)
    ax.set_xticks([0.1, 0.3, 0.5])
    ax.set_yticks([0, 0.5, 1.0])
    ax.tick_params(labelsize=TKF)
    ax.set_xlabel("synthetic perturber share", fontsize=LBF - 0.6, labelpad=2)
    ax.set_ylabel("alignment", fontsize=LBF - 0.6, labelpad=2)   # alignment statistic, not the lambda scale

    ins = ax.inset_axes([0.375, 0.13, 0.315, 0.32])
    th = np.concatenate([cf.theory.values, mvp.theory.values])
    si = np.concatenate([cf.sim.values, mvp.sim.values])
    lim = max(th.max(), si.max()) * 1.08
    ins.plot([0, lim], [0, lim], "-", lw=0.7, color=P["neutral_mid"], zorder=1)
    ins.plot(th, si, "o", ms=1.8, color=P["signal_dark"], mec="none", alpha=0.8, zorder=2)
    ins.set_xlim(0, lim); ins.set_ylim(0, lim)
    ins.set_xticks([]); ins.set_yticks([])
    ins.set_facecolor("none")
    for sp in ins.spines.values():
        sp.set_color(P["neutral_light"]); sp.set_linewidth(0.6); sp.set_visible(True)
    ax.text(0.533, 0.475, f"closed form vs simulation, {len(th)} settings",
            transform=ax.transAxes, fontsize=NTF - 1.2, color=P["neutral_mid"],
            ha="center", va="bottom")


def main():
    R = pd.read_csv(TAB / "bias_replication.csv")
    bp = pd.read_csv(TAB / "bias_predictability.csv")
    syn = pd.read_csv(TAB / "synthetic_validation.csv")
    cf = pd.read_csv(TAB / "closedform_full.csv")
    mvp = pd.read_csv(TAB / "mvp_closedform.csv")

    # consistency checks
    assert len(R) == 11, f"expected 11 competitions, got {len(R)}"
    assert (R.naive > 0).all(), "naive alignment should be positive in all 11 competitions"
    assert (R.p_corrected > 0.05).all(), \
        f"corrected alignment should be n.s. in all 11 competitions, min p = {R.p_corrected.min():.3f}"
    m = R[R.comp == "LaLiga23"].iloc[0]
    assert abs(m.naive - 0.284) < 0.002 and abs(m.corrected) < 0.002, \
        f"La Liga should give naive=+0.284 / corrected=-0.000, got {m.naive:+.3f}/{m.corrected:+.3f}"
    qm = bp.groupby(pd.qcut(bp.perturber_share, 5, labels=False)).naive.mean()
    assert (np.diff(qm.values) > 0).all(), f"naive alignment should rise monotonically with share, got {qm.round(3).tolist()}"
    rel = np.concatenate([cf.rel_err.values, mvp.rel_err.values])
    assert np.median(rel) < 0.05, f"closed-form median relative error should be < 0.05, got {np.median(rel):.3f}"
    print(f"  checks passed (11 competitions, naive all positive, mean {R.naive.mean():+.3f}, corrected all n.s., "
          f"min p={R.p_corrected.min():.3f}, La Liga {m.naive:+.3f}/{m.corrected:+.3f}, "
          f"quintiles monotone, closed-form median error {np.median(rel):.3f})")

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    panel_a(fig, R)
    panel_b(fig, bp)
    panel_c(fig, syn, cf, mvp)
    fig.text(0.006, 0.955, "(a)", fontsize=10, weight="bold", va="center")
    fig.text(0.520, 0.955, "(b)", fontsize=10, weight="bold", va="center")
    fig.text(0.520, 0.478, "(c)", fontsize=10, weight="bold", va="center")

    for ext, kw in [("pdf", {}), ("png", dict(dpi=300))]:
        fig.savefig(FIG / f"fig5.{ext}", **kw)
    plt.close(fig)
    print("saved fig5.{pdf,png}")


if __name__ == "__main__":
    main()
