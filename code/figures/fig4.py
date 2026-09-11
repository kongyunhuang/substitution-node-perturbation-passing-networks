"""
Draw Figure 4, screening of 36 candidate predictors of reorganization magnitude.

(a) Invariance corridor: mean-removed high-minus-low difference of the response-versus-minute
profile for 34 median-split factors, against a +-1.96 SE corridor. (b) Mediation plane: correlation
with minute versus own effect (beta/SE). (c) Half-normal QQ of |z| against the dependence-aware MC envelope.
Inputs: results/tables/{mechanism_mine,invariance_curves,invariance_collapse,volume_artifact,
        mediation_plane,mechanism_null_mc,mechanism_null_global}.csv
Outputs: results/figures/fig4.{pdf,png}
Run: python code/figures/fig4.py
"""

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

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
FIG_W, FIG_H = 5.20, 4.30
TKF, LBF, NTF = 7.0, 8.0, 6.2
A_B, A_H = 2.55 / FIG_H, 1.42 / FIG_H     # (a) axes bottom / height
LOW_B, LOW_H = 0.58 / FIG_H, 1.30 / FIG_H  # (b)(c) axes bottom / height
NAME = {"trailing": "trailing", "t_remain": "time remaining", "margin": "score margin",
        "pre_btw_Z": "pre-window betweenness"}


def zabs(p):
    return norm.isf(np.clip(np.asarray(p, float), 1e-300, 1.0) / 2.0)


def panel_a(fig, cur, coll):
    """(a) Invariance corridor: mean-removed high-minus-low differences inside the noise corridor."""
    ax = fig.add_axes([0.115, A_B, 0.800, A_H])
    d = cur[cur.factor != "__all__"].copy()
    for (f, sp), g in d.groupby(["factor", "split"]):
        d.loc[g.index, "cen"] = g["mean"] - g["mean"].mean()      # mean-remove each half
    x = np.sort(d.minute_mid.unique())
    # corridor: 1.96 x median combined SE per bin
    band = []
    for b in x:
        s = d[d.minute_mid == b]
        band.append(1.96 * float(np.median(np.sqrt(s.se ** 2 + s.se ** 2))))
    band = np.array(band)
    xb = np.concatenate([[x[0] - 1.4], x, [x[-1] + 1.4]])
    bb = np.concatenate([[band[0]], band, [band[-1]]])
    ax.fill_between(xb, -bb, bb, color=P["neutral_light"], alpha=0.5, lw=0, zorder=1)
    ax.axhline(0, color=P["neutral_mid"], lw=0.7, zorder=2)

    top = coll.nlargest(1, "ratio_cen").factor.iloc[0]
    keep = set(coll.factor)                      # 34 factors with complete high/low pairs
    rng = np.random.default_rng(7)
    topx, topy = [], []
    for f, g in d.groupby("factor"):
        if f not in keep:
            continue
        g = g.sort_values("minute_mid")
        hi = g[g.split == "high"].set_index("minute_mid").cen
        lo = g[g.split == "low"].set_index("minute_mid").cen
        diff = (hi - lo).dropna()
        if f == top:
            topx, topy = list(diff.index), list(diff.values)
            continue
        ax.plot(diff.index + rng.uniform(-1.1, 1.1, len(diff)), diff.values, "o", ms=1.9,
                mfc=P["neutral_mid"], mec="none", alpha=0.55, zorder=3)
    ax.plot(topx, topy, "-o", lw=0.9, ms=2.2, color=P["neutral_dark"], zorder=4)
    r = coll.set_index("factor").loc[top]
    ax.text(x[-1] + 0.9, topy[-1], f"{NAME.get(top, top)}\n{r.ratio_cen:.2f}$\\times$ noise",
            fontsize=NTF, color=P["neutral_dark"], va="center", ha="left", linespacing=1.2)
    ax.set_xlim(x[0] - 2.5, x[-1] + 2.0)
    ax.set_ylim(-0.60, 0.42)   # room for the inset at the bottom
    ax.set_xticks([45, 55, 65, 75])
    ax.set_yticks([-0.3, 0, 0.3])
    ax.tick_params(labelsize=TKF)
    ax.set_xlabel("substitution minute", fontsize=LBF, labelpad=2)
    ax.set_ylabel("difference between\nsplit halves (SD)", fontsize=LBF, labelpad=2,
                  linespacing=1.15)
    ax.text(0.0, 1.03, "each dot: one of 34 candidate factors in one time bin, batches split "
            "at the factor's median; grey corridor: $\\pm$1.96 SE",
            transform=ax.transAxes, fontsize=NTF - 0.6, color=P["neutral_mid"],
            ha="left", va="bottom")

    # inset: two halves, mean-removed, difference = one dot
    ins = ax.inset_axes([0.015, 0.015, 0.170, 0.215])
    tx = np.array([0, 1, 2])
    hi_ = np.array([0.55, 0.15, -0.70])
    lo_ = np.array([-0.55, -0.10, 0.65])
    ins.plot(tx, hi_, "-", lw=0.9, color=P["neutral_dark"])
    ins.plot(tx, lo_, "--", lw=0.9, color=P["neutral_mid"])
    ins.annotate("", xy=(1, hi_[1]), xytext=(1, lo_[1]),
                 arrowprops=dict(arrowstyle="<->", lw=0.7, color=P["neutral_black"],
                                 shrinkA=0.5, shrinkB=0.5))
    ins.text(1.30, 0.02, "= one dot", fontsize=NTF - 1.4, color=P["neutral_black"],
             va="center", ha="left")
    ins.text(-0.15, hi_[0], "high", fontsize=NTF - 1.4, color=P["neutral_dark"],
             va="center", ha="right")
    ins.text(-0.15, lo_[0], "low", fontsize=NTF - 1.4, color=P["neutral_mid"],
             va="center", ha="right")
    ins.set_xlim(-0.95, 2.9)
    ins.set_ylim(-1.05, 1.05)
    ins.set_xticks([]); ins.set_yticks([])
    for sp in ins.spines.values():
        sp.set_visible(False)
    ins.set_facecolor("none")
    ins.text(0.0, 1.06, "two halves, each mean-removed",
             transform=ins.transAxes, fontsize=NTF - 1.4, color=P["neutral_mid"],
             ha="left", va="bottom")


def panel_b(fig, med):
    """(b) Mediation plane: correlation with minute x own effect."""
    ax = fig.add_axes([0.115, LOW_B, 0.330, LOW_H])
    anchor = med.predictor.isin(["minute", "t_remain"])          # timing itself, reference only
    d = med[~anchor]
    ax.axhline(0, color=P["neutral_mid"], lw=0.7, ls=(0, (4, 3)), zorder=1)
    ax.axvline(0, color=P["neutral_mid"], lw=0.7, ls=(0, (4, 3)), zorder=1)
    for v in (-1.96, 1.96):
        ax.plot([-1.05, 1.05], [v, v], color=P["neutral_light"], lw=0.6, ls=(0, (1, 1.6)),
                zorder=1)
    sig = d.q_main < 0.05
    ax.plot(d.corr_minute[~sig], d.z_main[~sig], "o", ms=2.8, mfc="white",
            mec=P["neutral_mid"], mew=0.7, zorder=3)
    ax.plot(d.corr_minute[sig], d.z_main[sig], "o", ms=3.4, color=P["signal_dark"], zorder=4)
    for _, r in med[anchor].iterrows():                          # open squares for timing itself
        ax.plot([r.corr_minute], [r.z_main], "s", ms=3.0, mfc="white",
                mec=P["neutral_dark"], mew=0.8, zorder=4)
    ax.text(0.98, 2.25, "substitution minute\nitself (reference)", fontsize=NTF - 1.2,
            color=P["neutral_dark"], ha="right", va="center", linespacing=1.15)
    top = d.reindex((d.corr_minute.abs() * (d.z_main.abs() > 1.96)).sort_values().index[-1:])
    # readable label instead of the variable name
    disp = {"batch_seq": "batch sequence"}
    for _, r in top.iterrows():
        ax.annotate(f"{disp.get(r.predictor, str(r.predictor).replace('_', ' '))}\n$q$ = {r.q_main:.2f}",
                    xy=(r.corr_minute, r.z_main), xytext=(r.corr_minute - 0.16, r.z_main - 1.85),
                    fontsize=NTF - 0.8, color=P["neutral_dark"], ha="center", va="center",
                    linespacing=1.15,
                    arrowprops=dict(arrowstyle="-", lw=0.6, color=P["neutral_mid"],
                                    shrinkA=0, shrinkB=2))
    ax.set_xlim(-1.12, 1.12)
    ax.set_ylim(-5.4, 5.4)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-4, 0, 4])
    ax.tick_params(labelsize=TKF)
    ax.set_xlabel("correlation with substitution minute", fontsize=LBF - 0.6, labelpad=2)
    ax.set_ylabel("own effect on\nreorganization ($\\beta$/SE)", fontsize=LBF - 0.6,
                  labelpad=2, linespacing=1.15)
    ax.text(0.0, 1.02, "a mediator must sit far from both dashed lines",
            transform=ax.transAxes, fontsize=NTF - 0.6, color=P["neutral_mid"],
            ha="left", va="bottom")


def panel_c(fig, m, env, glob):
    """(c) Half-normal QQ with the dependence-aware null envelope."""
    ax = fig.add_axes([0.625, LOW_B, 0.345, LOW_H])
    e = env[env.family == "interaction"].sort_values("k")
    x, lo, hi = e.x_expected.values, e.lo.values, e.hi.values
    ax.fill_between(x, lo, hi, color=P["neutral_light"], alpha=0.5, lw=0, zorder=1)
    ax.plot([0, x[-1]], [0, x[-1]], color=P["neutral_mid"], lw=0.8, ls=(0, (4, 3)), zorder=2)
    pm = {r.family: r.mc_p for _, r in glob[glob.stat == "mean_abs_z"].iterrows()}
    for lab, p, col, mk in [("naive", m.p_main_naive, P["neutral_dark"], "s"),
                            ("residualized", m.p_main, P["signal_dark"], "o"),
                            ("$\\times$ timing", m.p_inter, P["accent"], "^")]:
        y = np.sort(zabs(p))
        ax.plot(x, y, mk, ms=2.2, color=col, mec="white", mew=0.3, zorder=4)
        ax.plot(x, y, "-", lw=0.6, color=col, alpha=0.5, zorder=3)
        ax.text(x[-1] + 0.06, y[-1], lab, fontsize=NTF - 0.4, color=col, va="center",
                ha="left")
    ax.text(x[-1] + 0.06, zabs(m.p_inter).max() - 0.62,
            f"MC $p$ = {pm['interaction']:.2f}", fontsize=NTF - 0.8, color=P["accent"],
            va="center", ha="left")
    ax.set_xlim(0, 3.55)
    ax.set_ylim(-0.2, 7.6)
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 2, 4, 6])
    ax.tick_params(labelsize=TKF)
    ax.set_xlabel("expected $|z|$ (null)", fontsize=LBF, labelpad=2)
    ax.set_ylabel("observed $|z|$", fontsize=LBF, labelpad=2)


def main():
    m = pd.read_csv(TAB / "mechanism_mine.csv")
    cur = pd.read_csv(TAB / "invariance_curves.csv")
    coll = pd.read_csv(TAB / "invariance_collapse.csv")
    va = pd.read_csv(TAB / "volume_artifact.csv")
    med = pd.read_csv(TAB / "mediation_plane.csv")
    env = pd.read_csv(TAB / "mechanism_null_mc.csv")
    glob = pd.read_csv(TAB / "mechanism_null_global.csv")

    # consistency checks
    assert len(m) == 36, f"expected 36 candidates, got {len(m)}"
    n3 = ((m.q_naive < 0.05).sum(), (m.q_main < 0.05).sum(), (m.q_inter < 0.05).sum())
    assert n3 == (23, 6, 0), f"expected hits 23/6/0, got {n3}"
    assert coll.ratio_cen.max() < 2.0 and 0.8 < coll.ratio_cen.median() < 1.3, \
        f"ratio_cen out of range: max {coll.ratio_cen.max():.2f}, median {coll.ratio_cen.median():.2f}"
    g = glob.set_index(["family", "stat"]).observed
    assert g[("interaction", "n_outside_pointwise_95")] == 0 and \
        g[("main", "n_outside_pointwise_95")] >= 30, "unexpected counts outside the dependence-aware envelope"
    assert 0.45 < va.r2_nonlinear.iloc[0] < 0.60, f"volume artifact R^2 = {va.r2_nonlinear.iloc[0]:.3f} out of range"
    assert len(med) == 36, f"(b) expected 36 factors in mediation_plane.csv, got {len(med)}"
    cand = med[(med.corr_minute.abs() > 0.2) & (med.z_main.abs() > 1.96) &
               (~med.predictor.isin(["minute", "t_remain"]))]
    assert len(cand) <= 1 and (cand.q_main >= 0.05).all(), \
        f"(b) expected no FDR-significant factor in the mediator corner, got {list(cand.predictor)}"
    assert len(coll) == 34, f"(a) expected 34 factors (minute and opp_sub dropped), got {len(coll)}"
    print(f"  checks passed (36 candidates, 23/6/0, ratio_cen max {coll.ratio_cen.max():.2f}/"
          f"median {coll.ratio_cen.median():.2f}, interaction 0/36 outside, R^2={va.r2_nonlinear.iloc[0]:.2f})")

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    panel_a(fig, cur, coll)
    panel_b(fig, med)
    panel_c(fig, m, env, glob)
    fig.text(0.006, (A_B * FIG_H + A_H * FIG_H + 0.17) / FIG_H, "(a)", fontsize=10,
             weight="bold", va="center")
    ylet = (LOW_B * FIG_H + LOW_H * FIG_H + 0.15) / FIG_H
    fig.text(0.006, ylet, "(b)", fontsize=10, weight="bold", va="center")
    fig.text(0.510, ylet, "(c)", fontsize=10, weight="bold", va="center")

    for ext, kw in [("pdf", {}), ("png", dict(dpi=300))]:
        fig.savefig(FIG / f"fig4.{ext}", **kw)
    plt.close(fig)
    print("saved fig4.{pdf,png}")


if __name__ == "__main__":
    main()
