"""
Draw Figure 3, territorial displacement and its competitive consequences.

(a) Early/mid/late pitches with significant 10 yd origin-share bins coloured and the 12-bin DiD
profile as bars below on the same x axis. (b) Subgroup means with 95% CI over the per-substitution
cloud. (c) Standardized effects of 11 outcomes against a |r| < 0.05 band; only territory survives.
Inputs: results/tables/{territory_profile.npz,territory_subgroups.csv,dterr_audit.csv,consequence_mine.csv},
        data/analysis_table.parquet
Outputs: results/figures/fig3.{pdf,png}
Run: python code/figures/fig3.py
"""

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Rectangle
from mplsoccer import Pitch
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
     "neutral_black": "#272727", "signal_dark": "#0B3954", "accent": "#FF5A5F"}
CMAP = plt.cm.RdBu_r
COL_POS, COL_NEG = CMAP(0.80), CMAP(0.20)       # red = origin share up, blue = down
PITCH_LINE = "#B9BEC4"
GROUPS = [("strategic", "early  (HT–60$'$)"), ("regular", "mid  (61–75$'$)"),
          ("late", "late  (>75$'$)")]

# layout in inches -> figure fractions; pitches must be exactly 1.5:1 or mplsoccer shrinks them
FIG_W, FIG_H = 5.20, 4.20
A_BOT = 1.75                                    # bottom of block (a), in
PW_IN, PH_IN = 1.36, 1.36 / 1.5                 # one pitch, 1.5:1
COL_L = [0.1000, 0.4077, 0.7154]                # left edges of the three columns
PW, PH = PW_IN / FIG_W, PH_IN / FIG_H
# from A_BOT up: 0.55 footnote | 0.50 bar strip | 0.05 gap | pitch | 0.26 titles
PITCH_B = (A_BOT + 1.10) / FIG_H
STRIP_T, STRIP_H = (A_BOT + 1.05) / FIG_H, 0.50 / FIG_H
X_MID = (COL_L[0] + COL_L[2] + PW) / 2          # horizontal centre of the three pitches
YLIM = 8.6                                      # shared y range of the bar strips
TKF, LBF, TTF, NTF = 7.0, 8.0, 7.6, 6.2         # tick / label / title / note font sizes
LOW_B, LOW_H = 0.40 / FIG_H, 1.20 / FIG_H       # (b)(c) axes bottom and height


def panel_a(fig, tp, da):
    """(a) Three pitches with the 12-bin bar strip below, shared x axis."""
    pitch = Pitch(pitch_type="statsbomb", line_color=PITCH_LINE, linewidth=0.6,
                  line_zorder=2, pad_left=0, pad_right=0, pad_top=0, pad_bottom=0)
    for i_, (g, lab) in enumerate(GROUPS):
        prof, se = tp[f"prof_{g}"] * 100, tp[f"se_{g}"] * 100
        edges, ctr = tp["edges"], tp["centers"]
        sigm = np.abs(prof) >= 2 * se
        # printed net shift = group mean from dterr_audit.csv; the profile moment differs by binning
        net = float(da[da.timing == g].did_terr_excl.mean())
        n_ = int(tp[f"n_{g}"][0])
        pre_x = float(tp[f"pre_x_{g}"][0])
        is_late = g == "late"

        # top row: pitch, significant bins only
        axp = fig.add_axes([COL_L[i_], PITCH_B, PW, PH])
        pitch.draw(ax=axp)
        for k in range(len(prof)):
            if not sigm[k]:
                continue
            axp.add_patch(Rectangle((edges[k], 0), edges[k + 1] - edges[k], 80,
                                    facecolor=COL_POS if prof[k] > 0 else COL_NEG,
                                    alpha=0.45, edgecolor="none", zorder=1))
        axp.plot([pre_x, pre_x], [12, 68], color=P["neutral_mid"], lw=0.7,
                 ls=(0, (2.2, 1.6)), zorder=3)
        axp.set_title(lab, fontsize=TTF, pad=10.5, color=P["neutral_black"],
                      weight="bold" if is_late else "normal")
        axp.text(0.5, 1.035, f"$n$ = {n_}", fontsize=NTF, transform=axp.transAxes,
                 ha="right", va="bottom", color=P["neutral_mid"])
        axp.text(0.56, 1.035, f"net {net:+.2f} yd", fontsize=NTF,
                 transform=axp.transAxes, ha="left", va="bottom",
                 color=P["signal_dark"] if is_late else P["neutral_mid"],
                 weight="bold" if is_late else "normal")

        # bottom row: bar strip, same x axis
        axb = fig.add_axes([COL_L[i_], STRIP_T - STRIP_H, PW, STRIP_H])
        for k in range(len(prof)):
            axb.bar(ctr[k], prof[k], width=9.0, zorder=3,
                    color=COL_POS if prof[k] > 0 else COL_NEG,
                    alpha=0.95 if sigm[k] else 0.30,
                    edgecolor=P["neutral_black"] if sigm[k] else "none", linewidth=0.5)
            axb.plot([ctr[k], ctr[k]], [prof[k] - se[k], prof[k] + se[k]],
                     color=P["neutral_black"] if sigm[k] else P["neutral_mid"],
                     lw=0.7, zorder=4, alpha=0.85 if sigm[k] else 0.40)
        axb.axhline(0, color=P["neutral_mid"], lw=0.7, zorder=2)
        axb.set_xlim(0, 120)
        axb.set_ylim(-YLIM, YLIM)
        axb.set_xticks([0, 60, 120])
        axb.set_yticks([-5, 0, 5])
        axb.tick_params(axis="x", labelsize=TKF, length=2.5, pad=1.5)
        if i_ == 0:
            axb.tick_params(axis="y", labelsize=TKF, pad=1.5)
            axb.set_ylabel("change in origin\nshare (DiD, %)", fontsize=LBF - 1.0,
                           labelpad=2, linespacing=1.1)
        else:
            axb.set_yticklabels([])
            axb.tick_params(axis="y", labelsize=TKF, length=2)

    fig.text(X_MID, (A_BOT + 0.30) / FIG_H, "pass origin along the pitch (yd)",
             fontsize=LBF, ha="center", va="center")
    fig.text(X_MID, (A_BOT + 0.10) / FIG_H,
             "dashed line: mean origin before the substitution   ·   "
             "solid bar + outline: $|$mean$|\\geq$2 SE   ·   whiskers: $\\pm$1 SE",
             fontsize=NTF - 0.4, ha="center", va="center", color=P["neutral_mid"])
    ytop = (A_BOT + 1.10 + PH_IN + 0.38) / FIG_H    # above pitch top and titles
    fig.text(0.006, ytop, "(a)", fontsize=10, weight="bold", va="center")
    fig.text(0.068, ytop, "attack", fontsize=NTF, style="italic", ha="left",
             va="center", color=P["neutral_mid"])
    fig.add_artist(FancyArrowPatch((0.125, ytop), (0.180, ytop),
                                   transform=fig.transFigure, arrowstyle="-|>",
                                   mutation_scale=5, lw=0.8, color=P["neutral_mid"]))


XLIM_B = 35.0            # x range of (b); points beyond it are counted in the note


def panel_b(fig, sg, cloud):
    """(b) Subgroup means with 95% CI over the per-substitution cloud (grey); values printed at right."""
    ax = fig.add_axes([0.1058, LOW_B, 0.2308, LOW_H])
    SUB = [("all", "all"), ("home", "home"), ("away", "away"),
           ("lead", "leading"), ("draw", "drawing"), ("trail", "trailing")]
    ys = np.arange(len(SUB))[::-1]
    rng = np.random.default_rng(0)                      # fixed jitter seed
    # zero line only over the data rows
    ax.plot([0, 0], [-0.42, len(SUB) - 1 + 0.42], color=P["neutral_mid"], lw=0.7,
            ls=(0, (4, 3)), zorder=1)
    n_all = len(cloud["all"])
    n_out = int((np.abs(cloud["all"]) > XLIM_B).sum())  # points beyond the axis, "all" row
    for y, (key, lab) in zip(ys, SUB):
        r = sg[sg.subgroup == key].iloc[0]
        first = key == "all"
        c = P["signal_dark"] if first else P["neutral_dark"]
        v = cloud[key]
        ax.plot(np.clip(v, -XLIM_B * 1.02, XLIM_B * 1.02),
                y + rng.uniform(-0.24, 0.24, len(v)), ".", ms=1.1,
                color=P["neutral_light"], alpha=0.55, zorder=2, mec="none")
        ax.plot([r.lo, r.hi], [y, y], color=c, lw=2.2 if first else 1.6,
                zorder=4, solid_capstyle="butt")
        for e_ in (r.lo, r.hi):                          # end caps
            ax.plot([e_, e_], [y - 0.16, y + 0.16], color=c, lw=1.0, zorder=5)
        ax.plot([r["mean"]], [y], "o", ms=2.8, color="white", mec=c, mew=1.0, zorder=6)
        star = "*" if r.p < 0.05 else ""
        ax.text(XLIM_B * 1.22, y, f"{r['mean']:+.2f}{star}  [{r.lo:+.2f}, {r.hi:+.2f}]",
                fontsize=5.8, va="center", ha="left", color=c,
                weight="bold" if first else "normal")
    ax.set_yticks(ys)
    ax.set_yticklabels([l for _, l in SUB], fontsize=TKF)
    ax.tick_params(axis="y", length=0, pad=1.5)
    ax.tick_params(axis="x", labelsize=TKF)
    ax.set_xlabel("territorial shift per substitution (yd)", fontsize=LBF, labelpad=2)
    ax.set_xlim(-XLIM_B, XLIM_B)
    ax.set_xticks([-30, -15, 0, 15, 30])
    ax.set_ylim(ys[-1] - 0.95, ys[0] + 0.78)
    ax.spines["left"].set_visible(False)
    # notes kept within the panel width
    ax.text(-XLIM_B * 0.98, ys[-1] - 0.66,
            f"gray: {n_all} substitutions ({n_out} beyond axis)",
            fontsize=NTF - 1.0, color=P["neutral_mid"], ha="left", va="center")
    ax.text(XLIM_B * 1.22, ys[0] + 0.55, "group mean [95% CI]",
            fontsize=NTF - 1.0, color=P["neutral_mid"], ha="left", va="center")
    ax.text(XLIM_B * 1.22, ys[-1] - 0.66, "* $p$ < 0.05",
            fontsize=NTF - 1.0, color=P["neutral_mid"], ha="left", va="center")


def panel_c(fig, cm):
    """(c) Standardized effects of 11 outcomes (SE back-solved from beta and p) against the band."""
    ax = fig.add_axes([0.6950, LOW_B, 0.2800, LOW_H])
    c = cm.copy()
    c["z"] = norm.isf(c.p / 2) * np.sign(c.beta)
    c["r"] = c.z / np.sqrt(c.n)
    c["r_se"] = c.r / c.z
    c = c.sort_values("r", ascending=False).reset_index(drop=True)
    NAME = {"d_terr": "territory", "d_compl": "pass completion", "d_prog": "forward-pass share",
            "net_after": "net goals (after)", "opp_d_terr": "opponent territory",
            "win": "match outcome", "opp_d_compl": "opponent completion", "d_vol": "pass volume",
            "net_rem": "net goals (rest)", "d_poss": "possession share",
            "next_for": "next goal scored"}
    ys = np.arange(len(c))[::-1]
    ax.axvspan(-0.05, 0.05, color="#EEF1F4", zorder=0)
    ax.axvline(0, color=P["neutral_mid"], lw=0.7, ls=(0, (4, 3)), zorder=1)
    for y, (_, r) in zip(ys, c.iterrows()):
        surv = r.q < 0.05
        col = P["accent"] if surv else P["neutral_mid"]
        lo, hi = r.r - 1.96 * r.r_se, r.r + 1.96 * r.r_se
        ax.add_patch(Rectangle((lo, y - 0.30), hi - lo, 0.60, facecolor=col,
                               alpha=0.42 if surv else 0.30, edgecolor="none", zorder=3))
        ax.plot([r.r, r.r], [y - 0.30, y + 0.30], color=col, lw=1.2, zorder=4)
    ax.set_yticks(ys)
    ax.set_yticklabels([NAME[o] for o in c.outcome], fontsize=6.9)
    ax.tick_params(axis="y", length=0, pad=1.5)
    ax.tick_params(axis="x", labelsize=TKF)
    ax.set_xlabel("standardized effect (partial $r$)", fontsize=LBF, labelpad=2)
    ax.set_xlim(-0.105, 0.155)
    ax.set_xticks([-0.05, 0, 0.05, 0.10])
    ax.set_ylim(ys[-1] - 0.62, ys[0] + 0.70)
    ax.spines["left"].set_visible(False)
    ax.text(1.0, 1.015, "shaded: |$r$| < 0.05", fontsize=NTF, transform=ax.transAxes,
            color=P["neutral_mid"], ha="right", va="bottom")


def main():
    tp = dict(np.load(TAB / "territory_profile.npz"))
    sg = pd.read_csv(TAB / "territory_subgroups.csv")
    cm = pd.read_csv(TAB / "consequence_mine.csv")
    da = pd.read_csv(TAB / "dterr_audit.csv")

    # consistency checks
    for g, _ in GROUPS:
        mom, ref = float(tp[f"moment_{g}"][0]), da[da.timing == g].did_terr_excl.mean()
        assert abs(mom - ref) < 0.15, f"{g}: profile moment {mom:+.3f} vs d_terr {ref:+.3f}"
    assert abs(sg.loc[sg.subgroup == "all", "mean"].iloc[0] - 1.074) < 0.01, "overall mean should be +1.074"
    sig_out = cm[cm.q < 0.05]
    assert len(sig_out) == 1 and sig_out.iloc[0].outcome == "d_terr", "territory should be the only outcome with q < 0.05"
    assert (sg[sg.subgroup != "all"].p > 0.05).all(), "subgroups should not be individually significant"

    # (b) cloud: per-batch d_terr from dterr_audit.csv, grouped as in fig3_territory_profile.py
    at = pd.read_parquet(ROOT / "data" / "analysis_table.parquet")[
        ["batch_id", "is_home", "score_state"]]
    m = da.merge(at, on="batch_id", how="left").dropna(subset=["did_terr_excl"])
    cloud = {"all": m.did_terr_excl.values,
             "home": m[m.is_home].did_terr_excl.values,
             "away": m[~m.is_home].did_terr_excl.values,
             "lead": m[m.score_state == "lead"].did_terr_excl.values,
             "draw": m[m.score_state == "draw"].did_terr_excl.values,
             "trail": m[m.score_state == "trail"].did_terr_excl.values}
    for k, v in cloud.items():                       # cloud must match the subgroup table
        r = sg[sg.subgroup == k].iloc[0]
        assert len(v) == int(r.n) and abs(v.mean() - r["mean"]) < 1e-6, \
            f"(b) cloud {k}: n={len(v)}, mean={v.mean():.4f} vs table {int(r.n)}, {r['mean']:.4f}"
    print("  checks passed (profile moment, overall mean, territory only, subgroups, cloud)")

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    panel_a(fig, tp, da)
    panel_b(fig, sg, cloud)
    panel_c(fig, cm)
    ylet = (LOW_B * FIG_H + LOW_H * FIG_H + 0.09) / FIG_H
    fig.text(0.006, ylet, "(b)", fontsize=10, weight="bold", va="center")
    fig.text(0.560, ylet, "(c)", fontsize=10, weight="bold", va="center")

    for ext, kw in [("pdf", {}), ("png", dict(dpi=300))]:
        fig.savefig(FIG / f"fig3.{ext}", **kw)
    plt.close(fig)
    print("saved fig3.{pdf,png}")


if __name__ == "__main__":
    main()
