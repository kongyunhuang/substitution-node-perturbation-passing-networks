"""
Draw Figure 2, substitution timing and the reorganization of the two-layer passing network.

All quantities are teammates-only group-mean DiD responses. (a) Late-group two-layer network
with role-to-zone coupling links. (b) Zone response maps per timing group. (c) Role x timing
matrix. (d) Sliding-window SNR versus substitution minute. (e) Specification curve.
Inputs: results/tables/{role_did,reorg_maps,coupling_didzone,role_positions_empirical}.npz and
        {control_robustness,cross_gradient_clean,timing_curve,entropy_balance_outcomes,mvp_multiscale,cross_gradient_no_overtime,gradient_selfexcl}.csv
Outputs: results/figures/fig2.{pdf,png}
Run: python code/figures/fig2.py
"""

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle
from matplotlib.transforms import Affine2D
from mplsoccer import Pitch

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
TAB = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"
CACHE = TAB / "role_did.npz"
NX, NY = 6, 4
PL, PW = 120, 80

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "cm", "axes.unicode_minus": False,
    "font.size": 8, "axes.labelsize": 9, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "axes.edgecolor": "#333333",
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.major.size": 3, "ytick.major.size": 3, "legend.frameon": False,
    "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})
P = {
    "neutral_dark": "#4D4D4D", "neutral_mid": "#767676", "neutral_light": "#CFCECE",
    "neutral_black": "#272727", "signal_dark": "#0B3954", "signal_mid": "#087E8B",
    "accent": "#FF5A5F",
}
CMAP_D = plt.cm.RdBu_r
C_GAIN = "#B2182B"
C_LOSS = "#2166AC"
PITCH_LINE = "#C9CED4"  # pitch lines, lightest
# colour roles: response = RdBu fill, intra-layer edges = teal, inter-layer links = grey
C_EDGE = "#087E8B"       # intra-layer edges, both layers
C_XLAYER = "#8A8F98"     # inter-layer coupling links
# significance (|mean| >= 2SE) marking, env SIG_STYLE:
#   "dot"    small dots everywhere
#   "subtle" no dots on (a), small dots on (b)(c)
#   "fade"   no dots, non-significant items faded
import os as _os
SIG_STYLE = _os.environ.get("SIG_STYLE", "subtle")
SIG_A = 0.34             # alpha of non-significant items in fade mode


def sig_dot_color(rgba):
    """White dot on dark fills, near-black on light fills."""
    lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
    return "white" if lum < 0.5 else "#272727"

ROLES = ["GK", "RB", "CB", "LB", "DM", "CM", "AM", "RW", "LW", "ST"]
NR = len(ROLES)
# role layout: empirical mean positions, de-overlapped (fig2_role_positions.py)
ROLE_TEMPLATE = np.load(TAB / "role_positions_empirical.npz")["xy_deoverlap"].copy()
# manual nudge: DM up and left, away from CB
ROLE_TEMPLATE[ROLES.index("DM")] += np.array([-16.0, -28.0])

SHX, SHY, GAP = 0.42, 0.40, 76.0   # y shear, y compression, layer gap (max ~77 before (a) becomes height-limited)
GROUPS = [("strategic", "early (HT–60$'$)"), ("regular", "mid (61–75$'$)"),
          ("late", "late (>75$'$)")]


def proj(x, y, lay, x0=0.0):
    x, yd = np.asarray(x, float), PW - np.asarray(y, float)
    return x + SHX * yd + x0, SHY * yd + lay * GAP


def draw_backing(ax, lay, color, z0, x0, alpha=1.0):
    xs, ys = proj([-3.5, 123.5, 123.5, -3.5], [-2.5, -2.5, 82.5, 82.5], lay, x0)
    ax.add_patch(Polygon(np.c_[xs, ys], closed=True, facecolor=color,
                         edgecolor="none", zorder=z0, alpha=alpha))


def draw_pitch_proj(ax, lay, line_c, lw, z0, x0):
    before = set(ax.get_children())
    Pitch(pitch_type="statsbomb", pitch_color="none", line_color=line_c,
          linewidth=lw, goal_type="box", corner_arcs=True).draw(ax=ax)
    tr = Affine2D(np.array([[1.0, -SHX, SHX * PW + x0],
                            [0.0, -SHY, SHY * PW + lay * GAP],
                            [0.0, 0.0, 1.0]]))
    for art in ax.get_children():
        if art not in before:
            art.set_transform(tr + ax.transData)
            art.set_zorder(z0 + 1)
            art.set_clip_on(False)


def draw_border(ax, lay, z0, x0):
    bx, by = proj([0, PL, PL, 0, 0], [0, 0, PW, PW, 0], lay, x0)
    ax.plot(bx, by, color=P["neutral_dark"], lw=0.9, zorder=z0)


def zone_center(k):
    return (k // NY + 0.5) * 20.0, (k % NY + 0.5) * 20.0


def draw_intra(ax, pts, sizes, M, glob, zbase, kmax=8, thr=0.32):
    """Intra-layer directed DiD edges, i->j arrows: solid = strengthened, dashed = weakened.
    shrinkA/B from node radii (sizes = scatter areas in pt^2) so arrowheads stop at the node edge."""
    idx = np.dstack(np.unravel_index(np.argsort(-np.abs(M), axis=None), M.shape))[0]
    shown, used = 0, set()
    for i, j in idx:
        v = M[i, j]
        if i == j or shown >= kmax or abs(v) < thr * glob:
            continue
        ra = np.sqrt(sizes[i] / np.pi) + 1.0   # source radius (pt) + gap
        rb = np.sqrt(sizes[j] / np.pi) + 2.0   # target radius (pt) + gap
        lw = 0.5 + 1.6 * abs(v) / glob
        ax.add_patch(FancyArrowPatch(
            pts[i], pts[j], arrowstyle="-|>",
            mutation_scale=5.0 + 3.0 * abs(v) / glob, lw=lw, color=C_EDGE,
            linestyle="-" if v > 0 else (0, (2.4, 1.6)),
            alpha=0.92 if v > 0 else 0.85, zorder=zbase,
            shrinkA=ra, shrinkB=rb, connectionstyle="arc3,rad=0.14"))
        used.add(int(i)); used.add(int(j))
        shown += 1
    return used


def draw_hero(ax, D, npz, cc, g, rnorm, znorm, glob_ze, glob_re, glob_cc):
    """Late-group two-layer response network: RdBu fills, teal intra-layer arrows, grey
    inter-layer coupling links with a white halo; node size = baseline participation."""
    znode, zse = npz[f"mean_{g}"], npz[f"se_{g}"]
    C_LINK = C_XLAYER  # inter-layer links
    zpts = {k: proj(*zone_center(k), 0, 0) for k in range(24)}
    rpts = {i: proj(*ROLE_TEMPLATE[i], 1, 0) for i in range(NR)}
    rbase = D[f"rbase_{g}"]
    # node area ~ baseline participation, radius minus 3 pt, floor 0.8 pt
    rsizes = {i: np.pi * max(np.sqrt((40 + 220 * rbase[i] / rbase.max()) / np.pi) - 3.0, 0.8) ** 2
              for i in range(NR)}
    Z_S = 14.0  # zone node size
    zsizes = {k: Z_S for k in range(24)}
    # lower layer: zone heatmap, zone-to-zone arrows, zone nodes
    draw_backing(ax, 0, "white", 1, 0)
    for k in range(24):
        ix, iy = divmod(k, NY)
        cx0, cy0 = ix * 20.0, iy * 20.0
        xs, ys = proj([cx0, cx0 + 20, cx0 + 20, cx0], [cy0, cy0, cy0 + 20, cy0 + 20], 0, 0)
        sig = np.abs(znode[k]) >= 2 * zse[k]
        ax.add_patch(Polygon(np.c_[xs, ys], closed=True, facecolor=CMAP_D(znorm(znode[k])),
                             edgecolor="white", linewidth=0.6, zorder=2,
                             alpha=1.0 if (sig or SIG_STYLE != "fade") else SIG_A))
        if sig and SIG_STYLE == "dot":
            px, py = proj(cx0 + 16.5, cy0 + 16.5, 0, 0)
            ax.plot(px, py, ".", color=P["neutral_black"], ms=2.6, zorder=6.8)
    draw_pitch_proj(ax, 0, PITCH_LINE, 0.6, 3, 0)
    draw_border(ax, 0, 6, 0)
    zconn = draw_intra(ax, zpts, zsizes, D[f"zedge_{g}"], glob_ze, 5.5, kmax=16, thr=0.20)
    # inter-layer coupling dC (role -> zone), no arrowheads: solid = engages, dashed = disengages
    # zorder 7: above the lower pitch, under the translucent upper backing
    idxc = np.dstack(np.unravel_index(np.argsort(-np.abs(cc), axis=None), cc.shape))[0]
    links, tgt_zones = [], set()
    for ri, zi in idxc:
        if len(links) >= 8 or abs(cc[ri, zi]) < 0.30 * glob_cc:
            continue
        links.append((int(ri), int(zi), float(cc[ri, zi])))
        tgt_zones.add(int(zi))
    # zone nodes only where an edge or link lands
    for k in sorted(zconn | tgt_zones):
        zx, zy = zpts[k]
        ax.scatter([zx], [zy], s=Z_S, c=[P["neutral_mid"]], edgecolor="white",
                   linewidth=0.4, zorder=6.5)
    for ri, zi, v in links:
        xa, ya = proj(*ROLE_TEMPLATE[ri], 1, 0)
        xb, yb = proj(*zone_center(zi), 0, 0)
        rad = 0.12 if v > 0 else -0.12
        # thinner and fainter than intra-layer edges
        lw = 0.35 + 1.0 * abs(v) / glob_cc
        cs = f"arc3,rad={rad}"
        ax.add_patch(FancyArrowPatch((xa, ya), (xb, yb), arrowstyle="-",
                     lw=lw + 0.9, color="white", alpha=0.55, zorder=6.9,
                     shrinkA=3.5, shrinkB=2.5, connectionstyle=cs))
        ax.add_patch(FancyArrowPatch((xa, ya), (xb, yb), arrowstyle="-",
                     lw=lw, color=C_LINK,
                     linestyle="-" if v > 0 else (0, (2.5, 1.8)), alpha=0.45, zorder=7,
                     shrinkA=3.5, shrinkB=2.5, connectionstyle=cs))
    # enlarge zone nodes that receive a link
    for zi in tgt_zones:
        zx, zy = zpts[zi]
        ax.add_patch(Circle((zx, zy), 2.0, facecolor=P["neutral_mid"],
                            edgecolor="white", linewidth=0.6, zorder=7.2))
    # upper layer: translucent backing, role-to-role arrows
    draw_backing(ax, 1, "white", 8, 0, alpha=0.62)
    draw_pitch_proj(ax, 1, PITCH_LINE, 0.55, 8, 0)
    draw_border(ax, 1, 10, 0)
    draw_intra(ax, rpts, rsizes, D[f"redge_{g}"], glob_re, 10.55, kmax=14, thr=0.20)
    # role nodes: fill = response, size = baseline
    rout = D[f"rout_{g}"]
    role_top = {int(np.argmax(rout))}     # most withdrawn role, same rule as (c)
    rn, rse = D[f"rnode_{g}"], D[f"rnode_se_{g}"]
    TOP_LABEL = {ROLES.index(r) for r in ("LB", "DM", "LW", "CM")}  # labels above the node
    CONV = 0.55  # pt -> data units for this axis
    for i, role in enumerate(ROLES):
        x, y = proj(*ROLE_TEMPLATE[i], 1, 0)
        r_data = CONV * np.sqrt(rsizes[i] / np.pi)      # node radius (data units)
        # dashed ring outside the node for the most withdrawn role
        if i in role_top:
            rr = r_data + 2.2
            ax.add_patch(Circle((x, y), rr, facecolor="none", edgecolor=P["accent"],
                                linewidth=1.2, linestyle=(0, (2.2, 1.5)), zorder=11.5))
        node_c = CMAP_D(rnorm(rn[i]))
        sig_n = np.abs(rn[i]) >= 2 * rse[i]
        ax.scatter([x], [y], s=rsizes[i], c=[node_c],
                   edgecolor=P["neutral_dark"], linewidth=0.6, zorder=12,
                   alpha=1.0 if (sig_n or SIG_STYLE != "fade") else SIG_A)
        if sig_n and SIG_STYLE == "dot":
            ax.plot(x, y, ".", color=sig_dot_color(node_c), ms=2.6, zorder=13)
        gap = (rr if i in role_top else r_data) + 1.2   # ring edge if ringed, else node edge
        if i in TOP_LABEL:
            ax.text(x, y + gap, role, fontsize=5.8, ha="center", va="bottom",
                    color=P["neutral_dark"], zorder=14,
                    bbox=dict(boxstyle="round,pad=0.10", facecolor="white", alpha=0.92,
                              edgecolor="none"))
        else:
            ax.text(x, y - gap, role, fontsize=5.8, ha="center", va="top",
                    color=P["neutral_dark"], zorder=14,
                    bbox=dict(boxstyle="round,pad=0.10", facecolor="white", alpha=0.92,
                              edgecolor="none"))


def draw_flat(ax, D, npz, g, znorm):
    """(b) Flat zone DiD map: cell fill = response, dot where |mean| >= 2SE. No flow arrows."""
    znode, zse = npz[f"mean_{g}"], npz[f"se_{g}"]
    for k in range(24):
        ix, iy = divmod(k, NY)
        sig = np.abs(znode[k]) >= 2 * zse[k]
        ax.add_patch(Rectangle((ix * 20.0, iy * 20.0), 20.0, 20.0,
                               facecolor=CMAP_D(znorm(znode[k])), edgecolor="white",
                               linewidth=0.4, zorder=2,
                               alpha=1.0 if (sig or SIG_STYLE != "fade") else SIG_A))
        if sig and SIG_STYLE != "fade":
            ax.scatter(ix * 20.0 + 16.0, iy * 20.0 + 16.0, s=5.0, facecolor="white",
                       edgecolor=P["neutral_black"], linewidth=0.35, zorder=6)
    before = set(ax.get_children())
    Pitch(pitch_type="statsbomb", pitch_color="none", line_color="#FFFFFF",
          linewidth=0.4, goal_type="box", corner_arcs=True).draw(ax=ax)
    for art in ax.get_children():
        if art not in before:
            art.set_zorder(3)


def main():
    npz = np.load(TAB / "reorg_maps.npz")
    assert CACHE.exists(), "role_did.npz missing; run fig2_role_responses.py"
    D = dict(np.load(CACHE))
    CPL_PATH = TAB / "coupling_didzone.npz"
    assert CPL_PATH.exists(), "coupling_didzone.npz missing; run fig2_role_zone_coupling.py"
    CPL = dict(np.load(CPL_PATH))  # role -> zone coupling dC
    # consistency checks
    for g, _ in GROUPS:
        n_h, n_f = int(D[f"n_{g}"][0]), int(npz[f"n_{g}"][0])
        n_c = int(CPL[f"n_{g}"][0])
        dmax = np.abs(D[f"znode_{g}"] - npz[f"mean_{g}"]).max()
        # coupling row sums must match the role response (normalised)
        rm = CPL[f"cc_{g}"].sum(1)
        rm_dev = np.abs(rm / np.abs(rm).sum() - D[f"rnode_{g}"] / np.abs(D[f"rnode_{g}"]).sum()).max()
        print(f"  {g}: n={n_h} (frozen {n_f}, coupling {n_c}) zone dev={dmax:.2e} coupling margin dev={rm_dev:.1e}")
        assert n_h == n_f == n_c and dmax < 1e-9 and rm_dev < 1e-9, f"{g}: inconsistent sources"
    assert [int(D[f"n_{g}"][0]) for g, _ in GROUPS] == [425, 404, 73], "expected n = 425/404/73"
    nsig = [int((np.abs(D[f"rnode_{g}"]) >= 2 * D[f"rnode_se_{g}"]).sum()) for g, _ in GROUPS]
    print(f"  significant role nodes = {nsig}")
    assert nsig == [2, 5, 8], "expected significant role counts 2/5/8"
    # headline SNR from control_robustness.csv (subset=all), same source as the text
    cro0 = pd.read_csv(TAB / "control_robustness.csv")
    cro0 = cro0[cro0.subset == "all"].set_index("timing")
    cg = pd.read_csv(TAB / "cross_gradient_clean.csv")
    snr = {g: cro0.loc[g, "snr_excl"] for g, _ in GROUPS}
    w = cg.pivot(index="comp", columns="timing", values="snr_excl")
    n_le = int((w["late"] > w["strategic"]).sum())
    n_me = int((w["regular"] > w["strategic"]).sum())
    n_lm = int((w["late"] > w["regular"]).sum())
    ncomp = len(w)
    print(f"  sign tests: late>early {n_le}/{ncomp}, mid>early {n_me}/{ncomp}, late>mid {n_lm}/{ncomp}")
    assert (n_le, n_me, n_lm, ncomp) == (10, 10, 5, 10), "expected sign tests 10/10, 10/10, 5/10"
    # shared scales
    zvmax = max(np.abs(npz[f"mean_{g}"]).max() for g, _ in GROUPS)
    rvmax = max(np.abs(D[f"rnode_{g}"]).max() for g, _ in GROUPS)
    znorm = TwoSlopeNorm(vcenter=0, vmin=-zvmax, vmax=zvmax)
    rnorm = TwoSlopeNorm(vcenter=0, vmin=-rvmax, vmax=rvmax)
    glob_ze = max(np.abs(D[f"zedge_{g}"]).max() for g, _ in GROUPS)
    glob_re = max(np.abs(D[f"redge_{g}"]).max() for g, _ in GROUPS)
    glob_cc = np.abs(CPL["cc_late"]).max()

    FW, FH = 5.0, 5.0  # square canvas
    fig = plt.figure(figsize=(FW, FH))

    # layout grid: top row a|b|c, bottom row d|e
    TOP, BOT_U = 0.945, 0.455      # top row extent
    CBAR_Y = 0.365                 # both colourbars
    BOT_L, H_L = 0.045, 0.250      # bottom row extent
    LY_TOP, LY_LOW = 0.958, 0.310  # panel letter baselines

    # (a) late-group two-layer response
    A_DX, A_DY = 0.015, -0.07       # (a) offset
    XSPAN, GAP_REF = 173.0, 50.0    # xlim span; reference layer gap
    A_DY_GAP = 0.0    # no gap compensation
    axa = fig.add_axes([0.005 + A_DX, TOP - 0.535 + A_DY + A_DY_GAP, 0.60, 0.535])
    axa.set_anchor("N")            # top-align with (b)(c)
    axa.set_axis_off()
    draw_hero(axa, D, npz, CPL["cc_late"], "late", rnorm, znorm, glob_ze, glob_re, glob_cc)
    axa.set_xlim(-5, 168)
    axa.set_ylim(-25, SHY * PW + GAP + 20)
    axa.set_aspect("equal")
    tx, _ = proj(60, 40, 1, 0)
    axa.text(tx, SHY * PW + GAP + 15.5, "late (>75$'$) substitutions",
             fontsize=9, weight="bold", ha="center", color=P["neutral_black"])
    axa.text(tx, SHY * PW + GAP + 9.8,
             f"mean over $n$ = {int(D['n_late'][0])},  SNR = {snr['late']:.2f}",
             fontsize=6.8, ha="center", color=P["neutral_mid"], style="italic")
    # attack arrow, upper layer
    ax_, ay_ = proj(10, 16, 1, 0)
    bx_, by_ = proj(34, 16, 1, 0)
    axa.add_patch(FancyArrowPatch((ax_, ay_), (bx_, by_), arrowstyle="-|>",
                                  mutation_scale=6, lw=0.9, color=P["neutral_mid"],
                                  zorder=20))
    axa.text((ax_ + bx_) / 2, ay_ + 1.8, "attack", fontsize=5.8, ha="center", va="bottom",
             color=P["neutral_mid"], style="italic", zorder=20)
    # attack arrow, below the lower pitch
    axa.add_patch(FancyArrowPatch((10, -5.5), (34, -5.5), arrowstyle="-|>", mutation_scale=6,
                                  lw=0.9, color=P["neutral_mid"], zorder=20))
    axa.text(36, -5.5, "attack", fontsize=5.8, ha="left", va="center",
             color=P["neutral_mid"], style="italic", zorder=20)
    # layer labels
    axa.text(1, SHY * PW + GAP + 3.0, "player layer",
             fontsize=6.5, ha="left", color=P["neutral_dark"], style="italic", zorder=20,
             bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.8,
                       edgecolor="none"))
    axa.text(1, SHY * PW + 9.5, "pitch-passing layer",
             fontsize=6.5, ha="left", color=P["neutral_dark"], style="italic", zorder=20,
             bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.8,
                       edgecolor="none"))
    # legend, two rows
    LF = 5.6
    ly1 = -13.5
    axa.scatter([5], [ly1], s=11, c=[CMAP_D(0.5)], edgecolor=P["neutral_dark"], lw=0.5, zorder=20)
    axa.scatter([9.5], [ly1], s=44, c=[CMAP_D(0.85)], edgecolor=P["neutral_dark"], lw=0.5, zorder=20)
    axa.text(13.5, ly1, "size $\\propto$ baseline", fontsize=LF, va="center",
             color=P["neutral_black"], zorder=20)
    axa.add_patch(FancyArrowPatch((52, ly1), (64, ly1), arrowstyle="-|>", mutation_scale=5.5,
                                  lw=1.3, color=C_EDGE, zorder=20))
    axa.text(66, ly1, "passing edge: strengthened", fontsize=LF, va="center",
             color=P["neutral_black"], zorder=20)
    axa.add_patch(FancyArrowPatch((123, ly1), (135, ly1), arrowstyle="-|>", mutation_scale=5.5,
                                  lw=1.2, color=C_EDGE, linestyle=(0, (1.6, 1.5)), zorder=20))
    axa.text(137, ly1, "weakened", fontsize=LF, va="center", color=P["neutral_black"], zorder=20)
    ly2 = -19.5
    axa.plot([3, 15], [ly2, ly2], color=C_XLAYER, lw=1.4, zorder=20)
    axa.text(17, ly2, "role$\\to$zone: engages", fontsize=LF, va="center",
             color=P["neutral_black"], zorder=20)
    axa.plot([64, 76], [ly2, ly2], color=C_XLAYER, lw=1.3, ls=(0, (1.8, 1.6)), zorder=20)
    axa.text(78, ly2, "disengages", fontsize=LF, va="center", color=P["neutral_black"], zorder=20)
    axa.add_patch(Circle((110, ly2), 2.2, facecolor="none", edgecolor=P["accent"],
                         linewidth=1.0, linestyle=(0, (2.2, 1.5)), zorder=20))
    axa.text(114, ly2, "most withdrawn role", fontsize=LF, va="center",
             color=P["neutral_black"], zorder=20)
    fig.text(0.012 + A_DX, LY_TOP + A_DY + A_DY_GAP, "(a)", fontsize=10, weight="bold",
             va="center")

    # (b) zone DiD maps, three groups
    BX, BW = 0.610, 0.215
    BH = BW * (FW / FH) / 1.5  # 120x80 aspect
    BGAP = (TOP - BOT_U - 3 * BH) / 2          # equal spacing from TOP to BOT_U
    bys = [TOP - BH - i * (BH + BGAP) for i in range(3)]
    for (g, glab), by in zip(GROUPS, bys):
        axb = fig.add_axes([BX, by, BW, BH])
        axb.set_axis_off()
        draw_flat(axb, D, npz, g, znorm)
        axb.set_xlim(-2, 122)
        axb.set_ylim(82, -2)
        axb.set_aspect("equal")
        # group label + n
        axb.text(60, -4, f"{glab}   $n$={int(D[f'n_{g}'][0])}", fontsize=6.0,
                 ha="center", va="bottom", color=P["neutral_black"],
                 transform=axb.transData)
        if g == "strategic":
            fig.text(0.583, LY_TOP, "(b)", fontsize=10, weight="bold", va="center")
    # shared zone colourbar
    caxz = fig.add_axes([BX + 0.015, CBAR_Y, BW - 0.03, 0.012])
    cbz = fig.colorbar(plt.cm.ScalarMappable(cmap=CMAP_D, norm=znorm), cax=caxz,
                       orientation="horizontal")
    cbz.set_ticks([-zvmax, 0, zvmax])
    cbz.set_ticklabels([f"$-${zvmax:.2f}", "0", f"+{zvmax:.2f}"])
    cbz.ax.tick_params(labelsize=5.6, width=0.5, length=2)
    cbz.outline.set_linewidth(0.5)
    caxz.set_title("zone response $\\langle d\\rangle$", fontsize=6.2,
                   color=P["neutral_black"], pad=2.5)

    # (c) role response matrix 10x3
    axc = fig.add_axes([0.888, BOT_U, 0.082, TOP - BOT_U])
    M = np.column_stack([D[f"rnode_{g}"] for g, _ in GROUPS])
    for i in range(len(ROLES)):
        for j, (g, _) in enumerate(GROUPS):
            sig_c = np.abs(M[i, j]) >= 2 * D[f"rnode_se_{g}"][i]
            axc.add_patch(Rectangle((j, i), 1, 1, facecolor=CMAP_D(rnorm(M[i, j])),
                                    edgecolor="white", linewidth=0.5,
                                    alpha=1.0 if (sig_c or SIG_STYLE != "fade") else SIG_A))
            if sig_c and SIG_STYLE != "fade":
                axc.scatter(j + 0.5, i + 0.5, s=7.0, facecolor="white",
                            edgecolor=P["neutral_black"], linewidth=0.4, zorder=5)
            if i == int(np.argmax(D[f"rout_{g}"])):
                axc.add_patch(Rectangle((j + 0.07, i + 0.07), 0.86, 0.86,
                                        facecolor="none", edgecolor=P["accent"],
                                        linewidth=1.0, linestyle=(0, (2.2, 1.5)), zorder=6))
    axc.set_xlim(0, 3)
    axc.set_ylim(len(ROLES), 0)
    axc.set_xticks([0.5, 1.5, 2.5])
    axc.set_xticklabels(["early", "mid", "late"], fontsize=5.8, rotation=45,
                        ha="right")
    axc.set_yticks(np.arange(len(ROLES)) + 0.5)
    axc.set_yticklabels(ROLES, fontsize=5.6)
    axc.tick_params(length=0, pad=1.5)
    for s in axc.spines.values():
        s.set_visible(False)
    # significant role count per group
    for j, k in enumerate(nsig):
        axc.text(j + 0.5, -0.28, str(k), fontsize=6.8, ha="center",
                 color=P["neutral_black"], weight="bold")
    axc.text(1.5, -0.62, "roles with\n|$\\langle\\Delta\\rangle$| $\\geq$ 2SE",
             fontsize=5.6, ha="center", va="bottom", color=P["neutral_mid"])
    fig.text(0.848, LY_TOP, "(c)", fontsize=10, weight="bold", va="center")
    # role colourbar
    caxr = fig.add_axes([0.888, CBAR_Y, 0.082, 0.012])
    cbr = fig.colorbar(plt.cm.ScalarMappable(cmap=CMAP_D, norm=rnorm), cax=caxr,
                       orientation="horizontal")
    cbr.set_ticks([-rvmax, 0, rvmax])
    cbr.set_ticklabels([f"$-${rvmax:.2f}", "0", f"+{rvmax:.2f}"])
    cbr.ax.tick_params(labelsize=5.6, width=0.5, length=2)
    cbr.outline.set_linewidth(0.5)
    caxr.set_title("role response $\\langle\\Delta\\rangle$", fontsize=6.2,
                   color=P["neutral_black"], pad=2.5)

    # sequential timing colours, light -> dark = early -> late
    TSEQ = {"strategic": "#A9B8C6", "regular": "#4F7A96", "late": P["signal_dark"]}
    order = ["strategic", "regular", "late"]

    # (d) sliding-window SNR versus substitution minute
    tc_ = pd.read_csv(TAB / "timing_curve.csv")
    axd = fig.add_axes([0.075, BOT_L, 0.365, H_L])
    MAIN = "La Liga 2023/24"
    OD_LS = {"LaLiga15": "-", "PL15": (0, (3.2, 1.6)), "SerieA15": (0, (1.2, 1.4)),
             "WSL": (0, (4.5, 1.4, 1.0, 1.4)), "ISL": (0, (2.0, 1.2, 0.6, 1.2))}
    for src, gg in tc_[tc_.source != MAIN].groupby("source", sort=False):
        axd.plot(gg.center_min, gg.snr, color=P["neutral_mid"], lw=0.75, alpha=0.85,
                 ls=OD_LS.get(src, "-"), zorder=2, solid_capstyle="round")
    axd.plot([81.5, 85.0], [2.44, 2.44], color=P["signal_dark"], lw=2.0,
             solid_capstyle="round")                       # legend: main dataset first
    axd.text(85.8, 2.44, "La Liga 23/24", fontsize=5.4, va="center", color=P["signal_dark"])
    for k, (src, ls) in enumerate(OD_LS.items()):          # compact legend, upper right
        yy = 2.29 - k * 0.135
        axd.plot([81.5, 85.0], [yy, yy], color=P["neutral_mid"], lw=0.75, ls=ls,
                 solid_capstyle="round")
        axd.text(85.8, yy, src, fontsize=5.2, va="center", color=P["neutral_mid"])
    ml = tc_[tc_.source == MAIN].sort_values("center_min")
    axd.fill_between(ml.center_min, ml.lo, ml.hi, color=P["signal_dark"], alpha=0.075,
                     lw=0, zorder=3)
    axd.plot(ml.center_min, ml.snr, color=P["signal_dark"], lw=2.0, zorder=5,
             solid_capstyle="round")
    axd.axhspan(0.0, 1.0, color="#EFEFEC", zorder=0)          # noise band (SNR <= 1)
    axd.text(48.6, 0.86, "noise", fontsize=5.4, color=P["neutral_mid"], style="italic",
             va="center", ha="left")
    axd.axvline(60, color=P["accent"], lw=1.0, ls=(0, (5, 3)), zorder=4)
    axd.text(48.8, 2.54, "threshold\n$\\sim$60$'$", fontsize=6.2, color=P["accent"],
             ha="left", va="top", linespacing=1.15)
    axd.set_xlabel("substitution minute", fontsize=7.5)
    axd.set_ylabel("SNR (teammates-only, DiD)", fontsize=7.5)
    axd.set_xlim(48, 97)
    axd.set_ylim(0.74, 2.58)
    axd.set_xticks([50, 60, 70, 80])
    axd.set_yticks([1.0, 1.5, 2.0])
    axd.tick_params(labelsize=6.8)
    fig.text(0.028, LY_LOW, "(d)", fontsize=10, weight="bold", va="center")

    # (e) specification curve
    eb = pd.read_csv(TAB / "entropy_balance_outcomes.csv")
    eb = eb[eb.outcome == "zone_reorg_snr_excl"].set_index("group")
    cro = pd.read_csv(TAB / "control_robustness.csv")
    crx = cro[cro.subset == "cross_only"].set_index("timing")
    ms = pd.read_csv(TAB / "mvp_multiscale.csv")
    ot = pd.read_csv(TAB / "cross_gradient_no_overtime.csv")

    THREE = ["Bund15", "LaLiga15", "PL15", "SerieA15"]   # 2015/16 seasons: three-substitution era

    def cmean(df, cfg=None):
        d = df if cfg is None else df[df.config == cfg]
        return [d[d.timing == t].snr_excl.mean() for t in order]

    gse = pd.read_csv(TAB / "gradient_selfexcl.csv").set_index("timing")  # only for the subs-included row
    SPECS = [  # (label, three-group SNR, block)
        ("main (W15, 6$\\times$4)", [cro0.loc[t, "snr_excl"] for t in order], 0),
        ("subs included",           [gse.loc[t, "snr_incl"] for t in order], 0),
        ("cross-match controls",    [crx.loc[t, "snr_excl"] for t in order], 0),
        ("covariate-balanced",      [eb.loc[t, "est_weighted"] for t in order], 0),
        ("main (W15, 6$\\times$4)", cmean(cg), 1),
        ("grid 4$\\times$3",        cmean(ms, "W15_4x3"), 1),
        ("grid 10$\\times$7",       cmean(ms, "W15_10x7"), 1),
        ("window 12 min",           cmean(ms, "W12_6x4"), 1),
        ("no extra time",           cmean(ot), 1),
        ("three-sub era (4)",       cmean(cg[cg.comp.isin(THREE)]), 1),
        ("five-sub era (6)",        cmean(cg[~cg.comp.isin(THREE)]), 1),
        ("window 10 min",           cmean(ms, "W10_6x4"), 1),
    ]
    axe = fig.add_axes([0.660, BOT_L, 0.315, H_L])
    ys, yy, prev_b = [], 0.0, 0
    for _lab, _v, _b in SPECS:                        # top to bottom, gap between blocks
        if _b != prev_b:
            yy -= 1.70
            prev_b = _b
        ys.append(yy); yy -= 1.0
    ys = np.array(ys)
    axe.axhspan(ys[3] - 0.5, ys[0] + 0.95, color="#F2F4F6", zorder=0)   # upper block = La Liga
    axe.axvline(1.0, color=P["neutral_mid"], lw=0.8, ls=(0, (4, 3)), zorder=1)
    DYT = {"strategic": 0.17, "regular": 0.0, "late": -0.17}   # vertical offsets so mid/late do not overlap
    for yy, (lab, vals, b) in zip(ys, SPECS):
        axe.plot([min(vals), max(vals)], [yy, yy], color=P["neutral_light"], lw=0.9,
                 zorder=2, solid_capstyle="round")
        for t, v in zip(order, vals):
            if lab == "covariate-balanced" and t == "late":   # late group: balance not achieved, open marker
                axe.plot([v], [yy + DYT[t]], "o", ms=3.4, mfc="white", mec=TSEQ[t],
                         mew=0.8, zorder=4)
                axe.text(v, yy - 0.52, "open: balance not achieved", fontsize=5.0,
                         color=P["neutral_mid"], style="italic", va="center", ha="right")
            else:
                axe.plot([v], [yy + DYT[t]], "o", ms=3.4, color=TSEQ[t], mec="white",
                         mew=0.5, zorder=4)
    # block titles
    axe.text(1.03, ys[0] + 0.62, "La Liga 2023/24", fontsize=5.8, style="italic",
             color=P["neutral_mid"], ha="left")
    axe.text(1.03, ys[4] + 0.68, "mean across 10 open-data competitions", fontsize=5.8,
             style="italic", color=P["neutral_mid"], ha="left")
    axe.annotate("shorter window: predicted collapse",
                 xy=(1.18, ys[-1] - 0.17), xytext=(1.33, ys[-1] - 0.17), fontsize=5.5,
                 color=P["neutral_mid"], va="center", ha="left",
                 arrowprops=dict(arrowstyle="-", lw=0.5, color=P["neutral_mid"]))
    axe.set_yticks(ys)
    axe.set_yticklabels([s[0] for s in SPECS], fontsize=5.8)
    axe.set_xlabel("SNR (teammates-only, DiD)", fontsize=7.5)
    axe.set_xlim(0.90, 2.42)
    axe.set_ylim(ys[-1] - 0.75, ys[0] + 1.75)
    axe.set_xticks([1.0, 1.5, 2.0])
    axe.tick_params(axis="y", length=0, pad=1.5)
    axe.tick_params(axis="x", labelsize=6.6)
    axe.spines["left"].set_visible(False)
    # timing legend
    for i, (t, lab) in enumerate(zip(order, ["early", "mid", "late"])):
        axe.plot([1.72 + i * 0.24], [ys[0] + 1.42], "o", ms=3.6, color=TSEQ[t],
                 mec="white", mew=0.5, clip_on=False)
        axe.text(1.76 + i * 0.24, ys[0] + 1.42, lab, fontsize=5.8, va="center",
                 color=P["neutral_black"])
    fig.text(0.560, LY_LOW, "(e)", fontsize=10, weight="bold", va="center")

    for ext, kw in [("pdf", {}), ("png", dict(dpi=300))]:
        stem = "fig2" if SIG_STYLE == "subtle" else f"_design_drafts/sig_{SIG_STYLE}"
        fig.savefig(FIG / f"{stem}.{ext}", **kw)
    plt.close(fig)
    print("saved fig2.{pdf,png}")


if __name__ == "__main__":
    main()
