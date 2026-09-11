"""
Role-level difference-in-differences responses on both layers (input to Figure 2).

Per timing group: teammates-only DiD of role participation, role-to-role edges, zone
participation and zone-to-zone edges, plus baseline role participation and withdrawn-role
counts. Caches the aggregate, then renders a diagnostic three-column figure.
Inputs: data/{events_pass,did_controls,player_positions}.parquet, data/substitution_batches.csv,
        data/windows/*.parquet, results/tables/{reorg_maps.npz,gradient_selfexcl.csv,cross_gradient_clean.csv}
Outputs: results/tables/role_did.npz (read by fig2.py), results/figures/fig2_role_responses_diagnostic.{svg,pdf,tiff,png}
Run: python code/figures/fig2_role_responses.py
"""

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, Polygon
from matplotlib.transforms import Affine2D
from mplsoccer import Pitch

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
TAB = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"
CACHE = TAB / "role_did.npz"
NX, NY = 6, 4
PL, PW = 120, 80
WIN = 900

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
C_GAIN = "#B2182B"   # edge strengthened
C_LOSS = "#2166AC"   # edge weakened

ROLES = ["GK", "RB", "CB", "LB", "DM", "CM", "AM", "RW", "LW", "ST"]
ROLE_MAP = {
    "Goalkeeper": "GK", "Right Back": "RB", "Right Wing Back": "RB",
    "Right Center Back": "CB", "Left Center Back": "CB", "Center Back": "CB",
    "Left Back": "LB", "Left Wing Back": "LB",
    "Right Defensive Midfield": "DM", "Left Defensive Midfield": "DM",
    "Center Defensive Midfield": "DM",
    "Right Center Midfield": "CM", "Left Center Midfield": "CM",
    "Center Attacking Midfield": "AM", "Right Attacking Midfield": "AM",
    "Left Attacking Midfield": "AM",
    "Right Wing": "RW", "Right Midfield": "RW",
    "Left Wing": "LW", "Left Midfield": "LW",
    "Center Forward": "ST", "Right Center Forward": "ST", "Left Center Forward": "ST",
}
RIDX = {r: i for i, r in enumerate(ROLES)}
NR = len(ROLES)
# template role layout (display only)
ROLE_TEMPLATE = np.array([
    [8, 40], [30, 66], [24, 40], [30, 14], [45, 40],
    [57, 40], [69, 40], [74, 62], [74, 18], [88, 40]], dtype=float)

SHX, SHY, GAP = 0.42, 0.40, 50.0


def proj(x, y, lay, x0=0.0):
    x, yd = np.asarray(x, float), PW - np.asarray(y, float)
    return x + SHX * yd + x0, SHY * yd + lay * GAP


def draw_backing(ax, lay, color, z0, x0):
    xs, ys = proj([-3.5, 123.5, 123.5, -3.5], [-2.5, -2.5, 82.5, 82.5], lay, x0)
    ax.add_patch(Polygon(np.c_[xs, ys], closed=True, facecolor=color,
                         edgecolor="none", zorder=z0))


def draw_pitch(ax, lay, line_c, lw, z0, x0):
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


def zbin(x, y):
    zx = np.clip((x / (PL / NX)).astype(int), 0, NX - 1)
    zy = np.clip((y / (PW / NY)).astype(int), 0, NY - 1)
    return zx * NY + zy


def zvec(w, ex_ids=None):
    if ex_ids is not None:
        w = w[~w.player_id.isin(ex_ids) & ~w.recipient_id.isin(ex_ids)]
    Z = np.zeros((24, 24))
    if len(w):
        np.add.at(Z, (zbin(w.x.values, w.y.values), zbin(w.end_x.values, w.end_y.values)), 1.0)
    np.fill_diagonal(Z, 0)
    t = Z / Z.sum() if Z.sum() > 0 else Z
    return t.sum(0) + t.sum(1), t


def rolevec(w, mid, pos, ex_ids=None):
    """Role participation share (in+out) and role-by-role share matrix; ex_ids for teammates-only."""
    if ex_ids is not None:
        w = w[~w.player_id.isin(ex_ids) & ~w.recipient_id.isin(ex_ids)]
    rp = w.player_id.map(lambda p: pos.get((mid, p)))
    rr = w.recipient_id.map(lambda p: pos.get((mid, p)))
    ok = rp.notna() & rr.notna()
    M = np.zeros((NR, NR))
    if ok.sum() > 0:
        np.add.at(M, ([RIDX[a] for a in rp[ok]], [RIDX[a] for a in rr[ok]]), 1.0)
    t = M / M.sum() if M.sum() > 0 else M
    return t.sum(0) + t.sum(1), t


def aggregate():
    passes = pd.read_parquet(DATA / "events_pass.parquet")
    net = passes[passes.outcome.isna() & ~passes.is_set_piece].copy()
    t1 = passes[passes.period == 1].groupby("match_id").t_period_sec.max()
    net["tc"] = np.where(net.period == 1, net.t_period_sec, net.match_id.map(t1) + net.t_period_sec)
    ni = {k: g for k, g in net.groupby(["match_id", "team_id"])}
    b = pd.read_csv(DATA / "substitution_batches.csv").set_index("batch_id")
    sp = pd.read_parquet(DATA / "windows" / "window_specs.parquet")
    st = sp[(sp.kind == "time") & (sp.W_min == 15)].set_index(["batch_id", "side"])
    fl = pd.read_parquet(DATA / "windows" / "batch_flags.parquet")
    nontr = set(fl[(fl.W_min == 15) & ~fl.post_truncated].batch_id)
    ctrl = pd.read_parquet(DATA / "did_controls.parquet")
    ctrl = ctrl[ctrl.control_type != "none"].set_index("batch_id")
    pp = pd.read_parquet(DATA / "player_positions.parquet")
    pos = {(m, p): ROLE_MAP.get(q) for m, p, q in zip(pp.match_id, pp.player_id, pp.position)}

    def winp(mid, tid, lo, hi, lo_open):
        g = ni.get((mid, tid))
        if g is None:
            return None
        return g[(g.tc > lo) & (g.tc <= hi)] if lo_open else g[(g.tc >= lo) & (g.tc < hi)]

    G = ["strategic", "regular", "late"]
    agg = {g: dict(n=0, znode=np.zeros(24), zedge=np.zeros((24, 24)),
                   rnode=np.zeros(NR), rnode_sq=np.zeros(NR), redge=np.zeros((NR, NR)),
                   rbase=np.zeros(NR), rout=np.zeros(NR)) for g in G}
    for bid in nontr:
        if bid not in ctrl.index or (bid, "pre") not in st.index:
            continue
        r = b.loc[bid]
        pre, post = st.loc[(bid, "pre")], st.loc[(bid, "post")]
        wp = winp(r.match_id, r.team_id, pre.t_start, pre.t_end, False)
        wq = winp(r.match_id, r.team_id, post.t_start, post.t_end, True)
        c = ctrl.loc[bid]
        tc = c.control_t_center
        cmid, ctid = int(c.control_match_id), int(c.control_team_id)
        wcp = winp(cmid, ctid, tc - WIN, tc, False)
        wcq = winp(cmid, ctid, tc, tc + WIN, True)
        if any(w is None or len(w) < 10 for w in [wp, wq, wcp, wcq]):
            continue
        outs = [int(x) for x in str(r.players_out_id).split("|")]
        ins = [int(x) for x in str(r.players_in_id).split("|")]
        sw = outs + ins
        A = agg[r.timing_group]
        mid = r.match_id
        # zone-layer DiD
        zvq, zmq = zvec(wq, sw)
        zvp, zmp = zvec(wp, sw)
        zcq, zcmq = zvec(wcq)
        zcp, zcmp = zvec(wcp)
        A["znode"] += (zvq - zvp) - (zcq - zcp)
        A["zedge"] += (zmq - zmp) - (zcmq - zcmp)
        # role-layer DiD, teammates-only
        rvq, rmq = rolevec(wq, mid, pos, sw)
        rvp, rmp = rolevec(wp, mid, pos, sw)
        rcq, rcmq = rolevec(wcq, cmid, pos)
        rcp, rcmp = rolevec(wcp, cmid, pos)
        rn = (rvq - rvp) - (rcq - rcp)
        A["rnode"] += rn
        A["rnode_sq"] += rn * rn
        A["redge"] += (rmq - rmp) - (rcmq - rcmp)
        A["rbase"] += rvp
        for o in outs:
            ro = pos.get((mid, o))
            if ro:
                A["rout"][RIDX[ro]] += 1
        A["n"] += 1
    out = {}
    for g, A in agg.items():
        n = A["n"]
        out[f"n_{g}"] = np.array([n])
        out[f"znode_{g}"] = A["znode"] / n
        out[f"zedge_{g}"] = A["zedge"] / n
        rn_mean = A["rnode"] / n
        rn_var = np.maximum(A["rnode_sq"] / n - rn_mean ** 2, 0)
        out[f"rnode_{g}"] = rn_mean
        out[f"rnode_se_{g}"] = np.sqrt(rn_var / n)
        out[f"redge_{g}"] = A["redge"] / n
        out[f"rbase_{g}"] = A["rbase"] / n
        out[f"rout_{g}"] = A["rout"] / max(A["rout"].sum(), 1)
    np.savez(CACHE, **out)
    return out


def draw_stack(ax, x0, D, g, znode, zse, rnorm, znorm, glob_ze, glob_re):
    # lower layer: zone response
    draw_backing(ax, 0, "#DCE6EE", 1, x0)
    for k in range(24):
        ix, iy = divmod(k, NY)
        cx0, cy0 = ix * 20.0, iy * 20.0
        xs, ys = proj([cx0, cx0 + 20, cx0 + 20, cx0], [cy0, cy0, cy0 + 20, cy0 + 20], 0, x0)
        ax.add_patch(Polygon(np.c_[xs, ys], closed=True, facecolor=CMAP_D(znorm(znode[k])),
                             edgecolor="white", linewidth=0.5, zorder=2))
        if np.abs(znode[k]) >= 2 * zse[k]:
            px, py = proj(cx0 + 17.0, cy0 + 17.0, 0, x0)
            ax.plot(px, py, ".", color=P["neutral_black"], ms=2.2, zorder=6)
    draw_pitch(ax, 0, "#FFFFFF", 0.55, 3, x0)
    draw_border(ax, 0, 6, x0)
    zc = {k: ((k // NY + 0.5) * 20.0, (k % NY + 0.5) * 20.0) for k in range(24)}
    E = D[f"zedge_{g}"]
    idx = np.dstack(np.unravel_index(np.argsort(-np.abs(E), axis=None), E.shape))[0]
    shown = 0
    for i, j in idx:
        v = E[i, j]
        if i == j or shown >= 10 or abs(v) < 0.30 * glob_ze:
            continue
        xa, ya = proj(*zc[i], 0, x0)
        xb, yb = proj(*zc[j], 0, x0)
        ax.add_patch(FancyArrowPatch(
            (xa, ya), (xb, yb), arrowstyle="-|>",
            mutation_scale=4.5 + 3.0 * abs(v) / glob_ze, lw=0.5 + 1.8 * abs(v) / glob_ze,
            color=C_GAIN if v > 0 else C_LOSS, linestyle="-" if v > 0 else (0, (3, 2)),
            alpha=0.8, zorder=5, shrinkA=2, shrinkB=2, connectionstyle="arc3,rad=0.08"))
        shown += 1
    # inter-layer: perturbation line from the most withdrawn role
    rout = D[f"rout_{g}"]
    top_role = int(np.argmax(rout))
    xa, ya = proj(*ROLE_TEMPLATE[top_role], 1, x0)
    xb, yb = proj(60, 40, 0, x0)
    ax.add_patch(FancyArrowPatch((xa, ya), (xb, yb), arrowstyle="-|>", mutation_scale=9,
                                 lw=1.8, color=P["accent"], alpha=0.9, zorder=7,
                                 shrinkA=7, shrinkB=3))
    # upper layer: role response network
    draw_backing(ax, 1, "#EEF3F8", 8, x0)
    draw_pitch(ax, 1, P["neutral_mid"], 0.5, 8, x0)
    draw_border(ax, 1, 10, x0)
    R = D[f"redge_{g}"]
    idx = np.dstack(np.unravel_index(np.argsort(-np.abs(R), axis=None), R.shape))[0]
    segs_g, lws_g, segs_l, lws_l = [], [], [], []
    shown = 0
    for i, j in idx:
        v = R[i, j]
        if i == j or shown >= 12 or abs(v) < 0.30 * glob_re:
            continue
        xa, ya = proj(*ROLE_TEMPLATE[i], 1, x0)
        xb, yb = proj(*ROLE_TEMPLATE[j], 1, x0)
        if v > 0:
            segs_g.append([(xa, ya), (xb, yb)])
            lws_g.append(0.5 + 2.6 * abs(v) / glob_re)
        else:
            segs_l.append([(xa, ya), (xb, yb)])
            lws_l.append(0.5 + 2.6 * abs(v) / glob_re)
        shown += 1
    ax.add_collection(LineCollection(segs_g, colors=C_GAIN, linewidths=lws_g,
                                     alpha=0.8, zorder=10, capstyle="round"))
    ax.add_collection(LineCollection(segs_l, colors=C_LOSS, linewidths=lws_l,
                                     alpha=0.75, zorder=10, capstyle="round",
                                     linestyle=(0, (3, 2))))
    rn = D[f"rnode_{g}"]
    rse = D[f"rnode_se_{g}"]
    rbase = D[f"rbase_{g}"]
    for i, role in enumerate(ROLES):
        x, y = proj(*ROLE_TEMPLATE[i], 1, x0)
        ax.scatter([x], [y], s=30 + 260 * rbase[i] / rbase.max(),
                   c=[CMAP_D(rnorm(rn[i]))], edgecolor=P["neutral_dark"], linewidth=0.5,
                   zorder=12)
        if np.abs(rn[i]) >= 2 * rse[i]:
            ax.plot(x, y, ".", color=P["neutral_black"], ms=2.4, zorder=13)
        if i == int(np.argmax(rout)):
            ax.scatter([x], [y], s=210, facecolor="none", edgecolor=P["accent"],
                       linewidth=1.3, zorder=13)
        ax.text(x, y - 4.4, role, fontsize=5.0, ha="center", va="top",
                color=P["neutral_dark"], zorder=14)


def main():
    npz = np.load(TAB / "reorg_maps.npz")
    groups = [("strategic", "early ($\\leq$60$'$)"), ("regular", "mid (61–75$'$)"),
              ("late", "late (>75$'$)")]
    if CACHE.exists():
        D = dict(np.load(CACHE))
        print(f"loaded cache {CACHE.name}")
    else:
        print("aggregating (1-3 min)...")
        D = aggregate()
    for g, _ in groups:
        n_h, n_f = int(D[f"n_{g}"][0]), int(npz[f"n_{g}"][0])
        dmax = np.abs(D[f"znode_{g}"] - npz[f"mean_{g}"]).max()
        print(f"  {g}: n={n_h} (frozen {n_f}) zone-node mean deviation={dmax:.2e}")
        assert n_h == n_f and dmax < 1e-9, f"{g}: mismatch with frozen reorg_maps.npz"
    gse = pd.read_csv(TAB / "gradient_selfexcl.csv").set_index("timing")
    cg = pd.read_csv(TAB / "cross_gradient_clean.csv")
    snr = {g: gse.loc[g, "snr_excl"] for g, _ in groups}
    # shared colour scales
    zvmax = max(np.abs(npz[f"mean_{g}"]).max() for g, _ in groups)
    rvmax = max(np.abs(D[f"rnode_{g}"]).max() for g, _ in groups)
    znorm = TwoSlopeNorm(vcenter=0, vmin=-zvmax, vmax=zvmax)
    rnorm = TwoSlopeNorm(vcenter=0, vmin=-rvmax, vmax=rvmax)
    glob_ze = max(np.abs(D[f"zedge_{g}"]).max() for g, _ in groups)
    glob_re = max(np.abs(D[f"redge_{g}"]).max() for g, _ in groups)
    w = cg.pivot(index="comp", columns="timing", values="snr_excl")
    n_le = int((w["late"] > w["strategic"]).sum())
    n_me = int((w["regular"] > w["strategic"]).sum())
    n_lm = int((w["late"] > w["regular"]).sum())
    ncomp = len(w)
    for g, _ in groups:
        nsz = int((np.abs(D[f"rnode_{g}"]) >= 2 * D[f"rnode_se_{g}"]).sum())
        print(f"  {g}: significant role nodes={nsz}, most withdrawn={ROLES[int(np.argmax(D[f'rout_{g}']))]}")

    fig = plt.figure(figsize=(7.48, 4.55))
    axa = fig.add_axes([0.015, 0.42, 0.97, 0.565])
    axa.set_axis_off()
    axa.set_anchor("N")
    XSPACE = 168.0
    for gi, (g, glab) in enumerate(groups):
        x0 = gi * XSPACE
        draw_stack(axa, x0, D, g, npz[f"mean_{g}"], npz[f"se_{g}"], rnorm, znorm,
                   glob_ze, glob_re)
        tx, _ = proj(60, 40, 1, x0)
        axa.text(tx, SHY * PW + GAP + 15.0, glab, fontsize=8.5, weight="bold",
                 ha="center", color=P["neutral_black"])
        axa.text(tx, SHY * PW + GAP + 9.2,
                 f"mean over $n$ = {int(D[f'n_{g}'][0])} substitutions", fontsize=6.3,
                 ha="center", color=P["neutral_mid"], style="italic")
        sig = " (noise level)" if snr[g] < 1.2 else ""
        axa.text(tx, -15.0, f"SNR = {snr[g]:.2f}{sig}", fontsize=7.5, ha="center",
                 color=P["neutral_black"] if snr[g] > 1.2 else P["neutral_mid"])
    # row labels
    axa.text(-13, SHY * PW + GAP + 6.0, "player layer", fontsize=6.5, ha="left",
             color=P["neutral_dark"], style="italic", zorder=20)
    axa.text(-13, SHY * PW + GAP + 1.6, "role response $\\langle\\Delta\\rangle$",
             fontsize=5.8, ha="left", color=P["neutral_mid"], style="italic", zorder=20)
    axa.text(-13, SHY * PW + 12.5, "zone layer", fontsize=6.5, ha="left",
             color=P["neutral_dark"], style="italic", zorder=20,
             bbox=dict(boxstyle="round,pad=0.12", facecolor="white", alpha=0.75,
                       edgecolor="none"))
    axa.text(-13, SHY * PW + 7.8, "zone response $\\langle d\\rangle$", fontsize=5.8,
             ha="left", color=P["neutral_mid"], style="italic", zorder=20,
             bbox=dict(boxstyle="round,pad=0.12", facecolor="white", alpha=0.75,
                       edgecolor="none"))
    # colourbar and edge legend
    cbx = 2 * XSPACE + 122
    axa.text(cbx + 13, -4.5, "participation change (red = gained, blue = lost;",
             fontsize=5.8, ha="center", color=P["neutral_black"], zorder=20)
    axa.text(cbx + 13, -8.8, "layers scaled independently)",
             fontsize=5.8, ha="center", color=P["neutral_black"], zorder=20)
    cax = axa.inset_axes([cbx + 1, -13.5, 24, 1.9], transform=axa.transData)
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=CMAP_D, norm=znorm), cax=cax,
                      orientation="horizontal")
    cb.set_ticks([-zvmax, 0, zvmax])
    cb.set_ticklabels([f"$-${zvmax:.2f}", "0", f"+{zvmax:.2f}"])
    cb.ax.tick_params(labelsize=5.4, width=0.5, length=2)
    cb.outline.set_linewidth(0.5)
    axa.text(cbx + 13, -18.6, "(zone-layer scale)", fontsize=5.2, ha="center",
             color=P["neutral_mid"], zorder=20)
    axa.plot([cbx - 3, cbx + 3], [-24.5, -24.5], color=C_GAIN, lw=1.6, zorder=20)
    axa.text(cbx + 6, -24.5, "link strengthened", fontsize=5.8, va="center",
             color=P["neutral_black"], zorder=20)
    axa.plot([cbx - 3, cbx + 3], [-29.5, -29.5], color=C_LOSS, lw=1.4,
             ls=(0, (3, 2)), zorder=20)
    axa.text(cbx + 6, -29.5, "link weakened", fontsize=5.8, va="center",
             color=P["neutral_black"], zorder=20)
    axa.plot([cbx - 3, cbx + 3], [-34.5, -34.5], color=P["accent"], lw=1.8, zorder=20)
    axa.text(cbx + 6, -34.5, "perturbation (role replaced)", fontsize=5.8, va="center",
             color=P["neutral_black"], zorder=20)
    axa.set_xlim(-14, 2 * XSPACE + 172)
    axa.set_ylim(-38, SHY * PW + GAP + 21)
    axa.set_aspect("equal")
    axa.text(0.0, 0.99, "(a)", transform=axa.transAxes, fontsize=10, weight="bold")

    # (b)
    axb = fig.add_axes([0.095, 0.085, 0.52, 0.37])
    xs = np.array([0, 1, 2])
    order = ["strategic", "regular", "late"]
    for comp, row in w.iterrows():
        axb.plot(xs, row[order].values, color=P["neutral_light"], lw=0.9, alpha=0.85,
                 zorder=2, solid_capstyle="round")
    laliga = [snr[g] for g in order]
    axb.plot(xs, laliga, color=P["signal_dark"], lw=2.0, marker="o", ms=4.5,
             markeredgecolor="white", markeredgewidth=0.8, zorder=5)
    axb.axhline(1.0, color=P["neutral_mid"], lw=0.8, ls=(0, (4, 3)), zorder=1)
    axb.axvline(0.5, color=P["accent"], lw=1.0, ls=(0, (5, 3)), zorder=3)
    axb.annotate("threshold $\\sim$60$'$", xy=(0.5, 2.30), fontsize=7,
                 color=P["accent"], ha="center",
                 bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.9,
                           edgecolor="none"))
    axb.text(2.08, laliga[2] + 0.03, "La Liga 23/24", fontsize=6.8,
             color=P["signal_dark"], va="center")
    axb.text(2.08, 1.30, "10 open-data\ncompetitions", fontsize=6.5,
             color=P["neutral_mid"], va="center")
    axb.text(0.55, 0.62,
             f"late>early {n_le}/{ncomp}   mid>early {n_me}/{ncomp} ($p$=0.002)\n"
             f"late>mid {n_lm}/{ncomp} (n.s.): threshold, not dose–response",
             fontsize=6.4, color=P["neutral_black"],
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85,
                       edgecolor="#DDDDDD", linewidth=0.5))
    axb.set_xticks(xs)
    axb.set_xticklabels(["early", "mid", "late"])
    axb.set_ylabel("SNR (teammates-only, DiD)", fontsize=8)
    axb.set_xlim(-0.2, 2.75)
    axb.set_ylim(0.50, 2.45)
    axb.set_yticks([1.0, 1.5, 2.0])
    axb.text(-0.13, 1.03, "(b)", transform=axb.transAxes, fontsize=10, weight="bold")

    # (c)
    axc = fig.add_axes([0.735, 0.085, 0.235, 0.37])
    lim = [0.85, 2.35]
    axc.plot(lim, lim, ls="--", color="k", lw=0.8, alpha=0.4, zorder=1)
    TCOL = {"strategic": P["neutral_light"], "regular": P["signal_mid"],
            "late": P["signal_dark"]}
    for t in order:
        gg = cg[cg.timing == t]
        axc.plot(gg.snr_incl, gg.snr_excl, "o", color=TCOL[t], ms=3.6,
                 mec="white", mew=0.4, zorder=4)
        axc.plot(gse.loc[t, "snr_incl"], gse.loc[t, "snr_excl"], "s", color=TCOL[t],
                 ms=5, mec=P["neutral_black"], mew=0.6, zorder=5)
    axc.text(1.13, 0.97, "early", fontsize=6.3, color=P["neutral_mid"])
    axc.text(1.68, 1.28, "mid", fontsize=6.3, color=P["signal_mid"])
    axc.text(2.02, 1.72, "late", fontsize=6.3, color=P["signal_dark"])
    axc.text(1.55, 0.93, "on the diagonal:\nresponse carried\nby teammates",
             fontsize=6.2, color=P["neutral_black"])
    axc.set_xlabel("SNR incl. substituted players", fontsize=7)
    axc.set_ylabel("SNR teammates-only", fontsize=7)
    axc.set_xlim(lim)
    axc.set_ylim(lim)
    axc.set_xticks([1.0, 1.5, 2.0])
    axc.set_yticks([1.0, 1.5, 2.0])
    axc.set_aspect("equal")
    axc.text(-0.30, 1.03, "(c)", transform=axc.transAxes, fontsize=10, weight="bold")

    for ext, kw in [("svg", {}), ("pdf", {}),
                    ("tiff", dict(dpi=600, pil_kwargs={"compression": "tiff_lzw"})),
                    ("png", dict(dpi=300))]:
        fig.savefig(FIG / f"fig2_role_responses_diagnostic.{ext}", **kw)
    plt.close(fig)
    print("saved fig2_role_responses_diagnostic.{svg,pdf,tiff,png}")


if __name__ == "__main__":
    main()
