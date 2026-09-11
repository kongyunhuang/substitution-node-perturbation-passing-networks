"""
Draw Figure 1, the two-layer framework for one substitution.

Shear-projected pair of pitches: the player passing network (upper layer) above the 6x4 zone
layer coloured by teammates-only |dz| with the zone-to-zone flow network; player-to-zone
participation links between them, the substituted player's footprint in accent. The example
is the single-player late substitution with the largest teammates-only |dz|.
Inputs: data/{events_pass,did_controls}.parquet, data/substitution_batches.csv, data/windows/*.parquet
Outputs: results/figures/fig1.{pdf,png}
Run: python code/figures/fig1.py
"""

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, Polygon
from matplotlib.transforms import Affine2D
from mplsoccer import Pitch

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
FIG = ROOT / "results" / "figures"
NX, NY = 6, 4
PL, PW = 120, 80

# figure style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "cm", "axes.unicode_minus": False,
    "font.size": 8, "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})
P = {
    "neutral_dark": "#4D4D4D", "neutral_mid": "#767676", "neutral_light": "#CFCECE",
    "neutral_black": "#272727", "signal_dark": "#0B3954", "signal_mid": "#087E8B",
    "signal_light": "#5ABEAA", "accent": "#FF5A5F", "accent_light": "#FF8A80",
    "accent_pale": "#F5CBA7",
}
CMAP_Z = LinearSegmentedColormap.from_list("ocean", ["#F0F7FA", "#87CEEB", "#087E8B", "#0B3954"])
C_EDGE_P = "#2E3B4E"      # player-layer pass edges
C_EDGE_Z = "#16324A"      # zone-flow edges
C_LINK = "#9FB4C7"        # player-to-zone links (teammates)

# shear projection (x, y, layer) -> canvas; StatsBomb y grows downwards, so proj flips y
SHX, SHY, GAP = 0.42, 0.46, 62.0   # y shear, y compression, layer gap


def proj(x, y, lay):
    x, yd = np.asarray(x, float), PW - np.asarray(y, float)
    return x + SHX * yd, SHY * yd + lay * GAP


def zbin(x, y):
    zx = np.clip((x / (PL / NX)).astype(int), 0, NX - 1)
    zy = np.clip((y / (PW / NY)).astype(int), 0, NY - 1)
    return zx * NY + zy


def zvec(w, ex_ids=None):
    if ex_ids is not None:
        w = w[~w.player_id.isin(ex_ids) & ~w.recipient_id.isin(ex_ids)]
    Z = np.zeros((NX * NY, NX * NY))
    if len(w):
        np.add.at(Z, (zbin(w.x.values, w.y.values), zbin(w.end_x.values, w.end_y.values)), 1.0)
    np.fill_diagonal(Z, 0)
    t = Z / Z.sum() if Z.sum() > 0 else Z
    return t.sum(0) + t.sum(1), Z


def load():
    passes = pd.read_parquet(DATA / "events_pass.parquet")
    net = passes[passes.outcome.isna() & ~passes.is_set_piece].copy()
    t1 = passes[passes.period == 1].groupby("match_id").t_period_sec.max()
    net["tc"] = np.where(net.period == 1, net.t_period_sec, net.match_id.map(t1) + net.t_period_sec)
    b = pd.read_csv(DATA / "substitution_batches.csv").set_index("batch_id")
    sp = pd.read_parquet(DATA / "windows" / "window_specs.parquet")
    st = sp[(sp.kind == "time") & (sp.W_min == 15)].set_index(["batch_id", "side"])
    fl = pd.read_parquet(DATA / "windows" / "batch_flags.parquet")
    nontr = set(fl[(fl.W_min == 15) & ~fl.post_truncated].batch_id)
    ctrl = pd.read_parquet(DATA / "did_controls.parquet")
    okc = set(ctrl[ctrl.control_type != "none"].batch_id)
    return net, b, st, nontr, okc


def pick_batch(net, b, st, nontr, okc):
    best, best_mag = None, -1
    cand = b[(b.timing_group == "late") & (b.n_subs == 1)]
    cand = cand[cand.index.isin(nontr & okc)]
    for bid, r in cand.iterrows():
        if (bid, "pre") not in st.index:
            continue
        pre, post = st.loc[(bid, "pre")], st.loc[(bid, "post")]
        g = net[(net.match_id == r.match_id) & (net.team_id == r.team_id)]
        wp = g[(g.tc >= pre.t_start) & (g.tc < pre.t_end)]
        wq = g[(g.tc > post.t_start) & (g.tc <= post.t_end)]
        out_id = int(str(r.players_out_id).split("|")[0])
        in_id = int(str(r.players_in_id).split("|")[0])
        n_out = ((wp.player_id == out_id) | (wp.recipient_id == out_id)).sum()
        if len(wp) < 90 or len(wq) < 60 or n_out < 10:
            continue
        sw = [out_id, in_id]
        zp, _ = zvec(wp, sw)
        zq, _ = zvec(wq, sw)
        mag = np.linalg.norm(zq - zp)
        if mag > best_mag:
            best_mag, best = mag, (bid, r, wp, wq, out_id, in_id)
    return best, best_mag


# pitch drawing: mplsoccer StatsBomb pitch under the shear affine
def draw_backing(ax, lay, color, z0):
    """Layer backing, slightly larger than the pitch so goals stay visible."""
    xs, ys = proj([-3.5, 123.5, 123.5, -3.5], [-2.5, -2.5, 82.5, 82.5], lay)
    ax.add_patch(Polygon(np.c_[xs, ys], closed=True, facecolor=color,
                         edgecolor="none", zorder=z0))


def draw_pitch(ax, lay, line_c, lw=0.9, z0=1):
    before = set(ax.get_children())
    Pitch(pitch_type="statsbomb", pitch_color="none", line_color=line_c,
          linewidth=lw, goal_type="box", corner_arcs=True).draw(ax=ax)
    # same map as proj(), incl. the y flip
    tr = Affine2D(np.array([[1.0, -SHX, SHX * PW],
                            [0.0, -SHY, SHY * PW + lay * GAP],
                            [0.0, 0.0, 1.0]]))
    for art in ax.get_children():
        if art not in before:
            art.set_transform(tr + ax.transData)
            art.set_zorder(z0 + 1)
            art.set_clip_on(False)
    # attack-direction arrow above the top touchline
    ax0, ay0 = proj(86, -4.5, lay)
    ax1, ay1 = proj(112, -4.5, lay)
    ax.add_patch(FancyArrowPatch((ax0, ay0), (ax1, ay1), arrowstyle="-|>",
                                 mutation_scale=9, lw=1.0, color=P["neutral_mid"], zorder=z0 + 1))
    tx, ty = proj(99, -8.5, lay)
    ax.text(tx, ty, "attack", fontsize=7, color=P["neutral_mid"], ha="center", style="italic")


def main():
    net, b, st, nontr, okc = load()
    picked, mag = pick_batch(net, b, st, nontr, okc)
    bid, r, wp, wq, out_id, in_id = picked
    minute = int(r.minute_first)
    print(f"picked batch {bid}: {r.team_name} match {r.match_id} {minute}' "
          f"out={str(r.players_out).split('|')[0]} |dz_excl|={mag:.3f}")

    sw = [out_id, in_id]
    zp, _ = zvec(wp, sw)
    zq, _ = zvec(wq, sw)
    dz = np.abs(zq - zp)
    _, Zmat = zvec(wp)

    ppl = wp.groupby("player_id").agg(x=("x", "mean"), y=("y", "mean"), n=("x", "size"))
    ppl = ppl[ppl.n >= 3]
    edges = wp.groupby(["player_id", "recipient_id"]).size()
    wp2 = wp.copy()
    wp2["zone"] = zbin(wp2.x.values, wp2.y.values)
    part = wp2.groupby(["player_id", "zone"]).size()
    zc = {k: ((k // NY + 0.5) * PL / NX, (k % NY + 0.5) * PW / NY) for k in range(NX * NY)}

    fig, ax = plt.subplots(figsize=(5.51, 6.4))
    ax.set_axis_off()

    # lower layer: zones
    draw_backing(ax, 0, "#DCE6EE", 1)
    vmax = dz.max()
    for k in range(NX * NY):
        ix, iy = divmod(k, NY)
        x0, y0 = ix * PL / NX, iy * PW / NY
        xs, ys = proj([x0, x0 + PL / NX, x0 + PL / NX, x0], [y0, y0, y0 + PW / NY, y0 + PW / NY], 0)
        ax.add_patch(Polygon(np.c_[xs, ys], closed=True,
                             facecolor=CMAP_Z(0.06 + 0.88 * dz[k] / vmax),
                             edgecolor="white", linewidth=0.8, zorder=2))
    draw_pitch(ax, 0, "#FFFFFF", lw=0.9, z0=3)   # white pitch lines over the cells
    # zone flow (pre-window, whole team): top edges only
    fmax = Zmat.max()
    flows = [(i, j, Zmat[i, j]) for i in range(NX * NY) for j in range(NX * NY)
             if i != j and Zmat[i, j] >= 3]
    flows.sort(key=lambda t: -t[2])
    for i, j, n in flows[:12]:
        x0, y0 = proj(*zc[i], 0)
        x1, y1 = proj(*zc[j], 0)
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=6 + 5 * n / fmax,
                                     lw=0.6 + 2.6 * n / fmax, color=C_EDGE_Z,
                                     alpha=0.72, zorder=5,
                                     shrinkA=2.5, shrinkB=2.5,
                                     connectionstyle="arc3,rad=0.08"))
    act = np.array([Zmat[k].sum() + Zmat[:, k].sum() for k in range(NX * NY)])
    zxs, zys = proj([zc[k][0] for k in range(NX * NY)], [zc[k][1] for k in range(NX * NY)], 0)
    ax.scatter(zxs, zys, s=10, c="#12283A",
               edgecolor="white", linewidth=0.4, zorder=6)

    # inter-layer participation links
    for pid in ppl.index:
        if pid not in part.index.get_level_values(0):
            continue
        pz = part.loc[pid].sort_values(ascending=False)
        tot = pz.sum()
        for zone, cnt in pz.head(2).items():
            share = cnt / tot
            x0, y0 = proj(ppl.loc[pid, "x"], ppl.loc[pid, "y"], 1)
            x1, y1 = proj(*zc[zone], 0)
            if pid == out_id:
                ax.plot([x0, x1], [y0, y1], color=P["accent"], lw=0.9 + 1.8 * share,
                        alpha=0.95, zorder=7, solid_capstyle="round")
            else:
                ax.plot([x0, x1], [y0, y1], color=C_LINK, lw=0.55 + 1.2 * share,
                        alpha=0.38, zorder=4)

    # upper layer: players
    draw_backing(ax, 1, "#EEF3F8", 8)
    draw_pitch(ax, 1, P["neutral_mid"], lw=0.8, z0=8)
    # plane outlines
    bx, by = proj([0, PL, PL, 0, 0], [0, 0, PW, PW, 0], 1)
    ax.plot(bx, by, color=P["neutral_dark"], lw=1.1, zorder=9)
    bx, by = proj([0, PL, PL, 0, 0], [0, 0, PW, PW, 0], 0)
    ax.plot(bx, by, color=P["neutral_dark"], lw=1.1, zorder=6)

    # directed edges: one curved arrow per direction
    emax = edges.max()
    for (pi, pj), n in edges.items():
        if n < 2 or pi not in ppl.index or pj not in ppl.index:
            continue
        x0, y0 = proj(ppl.loc[pi, "x"], ppl.loc[pi, "y"], 1)
        x1, y1 = proj(ppl.loc[pj, "x"], ppl.loc[pj, "y"], 1)
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=3.2 + 2.8 * n / emax,
                                     lw=0.45 + 3.0 * n / emax, color=C_EDGE_P,
                                     alpha=0.75, zorder=10,
                                     shrinkA=3.0, shrinkB=4.5,
                                     connectionstyle="arc3,rad=0.12"))
    for pid, row in ppl.iterrows():
        x, y = proj(row.x, row.y, 1)
        if pid == out_id:
            ax.scatter([x], [y], s=150 + 90 * row.n / ppl.n.max(), c=P["accent"],
                       edgecolor="white", linewidth=1.2, zorder=13)
            ax.scatter([x], [y], s=430, facecolor="none", edgecolor=P["accent"],
                       linewidth=1.6, zorder=13)
        else:
            ax.scatter([x], [y], s=42 + 150 * row.n / ppl.n.max(), c=P["signal_dark"],
                       edgecolor="white", linewidth=0.9, zorder=12)

    # annotations
    xo, yo = proj(ppl.loc[out_id, "x"], ppl.loc[out_id, "y"], 1)
    ax.annotate(f"substituted off ($t_0$ = {minute}$'$)",
                xy=(xo, yo), xytext=(xo - 22, yo + 27),
                fontsize=8, color=P["accent"], zorder=20,
                arrowprops=dict(arrowstyle="-|>", lw=0.9, color=P["accent"],
                                shrinkB=8, connectionstyle="arc3,rad=-0.18"))
    # layer titles outside the planes
    ty_top = SHY * PW + GAP        # top edge of the upper plane
    ax.text(-10, ty_top + 8.5, "Player layer", fontsize=9.5,
            color=P["neutral_black"], weight="bold", zorder=20)
    ax.text(-10, ty_top + 3.6, "perturbation: node replacement",
            fontsize=7.5, color=P["neutral_dark"], style="italic", zorder=20)
    ax.text(-10, -6.5, "Pitch-passing layer", fontsize=9.5,
            color=P["neutral_black"], weight="bold", zorder=20)
    ax.text(-10, -11.6, "response: flow reorganization $|\\Delta z|$, teammates only",
            fontsize=7.5, color=P["neutral_dark"], style="italic", zorder=20)

    # legend, lower right
    lx, ly = 121.0, -5.0
    items = [("passes (player layer)", C_EDGE_P, 2.2, 0.8),
             ("zone-to-zone flow", C_EDGE_Z, 1.8, 0.75),
             ("player→zone participation", C_LINK, 1.0, 0.5),
             ("outgoing player footprint $f_{out}$", P["accent"], 1.6, 0.95)]
    for k, (lab, c, lw, al) in enumerate(items):
        ax.plot([lx, lx + 9], [ly - 5.6 * k] * 2, color=c, lw=lw, alpha=al,
                solid_capstyle="round", zorder=20)
        ax.text(lx + 11.5, ly - 5.6 * k, lab, fontsize=6.6, color=P["neutral_black"],
                va="center", zorder=20)

    # |dz| colourbar below the legend
    cax = ax.inset_axes([lx + 1, ly - 5.6 * 3 - 8.5, 26, 2.0], transform=ax.transData)
    sm = plt.cm.ScalarMappable(cmap=CMAP_Z, norm=plt.Normalize(0, vmax))
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=5.8, width=0.5, length=2)
    cb.set_ticks([0, vmax])
    cb.set_ticklabels(["0", f"{vmax:.3f}"])
    cb.outline.set_linewidth(0.5)
    ax.text(lx + 14, ly - 5.6 * 3 - 15.2, "$|\\Delta z|$ (teammates only)",
            fontsize=6.2, color=P["neutral_black"], ha="center", va="top", zorder=20)
    # example source note
    ax.text(-10, -20.5, f"example: La Liga 2023/24, late substitution ({minute}$'$), "
            "windows $\\pm$15 min", fontsize=6.2, color=P["neutral_mid"], zorder=20)

    ax.set_xlim(-14, 174)
    ax.set_ylim(-37, GAP + SHY * PW + 16)
    ax.set_aspect("equal")

    for ext, kw in [("pdf", {}), ("png", dict(dpi=300))]:
        fig.savefig(FIG / f"fig1.{ext}", **kw)
    plt.close(fig)
    print("saved fig1.{pdf,png}")


if __name__ == "__main__":
    main()
