"""
Role-to-zone coupling responses from the measured inter-layer network (input to Figure 2).

Teammates-only DiD of the role-by-zone endpoint participation matrix (10 roles x 24 zones)
per timing group, drawn as the inter-layer links of Figure 2a. Row sums must reproduce the
role-layer DiD cached in role_did.npz (tolerance 1e-9).
Inputs: data/{events_pass,did_controls,player_positions}.parquet, data/substitution_batches.csv,
        data/windows/*.parquet, results/tables/{role_did.npz,reorg_maps.npz} (checks only)
Outputs: results/tables/coupling_didzone.npz (cc_{group} 10x24, n_{group})
Run: python code/figures/fig2_role_zone_coupling.py
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
TAB = ROOT / "results" / "tables"
NX, NY = 6, 4
PL, PW = 120, 80
WIN = 900

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


def zbin(x, y):
    zx = np.clip((x / (PL / NX)).astype(int), 0, NX - 1)
    zy = np.clip((y / (PW / NY)).astype(int), 0, NY - 1)
    return zx * NY + zy


def couplingvec(w, mid, pos, ex_ids=None):
    """C[10 roles, 24 zones]: endpoint participation share (sums to 1); ex_ids for teammates-only."""
    if ex_ids is not None:
        w = w[~w.player_id.isin(ex_ids) & ~w.recipient_id.isin(ex_ids)]
    C = np.zeros((NR, 24))
    if not len(w):
        return C
    # endpoint 1: passer role x origin zone
    rp = w.player_id.map(lambda p: pos.get((mid, p)))
    okp = rp.notna()
    if okp.sum():
        np.add.at(C, ([RIDX[a] for a in rp[okp]],
                      zbin(w.x.values[okp.values], w.y.values[okp.values])), 1.0)
    # endpoint 2: receiver role x destination zone
    rr = w.recipient_id.map(lambda p: pos.get((mid, p)))
    okr = rr.notna()
    if okr.sum():
        np.add.at(C, ([RIDX[a] for a in rr[okr]],
                      zbin(w.end_x.values[okr.values], w.end_y.values[okr.values])), 1.0)
    s = C.sum()
    return C / s if s > 0 else C


def main():
    passes = pd.read_parquet(DATA / "events_pass.parquet")
    net = passes[passes.outcome.isna() & ~passes.is_set_piece].copy()
    t1 = passes[passes.period == 1].groupby("match_id").t_period_sec.max()
    net["tc"] = np.where(net.period == 1, net.t_period_sec,
                         net.match_id.map(t1) + net.t_period_sec)
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
    agg = {g: dict(n=0, cc=np.zeros((NR, 24))) for g in G}
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
        cq = couplingvec(wq, r.match_id, pos, sw)
        cp = couplingvec(wp, r.match_id, pos, sw)
        ccq = couplingvec(wcq, cmid, pos)
        ccp = couplingvec(wcp, cmid, pos)
        A["cc"] += (cq - cp) - (ccq - ccp)
        A["n"] += 1

    out = {}
    for g, A in agg.items():
        out[f"n_{g}"] = np.array([A["n"]])
        out[f"cc_{g}"] = A["cc"] / A["n"]
    np.savez(TAB / "coupling_didzone.npz", **out)

    # consistency checks
    D = dict(np.load(TAB / "role_did.npz"))
    npz = dict(np.load(TAB / "reorg_maps.npz"))
    print("coupling_didzone.npz checks")
    print("  (role margins must match cached rnode exactly; zone margins are informative only,")
    print("   cached znode drops same-zone passes while the coupling counts all endpoints)")
    ok = True
    for g in G:
        cc = out[f"cc_{g}"]
        n = int(out[f"n_{g}"][0])
        role_marg = cc.sum(1)   # sum over zones = role DiD, proportional to cached rnode
        zone_marg = cc.sum(0)   # sum over roles = zone DiD (includes same-zone passes)
        # normalise both, then compare elementwise
        rm_n = role_marg / np.abs(role_marg).sum()
        rn_n = D[f"rnode_{g}"] / np.abs(D[f"rnode_{g}"]).sum()
        role_dev = np.abs(rm_n - rn_n).max()
        r_zone = np.corrcoef(zone_marg, npz[f"mean_{g}"])[0, 1]
        print(f"  {g}: n={n}  role-margin deviation (normalised)={role_dev:.2e}  "
              f"zone-margin vs znode r={r_zone:.3f}  |cc|max={np.abs(cc).max():.4f}")
        ok &= (n == int(D[f"n_{g}"][0])) and role_dev < 1e-9
    assert [int(out[f"n_{g}"][0]) for g in G] == [425, 404, 73], "expected n = 425/404/73"
    assert ok, "role margins do not match cached rnode (tol 1e-9)"
    print("  checks passed; saved coupling_didzone.npz")


if __name__ == "__main__":
    main()
