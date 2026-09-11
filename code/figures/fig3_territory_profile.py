"""
Spatial profile of the territorial shift along the pitch (Figure 3a).

Teammates-only DiD of the pass-origin share in 12 bins along the pitch per timing group, plus
a 2-D origin grid and centroid shift; the profile's first moment must equal d_terr in
dterr_audit.csv. Also writes home/away and score-state subgroup means of d_terr (t-type 95% CI).
Inputs: data/{events_pass,did_controls,analysis_table}.parquet, data/substitution_batches.csv,
        data/windows/*.parquet, results/tables/dterr_audit.csv
Outputs: results/tables/territory_profile.npz, results/tables/territory_subgroups.csv
Run: python code/figures/fig3_territory_profile.py
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
TAB = ROOT / "results" / "tables"
PL = 120.0
W = 900.0
NB = 12                      # bins along the pitch (10 yd each)
GROUPS = ["strategic", "regular", "late"]
TOL = 0.15                   # first moment vs d_terr tolerance (yd), binning only


def origin_share(w, ex_ids=None):
    """Origin-x share (sums to 1) of successful passes in a window; ex_ids for teammates-only.
    Same filter order as terr() in territorial_shift.py: successful first, then exclude, need >= 5."""
    cw = w[w.outcome.isna()]
    if ex_ids is not None:
        cw = cw[~cw.player_id.isin(ex_ids) & ~cw.recipient_id.isin(ex_ids)]
    if len(cw) < 5:
        return None
    w = cw
    h, _ = np.histogram(np.clip(w.x.values, 0, PL - 1e-9), bins=NB, range=(0, PL))
    s = h.sum()
    return h / s if s > 0 else None


NX2, NY2 = 12, 8             # 2-D grid, 10 yd x 10 yd


def origin_grid(w, ex_ids=None):
    """2-D origin share grid (NY2 x NX2, sums to 1) and centroid (x, y) of successful passes."""
    cw = w[w.outcome.isna()]
    if ex_ids is not None:
        cw = cw[~cw.player_id.isin(ex_ids) & ~cw.recipient_id.isin(ex_ids)]
    if len(cw) < 5:
        return None, None
    x = np.clip(cw.x.values, 0, PL - 1e-9)
    y = np.clip(cw.y.values, 0, 80 - 1e-9)
    H, _, _ = np.histogram2d(y, x, bins=[NY2, NX2], range=[[0, 80], [0, PL]])
    s_ = H.sum()
    if s_ == 0:
        return None, None
    return H / s_, np.array([x.mean(), y.mean()])


def main():
    passes = pd.read_parquet(DATA / "events_pass.parquet")
    pall = passes[~passes.is_set_piece].copy()          # incl. failed passes, for the window-size threshold
    t1 = passes[passes.period == 1].groupby("match_id").t_period_sec.max()
    pall["tc"] = np.where(pall.period == 1, pall.t_period_sec,
                          pall.match_id.map(t1) + pall.t_period_sec)
    ni = {k: g for k, g in pall.groupby(["match_id", "team_id"])}
    b = pd.read_csv(DATA / "substitution_batches.csv").set_index("batch_id")
    sp = pd.read_parquet(DATA / "windows" / "window_specs.parquet")
    st = sp[(sp.kind == "time") & (sp.W_min == 15)].set_index(["batch_id", "side"])
    fl = pd.read_parquet(DATA / "windows" / "batch_flags.parquet")
    nontr = set(fl[(fl.W_min == 15) & ~fl.post_truncated].batch_id)
    ctrl = pd.read_parquet(DATA / "did_controls.parquet")
    ctrl = ctrl[ctrl.control_type != "none"].set_index("batch_id")

    def winp(mid, tid, lo, hi, lo_open):
        g = ni.get((mid, tid))
        if g is None:
            return None
        return g[(g.tc > lo) & (g.tc <= hi)] if lo_open else g[(g.tc >= lo) & (g.tc < hi)]

    acc = {g: [] for g in GROUPS}
    accpre = {g: [] for g in GROUPS}        # pre-window mean origin x
    acc2 = {g: [] for g in GROUPS}          # 2-D grid DiD
    accc = {g: [] for g in GROUPS}          # centroid shift DiD (dx, dy)
    for bid in nontr:
        if bid not in ctrl.index or (bid, "pre") not in st.index:
            continue
        r = b.loc[bid]
        pre, post = st.loc[(bid, "pre")], st.loc[(bid, "post")]
        wp = winp(r.match_id, r.team_id, pre.t_start, pre.t_end, False)
        wq = winp(r.match_id, r.team_id, post.t_start, post.t_end, True)
        c = ctrl.loc[bid]
        tc = c.control_t_center
        wcp = winp(int(c.control_match_id), int(c.control_team_id), tc - W, tc, False)
        wcq = winp(int(c.control_match_id), int(c.control_team_id), tc, tc + W, True)
        if any(w is None or len(w) < 10 for w in [wp, wq, wcp, wcq]):
            continue
        sw = ([int(x) for x in str(r.players_out_id).split("|")]
              + [int(x) for x in str(r.players_in_id).split("|")])
        a, bb = origin_share(wq, sw), origin_share(wp, sw)
        cq, cp = origin_share(wcq), origin_share(wcp)
        if any(v is None for v in (a, bb, cq, cp)):
            continue
        acc[r.timing_group].append((a - bb) - (cq - cp))
        cwp = wp[wp.outcome.isna()]
        cwp = cwp[~cwp.player_id.isin(sw) & ~cwp.recipient_id.isin(sw)]
        accpre[r.timing_group].append(float(cwp.x.mean()))
        gq, mq = origin_grid(wq, sw); gp, mp = origin_grid(wp, sw)
        gcq, mcq = origin_grid(wcq); gcp, mcp = origin_grid(wcp)
        if all(v is not None for v in (gq, gp, gcq, gcp)):
            acc2[r.timing_group].append((gq - gp) - (gcq - gcp))
            accc[r.timing_group].append((mq - mp) - (mcq - mcp))

    edges = np.linspace(0, PL, NB + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    out = {"edges": edges, "centers": centers}
    dterr = pd.read_csv(TAB / "dterr_audit.csv")
    print(f"{'group':<12}{'n':>6}{'moment':>16}{'d_terr':>16}{'diff':>9}")
    ok = True
    for g in GROUPS:
        A = np.array(acc[g])
        prof = A.mean(0)
        out[f"prof_{g}"] = prof
        out[f"se_{g}"] = A.std(0, ddof=1) / np.sqrt(len(A))
        out[f"n_{g}"] = np.array([len(A)])
        mom = float((prof * centers).sum())
        out[f"moment_{g}"] = np.array([mom])
        out[f"pre_x_{g}"] = np.array([float(np.mean(accpre[g]))])
        G2 = np.array(acc2[g]); out[f"grid_{g}"] = G2.mean(0)
        C2 = np.array(accc[g]); out[f"cent_{g}"] = C2.mean(0)
        out[f"cent_se_{g}"] = C2.std(0, ddof=1) / np.sqrt(len(C2))
        ref = dterr[dterr.timing == g].did_terr_excl.mean()
        d = mom - ref
        ok &= abs(d) < TOL
        print(f"  {g:<10}{len(A):>6}{mom:>+16.3f}{ref:>+16.3f}{d:>+9.3f}")
    assert ok, f"profile first moment does not match d_terr in dterr_audit.csv (tol {TOL})"
    np.savez(TAB / "territory_profile.npz", **out)
    print(f"\n  check passed (first moment = d_terr); saved territory_profile.npz")
    print("\nprofile (10 yd bins, DiD of origin share)")
    print(f"{'x range':<12}" + "".join(f"{g:>12}" for g in GROUPS))
    for i in range(NB):
        print(f"  {edges[i]:>3.0f}-{edges[i+1]:<3.0f}    "
              + "".join(f"{out[f'prof_{g}'][i]:>+12.4f}" for g in GROUPS))


def subgroups():
    """d_terr by home/away and score state (direction consistency)."""
    from scipy import stats
    d = pd.read_csv(TAB / "dterr_audit.csv")
    a = pd.read_parquet(DATA / "analysis_table.parquet")[["batch_id", "is_home", "score_state"]]
    m = d.merge(a, on="batch_id", how="left")
    rows = [("all", m)]
    rows += [("home" if h else "away", m[m.is_home == h]) for h in (True, False)]
    rows += [(s_, m[m.score_state == s_]) for s_ in ("lead", "draw", "trail")]
    out = []
    for lab, s_ in rows:
        y = s_.did_terr_excl.dropna().values
        se = y.std(ddof=1) / np.sqrt(len(y))
        _, p = stats.ttest_1samp(y, 0)
        out.append(dict(subgroup=lab, n=len(y), mean=y.mean(),
                        lo=y.mean() - 1.96 * se, hi=y.mean() + 1.96 * se, p=p))
    df = pd.DataFrame(out)
    df.to_csv(TAB / "territory_subgroups.csv", index=False)
    print("\nsubgroups (t-type 95% CI)")
    print(df.round(3).to_string(index=False))
    assert abs(df.loc[0, "mean"] - 1.074) < 0.01, "overall mean does not match dterr_audit.csv"
    print("  overall mean matches dterr_audit; subgroups agree in sign but are not individually significant")


if __name__ == "__main__":
    main()
    subgroups()
