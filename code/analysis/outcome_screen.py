"""
Screen of the eleven competitive outcomes against reorganization (Section 4.2,
Figure 3c). Post minus pre change per outcome regressed on volume-residualized
reorg with timing / score / home / strength / time-remaining controls; BH-FDR.
Inputs: data/{events_pass,events_goals,matches_meta,analysis_table}.parquet,
        data/substitution_batches.csv, data/windows/{window_specs,batch_flags}.parquet
Output: results/tables/consequence_mine.csv
Run:    python code/analysis/outcome_screen.py
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
TAB = ROOT / "results" / "tables"
NX, NY = 6, 4
PL, PW = 120, 80


def zone_share(w):
    zx = np.clip((w.x / (PL / NX)).astype(int), 0, NX - 1)
    zy = np.clip((w.y / (PW / NY)).astype(int), 0, NY - 1)
    ex = np.clip((w.end_x / (PL / NX)).astype(int), 0, NX - 1)
    ey = np.clip((w.end_y / (PW / NY)).astype(int), 0, NY - 1)
    Z = np.zeros((24, 24))
    np.add.at(Z, ((zx * NY + zy).values, (ex * NY + ey).values), 1.0)
    np.fill_diagonal(Z, 0)
    t = Z / Z.sum() if Z.sum() > 0 else Z
    return t.sum(0) + t.sum(1)


def feats(w):
    """Window features: completion rate, mean x (territory), progressive share, pass count."""
    if len(w) == 0:
        return dict(compl=np.nan, terr=np.nan, prog=np.nan, vol=0)
    compl = w.outcome.isna().mean()  # NaN outcome = completed
    cw = w[w.outcome.isna()]
    terr = cw.x.mean() if len(cw) else np.nan
    prog = ((cw.end_x - cw.x) > 10).mean() if len(cw) else np.nan
    return dict(compl=compl, terr=terr, prog=prog, vol=len(w))


def main():
    P = pd.read_parquet(DATA / "events_pass.parquet")
    # first-half length from all passes incl. set pieces (pipeline convention)
    t1 = P[P.period == 1].groupby("match_id").t_period_sec.max()
    P = P[~P.is_set_piece].copy()
    P["tc"] = np.where(P.period == 1, P.t_period_sec, P.match_id.map(t1) + P.t_period_sec)
    pi = {k: g for k, g in P.groupby(["match_id", "team_id"])}
    teams_in = {m: list(g.team_id.unique()) for m, g in P.groupby("match_id")}
    netpi = {k: g[g.outcome.isna()] for k, g in pi.items()}  # completed passes (reorg, territory)
    goals = pd.read_parquet(DATA / "events_goals.parquet")
    gd = {m: g for m, g in goals.groupby("match_id")}
    match_end = {m: g.tc.max() for m, g in P.groupby("match_id")}
    b = pd.read_csv(DATA / "substitution_batches.csv").set_index("batch_id")
    sp = pd.read_parquet(DATA / "windows" / "window_specs.parquet")
    stt = sp[(sp.kind == "time") & (sp.W_min == 15)].set_index(["batch_id", "side"])
    fl = pd.read_parquet(DATA / "windows" / "batch_flags.parquet")
    nontr = set(fl[(fl.W_min == 15) & ~fl.post_truncated].batch_id)
    at = pd.read_parquet(DATA / "analysis_table.parquet").set_index("batch_id")
    mm = pd.read_parquet(DATA / "matches_meta.parquet")
    pts, opp = {}, {}
    for _, r in mm.iterrows():
        hp = 3 if r.home_score > r.away_score else (1 if r.home_score == r.away_score else 0)
        pts[r.home_team_id] = pts.get(r.home_team_id, 0) + hp
        pts[r.away_team_id] = pts.get(r.away_team_id, 0) + (1 if hp == 1 else 3 - hp)
        opp[(r.match_id, r.home_team_id)] = r.away_team_id
        opp[(r.match_id, r.away_team_id)] = r.home_team_id
    result = {}  # (match,team) -> 1 win/0.5 draw/0 loss
    for _, r in mm.iterrows():
        wh = 1 if r.home_score > r.away_score else (0.5 if r.home_score == r.away_score else 0)
        result[(r.match_id, r.home_team_id)] = wh
        result[(r.match_id, r.away_team_id)] = 1 - wh if wh != 0.5 else 0.5

    def win(d, lo, hi, lo_open):
        if d is None:
            return d.iloc[0:0] if d is not None else None
        t = d.tc.values
        sel = ((t > lo) & (t <= hi)) if lo_open else ((t >= lo) & (t < hi))
        return d[sel]

    rows = []
    for bid in b.index:
        if bid not in nontr or (bid, "pre") not in stt.index or bid not in at.index:
            continue
        r = b.loc[bid]
        tid = r.team_id
        oid = opp.get((r.match_id, tid))
        netg = netpi.get((r.match_id, tid))
        allg = pi.get((r.match_id, tid))
        oallg = pi.get((r.match_id, oid))
        if netg is None or allg is None:
            continue
        pre, post = stt.loc[(bid, "pre")], stt.loc[(bid, "post")]
        wp = win(netg, pre.t_start, pre.t_end, False)
        wq = win(netg, post.t_start, post.t_end, True)
        if wp is None or wq is None or len(wp) < 10 or len(wq) < 10:
            continue
        reorg = np.linalg.norm(zone_share(wq) - zone_share(wp))
        # own-team deltas (all passes for completion / progressive share)
        ap = feats(win(allg, pre.t_start, pre.t_end, False)); aq = feats(win(allg, post.t_start, post.t_end, True))
        # opponent deltas
        if oallg is not None:
            op = feats(win(oallg, pre.t_start, pre.t_end, False)); oq = feats(win(oallg, post.t_start, post.t_end, True))
        else:
            op = oq = dict(compl=np.nan, terr=np.nan, prog=np.nan, vol=0)
        # possession share
        poss_pre = ap["vol"] / (ap["vol"] + op["vol"]) if (ap["vol"] + op["vol"]) else np.nan
        poss_post = aq["vol"] / (aq["vol"] + oq["vol"]) if (aq["vol"] + oq["vol"]) else np.nan
        # goals
        t0 = pre.t_end; gg = gd.get(r.match_id)
        def ng(lo):
            if gg is None:
                return 0
            af = gg[gg.t_cum > lo]
            return (af.team_id == tid).sum() - (af.team_id != tid).sum()
        nxt = np.nan
        if gg is not None and len(gg[gg.t_cum > t0]):
            nxt = int(gg[gg.t_cum > t0].sort_values("t_cum").iloc[0].team_id == tid)
        a = at.loc[bid]
        rows.append(dict(
            batch_id=bid, match_id=r.match_id, reorg=reorg, npass=min(len(wp), len(wq)),
            tnum={"strategic": 1, "regular": 2, "late": 3}[r.timing_group],
            leading=int(a.score_state == "lead"), trailing=int(a.score_state == "trail"),
            is_home=int(a.is_home), team_str=pts.get(tid), opp_str=pts.get(oid),
            t_remain=(match_end.get(r.match_id, t0) - t0) / 60,
            d_poss=poss_post - poss_pre, d_terr=aq["terr"] - ap["terr"],
            d_compl=aq["compl"] - ap["compl"], d_prog=aq["prog"] - ap["prog"],
            d_vol=aq["vol"] - ap["vol"],
            opp_d_compl=oq["compl"] - op["compl"], opp_d_terr=oq["terr"] - op["terr"],
            net_rem=ng(t0), net_after=ng(post.t_end), next_for=nxt, win=result.get((r.match_id, tid))))
    M = pd.DataFrame(rows)
    print(f"Sample: {len(M)} batches\n")

    def zc(s):
        return (s - s.mean()) / s.std()
    M["reorg_z"] = zc(M.reorg)
    npz = zc(M.npass)
    Xv = pd.DataFrame({"a": npz, "b": npz**2, "c": zc(1 / np.sqrt(M.npass)), "d": zc(np.log(M.npass)), "const": 1.0})
    M["reorg_resid"] = M.reorg_z - sm.OLS(M.reorg_z, Xv).fit().predict(Xv)
    for c in ["team_str", "opp_str", "t_remain"]:
        M[c + "_z"] = zc(M[c])

    OUT = {"d_poss": "possession share delta", "d_terr": "territory (mean x) delta", "d_compl": "pass completion delta",
           "d_prog": "forward-pass share delta", "d_vol": "pass volume delta", "opp_d_compl": "opponent completion delta (defensive disruption)",
           "opp_d_terr": "opponent territory delta", "net_rem": "net goals remaining", "net_after": "net goals after post-window",
           "next_for": "scores next goal", "win": "final result"}
    ctrl = "tnum + leading + trailing + is_home + team_str_z + opp_str_z + t_remain_z"
    res = []
    for o, lab in OUT.items():
        d = M.dropna(subset=[o, "reorg_resid"]).copy()
        if len(d) < 50:
            continue
        try:
            if o in ("next_for", "win") and set(d[o].dropna().unique()) <= {0, 1}:
                m = smf.logit(f"{o} ~ reorg_resid + {ctrl}", d).fit(disp=0)
            else:
                m = smf.mixedlm(f"{o} ~ reorg_resid + {ctrl}", d, groups=d.match_id).fit(reml=False)
            res.append(dict(outcome=o, label=lab, n=len(d),
                            beta=m.params.get("reorg_resid", np.nan), p=m.pvalues.get("reorg_resid", np.nan)))
        except Exception as e:
            res.append(dict(outcome=o, label=lab, n=len(d), beta=np.nan, p=np.nan))
    R = pd.DataFrame(res)
    R["q"] = multipletests(R.p.fillna(1), method="fdr_bh")[1]
    R = R.sort_values("p")
    R.to_csv(TAB / "consequence_mine.csv", index=False)

    print("=== reorg (residualized) -> outcomes, sorted by p, FDR ===")
    print(f"  {'outcome':22s} {'beta':>9s} {'p':>10s} {'q':>8s}")
    for _, x in R.iterrows():
        star = "*" if x.q < 0.05 else ""
        print(f"  {x.label:22s} {x.beta:+9.4f} {x.p:10.2e} {x.q:8.3f} {star}")
    hits = R[R.q < 0.05]
    print(f"\nHits (q<0.05): {len(hits)} -> {list(hits.label)}")
    print("Hits require cross-competition replication; an all-null screen supports no competitive consequence.")


if __name__ == "__main__":
    main()
