"""
Screen of 36 candidate predictors of reorganization magnitude with volume
residualization and FDR control (Section 4.3, Figure 4). Per-match random
intercept mixed models; reorg residualized on flexible npass; BH-FDR.
Inputs: data/{events_pass,events_goals,matches_meta,analysis_table}.parquet,
        data/substitution_batches.csv, data/windows/{window_specs,batch_flags}.parquet
Output: results/tables/mechanism_mine.csv
Run:    python code/analysis/mechanism_screen.py
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

warnings.filterwarnings("ignore")
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


def out_centrality(w, out_ids):
    """Pre-window centrality of the substituted-off players (degree share, betweenness, PageRank; mean over players)."""
    g = nx.DiGraph()
    for p, rcv in zip(w.player_id, w.recipient_id):
        if pd.notna(p) and pd.notna(rcv):
            g.add_edge(int(p), int(rcv), weight=g.get_edge_data(int(p), int(rcv), {}).get("weight", 0) + 1)
    if g.number_of_nodes() < 3:
        return np.nan, np.nan, np.nan
    deg = dict(g.degree(weight="weight"))
    tot = sum(deg.values()) or 1
    try:
        btw = nx.betweenness_centrality(g, weight=lambda u, v, d: 1.0 / d["weight"])
    except Exception:
        btw = {n: 0 for n in g.nodes}
    try:
        pr = nx.pagerank(g, weight="weight")
    except Exception:
        pr = {n: 0 for n in g.nodes}
    present = [p for p in out_ids if p in g]
    if not present:
        return np.nan, np.nan, np.nan
    return (np.mean([deg[p] / tot for p in present]),
            np.mean([btw[p] for p in present]),
            np.mean([pr[p] for p in present]))


def main():
    passes = pd.read_parquet(DATA / "events_pass.parquet")
    net = passes[passes.outcome.isna() & ~passes.is_set_piece].copy()
    t1 = passes[passes.period == 1].groupby("match_id").t_period_sec.max()
    net["tc"] = np.where(net.period == 1, net.t_period_sec, net.match_id.map(t1) + net.t_period_sec)
    ni = {k: g for k, g in net.groupby(["match_id", "team_id"])}
    goals = pd.read_parquet(DATA / "events_goals.parquet")
    gd = {m: g for m, g in goals.groupby("match_id")}
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

    rows = []
    for bid in b.index:
        if bid not in nontr or (bid, "pre") not in stt.index or bid not in at.index:
            continue
        r = b.loc[bid]
        g = ni.get((r.match_id, r.team_id))
        if g is None:
            continue
        pre, post = stt.loc[(bid, "pre")], stt.loc[(bid, "post")]
        wp = g[(g.tc >= pre.t_start) & (g.tc < pre.t_end)]
        wq = g[(g.tc > post.t_start) & (g.tc <= post.t_end)]
        if len(wp) < 10 or len(wq) < 10:
            continue
        reorg = np.linalg.norm(zone_share(wq) - zone_share(wp))
        out_ids = [int(x) for x in str(r.players_out_id).split("|")]
        cdeg, cbtw, cpr = out_centrality(wp, out_ids)
        gg = gd.get(r.match_id)
        margin = 0
        if gg is not None:
            bef = gg[gg.t_cum < pre.t_end]
            margin = (bef.team_id == r.team_id).sum() - (bef.team_id != r.team_id).sum()
        a = at.loc[bid]
        rows.append(dict(
            batch_id=bid, match_id=r.match_id, reorg=reorg, npass=min(len(wp), len(wq)),
            tnum={"strategic": 1, "regular": 2, "late": 3}[r.timing_group],
            out_deg=cdeg, out_btw=cbtw, out_pr=cpr,
            margin=margin, abs_margin=abs(margin),
            n_subs=r.n_subs, is_home=int(a.is_home), batch_seq=a.batch_seq,
            opp_sub=int(a.opp_sub_in_window), minute=r.minute_first,
            team_str=pts.get(r.team_id), opp_str=pts.get(opp.get((r.match_id, r.team_id))),
            # pre-window two-layer features from analysis_table
            pre_dens_P=a.pre_density_P_t15, pre_clus_P=a.pre_clustering_P_t15,
            pre_btw_P=a.pre_betweenness_P_t15, pre_lam_P=a.pre_lambda2_P_t15, pre_eff_P=a.pre_efficiency_P_t15,
            pre_dens_Z=a.pre_density_Z_t15, pre_clus_Z=a.pre_clustering_Z_t15,
            pre_btw_Z=a.pre_betweenness_Z_t15, pre_lam_Z=a.pre_lambda2_Z_t15, pre_eff_Z=a.pre_efficiency_Z_t15,
            pre_vers_P=a.pre_vers_P_w10_Z_t15, pre_vers_Z=a.pre_vers_Z_w10_Z_t15,
            preA_att_P=a.preA_attack_P_t15, preA_rnd_P=a.preA_random_P_t15,
            preA_att_Z=a.preA_attack_Z_t15, preA_rnd_Z=a.preA_random_Z_t15,
            ent_mean=a.ent_mean_6x4_t15, ent_max=a.ent_max_6x4_t15,
            v_mean=a.v_mean_t15, v_max=a.v_max_t15))
    M = pd.DataFrame(rows)
    M["str_diff"] = M.team_str - M.opp_str
    M["leading"] = (M.margin > 0).astype(int)
    M["trailing"] = (M.margin < 0).astype(int)
    M["t_remain"] = (5400 - M.minute * 60) / 60  # rough minutes remaining
    print(f"Sample: {len(M)} batches\n")

    PRED = ["out_deg", "out_btw", "out_pr", "margin", "abs_margin", "leading", "trailing",
            "n_subs", "is_home", "batch_seq", "opp_sub", "minute", "t_remain",
            "team_str", "opp_str", "str_diff",
            "pre_dens_P", "pre_clus_P", "pre_btw_P", "pre_lam_P", "pre_eff_P",
            "pre_dens_Z", "pre_clus_Z", "pre_btw_Z", "pre_lam_Z", "pre_eff_Z",
            "pre_vers_P", "pre_vers_Z", "preA_att_P", "preA_rnd_P", "preA_att_Z", "preA_rnd_Z",
            "ent_mean", "ent_max", "v_mean", "v_max"]

    def zc(s):
        return (s - s.mean()) / s.std()
    M["reorg_z"] = zc(M.reorg)
    # residualize reorg on [npass, npass^2, 1/sqrt(npass), log npass]
    npz = zc(M.npass)
    Xv = pd.DataFrame({"a": npz, "b": npz**2, "c": zc(1/np.sqrt(M.npass)), "d": zc(np.log(M.npass))})
    Xv["const"] = 1.0
    import statsmodels.api as sm
    fitv = sm.OLS(M.reorg_z, Xv).fit()
    M["reorg_resid"] = M.reorg_z - fitv.predict(Xv)  # reorg net of volume
    print(f"[volume control] reorg ~ flex(npass) R2={fitv.rsquared:.3f}; screening uses reorg_resid\n")

    res = []
    for p in PRED:
        d = M.dropna(subset=[p, "reorg_resid"]).copy()
        if len(d) < 50 or d[p].std() == 0:
            continue
        d["pz"] = zc(d[p])
        # (M) main effect, volume-residualized outcome, timing controlled
        try:
            m = smf.mixedlm("reorg_resid ~ pz + tnum", d, groups=d.match_id).fit(reml=False)
            bm, pm = m.params.get("pz", np.nan), m.pvalues.get("pz", np.nan)
        except Exception:
            bm, pm = np.nan, np.nan
        # naive spec, linear npass control
        try:
            mn = smf.mixedlm("reorg_z ~ pz + npass_z + tnum", d.assign(npass_z=zc(d.npass)),
                             groups=d.match_id).fit(reml=False)
            pm_naive = mn.pvalues.get("pz", np.nan)
        except Exception:
            pm_naive = np.nan
        # (G) predictor x timing interaction
        try:
            mg = smf.mixedlm("reorg_resid ~ pz * tnum", d, groups=d.match_id).fit(reml=False)
            bg, pg = mg.params.get("pz:tnum", np.nan), mg.pvalues.get("pz:tnum", np.nan)
        except Exception:
            bg, pg = np.nan, np.nan
        res.append(dict(predictor=p, n=len(d), beta_main=bm, p_main=pm, p_main_naive=pm_naive,
                        beta_inter=bg, p_inter=pg))
    R = pd.DataFrame(res)
    R["q_main"] = multipletests(R.p_main.fillna(1), method="fdr_bh")[1]
    R["q_naive"] = multipletests(R.p_main_naive.fillna(1), method="fdr_bh")[1]
    R["q_inter"] = multipletests(R.p_inter.fillna(1), method="fdr_bh")[1]
    R = R.sort_values("p_main")
    R.to_csv(TAB / "mechanism_mine.csv", index=False)

    print("=== (M) main effects on reorg, naive (linear npass) vs robust (nonlinear residual) ===")
    print(f"  {'predictor':12s} {'b_rob':>7s} {'q_naive':>8s} {'q_robust':>9s}  verdict")
    surv, died = [], []
    for _, x in R.iterrows():
        if x.q_naive < 0.05 or x.q_main < 0.05:
            if x.q_main < 0.05:
                tag = "survives (candidate)"; surv.append(x.predictor)
            else:
                tag = "volume artifact (lost after residualization)"; died.append(x.predictor)
            print(f"  {x.predictor:12s} {x.beta_main:+7.3f} {x.q_naive:8.3f} {x.q_main:9.3f}  {tag}")
    print(f"\n  Surviving (q<0.05 after volume residualization): {len(surv)} -> {surv}")
    print(f"  Lost to volume confounding: {len(died)}")

    print("\n=== (G) timing moderation, predictor x timing (residualized), FDR q<0.05 ===")
    hg = R[R.q_inter < 0.05].sort_values("p_inter")
    if len(hg):
        for _, x in hg.iterrows():
            print(f"  {x.predictor:12s}: beta_inter={x.beta_inter:+.3f} q={x.q_inter:.3f} *")
    else:
        print("  no FDR hits")
    print("\nSurviving candidates require cross-competition replication before interpretation.")


if __name__ == "__main__":
    main()
