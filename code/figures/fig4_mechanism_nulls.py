"""
Dependence-aware Monte-Carlo and permutation nulls for the mechanism screen (Figure 4c).

Rebuilds the 36-factor design matrix of mechanism_screen.py, takes the empirical correlation of
the main-effect and interaction statistics, and samples MVN(0, R) for order-statistic envelopes
and Monte-Carlo p-values. Optional mixedlm Freedman-Lane permutation check with --perm B (slow).
Inputs: data/{events_pass,events_goals,analysis_table,matches_meta}.parquet, data/substitution_batches.csv,
        data/windows/*.parquet, results/tables/mechanism_mine.csv
Outputs: results/tables/mechanism_null_mc.csv, mechanism_null_global.csv, [mechanism_null_perm.csv]
Run: python code/figures/fig4_mechanism_nulls.py [--perm 50]
"""

import importlib.util
import sys
import warnings
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]  # <repo>/code

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "data"
TAB = ROOT / "results" / "tables"
NSIM = 100_000
SEED = 20260726


def load_25():
    spec = importlib.util.spec_from_file_location("m25", CODE_ROOT / "analysis" / "mechanism_screen.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_design(m25):
    """Rebuild the mechanism_screen.py design matrix (same windows, >= 10 passes, 36 factors)."""
    passes = pd.read_parquet(DATA / "events_pass.parquet")
    net = passes[passes.outcome.isna() & ~passes.is_set_piece].copy()
    t1 = passes[passes.period == 1].groupby("match_id").t_period_sec.max()
    net["tc"] = np.where(net.period == 1, net.t_period_sec,
                         net.match_id.map(t1) + net.t_period_sec)
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
        reorg = np.linalg.norm(m25.zone_share(wq) - m25.zone_share(wp))
        out_ids = [int(x) for x in str(r.players_out_id).split("|")]
        cdeg, cbtw, cpr = m25.out_centrality(wp, out_ids)
        gg = gd.get(r.match_id)
        margin = 0
        if gg is not None:
            bef = gg[gg.t_cum < pre.t_end]
            margin = (bef.team_id == r.team_id).sum() - (bef.team_id != r.team_id).sum()
        a = at.loc[bid]
        rows.append(dict(
            batch_id=bid, match_id=r.match_id, reorg=reorg, npass=min(len(wp), len(wq)),
            tnum={"strategic": 1, "regular": 2, "late": 3}[r.timing_group],
            out_deg=cdeg, out_btw=cbtw, out_pr=cpr, margin=margin, abs_margin=abs(margin),
            n_subs=r.n_subs, is_home=int(a.is_home), batch_seq=a.batch_seq,
            opp_sub=int(a.opp_sub_in_window), minute=r.minute_first,
            team_str=pts.get(r.team_id), opp_str=pts.get(opp.get((r.match_id, r.team_id))),
            pre_dens_P=a.pre_density_P_t15, pre_clus_P=a.pre_clustering_P_t15,
            pre_btw_P=a.pre_betweenness_P_t15, pre_lam_P=a.pre_lambda2_P_t15,
            pre_eff_P=a.pre_efficiency_P_t15,
            pre_dens_Z=a.pre_density_Z_t15, pre_clus_Z=a.pre_clustering_Z_t15,
            pre_btw_Z=a.pre_betweenness_Z_t15, pre_lam_Z=a.pre_lambda2_Z_t15,
            pre_eff_Z=a.pre_efficiency_Z_t15,
            pre_vers_P=a.pre_vers_P_w10_Z_t15, pre_vers_Z=a.pre_vers_Z_w10_Z_t15,
            preA_att_P=a.preA_attack_P_t15, preA_rnd_P=a.preA_random_P_t15,
            preA_att_Z=a.preA_attack_Z_t15, preA_rnd_Z=a.preA_random_Z_t15,
            ent_mean=a.ent_mean_6x4_t15, ent_max=a.ent_max_6x4_t15,
            v_mean=a.v_mean_t15, v_max=a.v_max_t15))
    M = pd.DataFrame(rows)
    M["str_diff"] = M.team_str - M.opp_str
    M["leading"] = (M.margin > 0).astype(int)
    M["trailing"] = (M.margin < 0).astype(int)
    M["t_remain"] = (5400 - M.minute * 60) / 60
    return M


def zc(s):
    s = np.asarray(s, float)
    return (s - np.nanmean(s)) / np.nanstd(s, ddof=1)


def resid_on(Y, X):
    """Residualize each column of Y on X (least squares)."""
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return Y - X @ beta


def main(perm_B=0):
    m25 = load_25()
    frozen = pd.read_csv(TAB / "mechanism_mine.csv")
    PRED = list(frozen.predictor)
    print(f"rebuilding design matrix (36 factors)...")
    M = build_design(m25)
    print(f"  batches {len(M)}")

    # nonlinear pass-volume residualization, as in mechanism_screen.py
    npz = zc(M.npass)
    Xv = np.column_stack([np.ones(len(M)), npz, npz ** 2,
                          zc(1 / np.sqrt(M.npass)), zc(np.log(M.npass))])
    y = zc(M.reorg)
    y = y - Xv @ np.linalg.lstsq(Xv, y, rcond=None)[0]          # reorg_resid

    P = np.column_stack([zc(M[p].astype(float)) for p in PRED])  # n x 36
    ok = np.isfinite(P).all(1) & np.isfinite(y)
    P, y, tnum = P[ok], y[ok], M.tnum.values[ok].astype(float)
    n = len(y)
    one = np.ones(n)

    # main-effect family: residualize factors on [1, tnum]
    X0 = np.column_stack([one, tnum])
    Pm = resid_on(P, X0)
    ym = y - X0 @ np.linalg.lstsq(X0, y, rcond=None)[0]
    R_main = np.corrcoef(Pm, rowvar=False)

    # interaction family: residualize pz*tnum on [1, pz, tnum]
    I = np.empty_like(P)
    for j in range(P.shape[1]):
        Xj = np.column_stack([one, P[:, j], tnum])
        I[:, j] = P[:, j] * tnum - Xj @ np.linalg.lstsq(Xj, P[:, j] * tnum, rcond=None)[0]
    R_int = np.corrcoef(I, rowvar=False)

    # check: OLS |z| must rank-correlate with the frozen mixedlm |z|
    def ols_z(Xres, yres, df):
        r = (Xres * yres[:, None]).sum(0) / np.sqrt((Xres ** 2).sum(0) * (yres ** 2).sum())
        return r * np.sqrt(df) / np.sqrt(np.clip(1 - r ** 2, 1e-12, None))
    z_obs_main = ols_z(Pm, ym, n - 3)
    z_frozen = norm.isf(frozen.p_main.values / 2)
    rho = spearmanr(np.abs(z_obs_main), z_frozen).statistic
    assert rho >= 0.95, f"design matrix mismatch with mechanism_mine.csv (Spearman {rho:.3f} < 0.95)"
    print(f"  check passed: Spearman(OLS |z|, frozen mixedlm |z|) = {rho:.3f}")
    print(f"  predictor pairs with |r|>0.5: {int((np.abs(R_main[np.triu_indices(36,1)])>0.5).sum())}"
          f" (max {np.abs(R_main[np.triu_indices(36,1)]).max():.2f})")

    # sample MVN(0, R)
    rng = np.random.default_rng(SEED)
    out_env, out_glob = [], []
    for fam, R, p_obs in [("main", R_main, frozen.p_main.values),
                          ("interaction", R_int, frozen.p_inter.values)]:
        Rp = (R + R.T) / 2
        w, V = np.linalg.eigh(Rp)
        A = V @ np.diag(np.sqrt(np.clip(w, 0, None)))            # R = A A^T
        Z = np.abs(rng.standard_normal((NSIM, 36)) @ A.T)
        Zs = np.sort(Z, axis=1)
        k = np.arange(1, 37)
        x_exp = norm.ppf((1 + (k - 0.5) / 36) / 2)
        lo, med, hi = np.percentile(Zs, [2.5, 50, 97.5], axis=0)
        for i in range(36):
            out_env.append(dict(family=fam, k=int(k[i]), x_expected=x_exp[i],
                                lo=lo[i], med=med[i], hi=hi[i]))
        zo = np.sort(norm.isf(p_obs / 2))
        for stat, f in [("mean_abs_z", np.mean), ("max_abs_z", np.max)]:
            nullstat = f(Zs, axis=1)
            obs = f(zo)
            out_glob.append(dict(family=fam, stat=stat, observed=obs,
                                 mc_p=float((nullstat >= obs).mean()), n_sim=NSIM))
        n_out = int((zo > hi).sum())
        out_glob.append(dict(family=fam, stat="n_outside_pointwise_95",
                             observed=n_out, mc_p=np.nan, n_sim=NSIM))
        print(f"  [{fam}] above the pointwise 95% envelope: {n_out}/36 | "
              f"mean|z| MC p = {out_glob[-3]['mc_p']:.3f} | max|z| MC p = {out_glob[-2]['mc_p']:.3f}")

    pd.DataFrame(out_env).to_csv(TAB / "mechanism_null_mc.csv", index=False)
    pd.DataFrame(out_glob).to_csv(TAB / "mechanism_null_global.csv", index=False)
    print("saved mechanism_null_mc.csv + mechanism_null_global.csv")

    if perm_B:
        permutation_check(M.match_id.values[ok], P, y, tnum, PRED, perm_B)


def permutation_check(mid, P, y, tnum, PRED, B):
    """mixedlm + Freedman-Lane permutation for the interaction family. Slow; only with --perm."""
    import statsmodels.formula.api as smf
    rng = np.random.default_rng(SEED)
    n = len(y)
    zmax = []
    for bnum in range(B):
        yp = y[rng.permutation(n)]
        zs = []
        for j in range(P.shape[1]):
            d = pd.DataFrame(dict(y=yp, pz=P[:, j], tnum=tnum, match_id=mid))
            r = smf.mixedlm("y ~ pz * tnum", d, groups=d.match_id).fit(reml=False)
            zs.append(abs(r.params.get("pz:tnum", np.nan) / r.bse.get("pz:tnum", np.nan)))
        zmax.append(np.nanmax(zs))
        print(f"    perm {bnum + 1}/{B}: max|z| = {zmax[-1]:.2f}", flush=True)
    pd.DataFrame(dict(perm=np.arange(1, B + 1), max_abs_z=zmax)).to_csv(
        TAB / "mechanism_null_perm.csv", index=False)
    print(f"  permutation (mixedlm, B={B}): 95th percentile of max|z| = {np.percentile(zmax, 95):.2f}")


if __name__ == "__main__":
    B = int(sys.argv[sys.argv.index("--perm") + 1]) if "--perm" in sys.argv else 0
    main(B)
