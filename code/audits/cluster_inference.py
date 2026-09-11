"""Cluster-level (team-match, team) sign-flip and bootstrap inference for the headline p-values.

A: group SNR with signs shared within team-match / team clusters.
B: d_terr cluster bootstrap (team-match) and wild cluster bootstrap (team).
C: consequence regression with crossed match+team effects, cluster-robust OLS, wild bootstrap.
Inputs:  data/{events_pass,events_goals,matches_meta,analysis_table,did_controls}.parquet, data/substitution_batches.csv, data/windows/*, results/tables/dterr_audit.csv
Outputs: results/tables/cluster_flips.csv
Run: python code/audits/cluster_inference.py
"""

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

warnings.filterwarnings("ignore")
import statsmodels.api as sm
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
TAB = ROOT / "results" / "tables"
NX, NY, PL, PW = 6, 4, 120, 80
SEED_FLIP = 20260613   # as in control_robustness.py
SEED_BOOT = 20260808   # cluster bootstrap
GROUPS = ["strategic", "regular", "late"]
N_BOOT_SNR, N_FLIP, B_BOOT = 800, 400, 2000

OUT_ROWS = []  # analysis,target,method,n,n_clusters,estimate,p,ci_lo,ci_hi,note


def add_row(analysis, target, method, n, n_clusters, estimate, p, ci_lo=np.nan, ci_hi=np.nan, note=""):
    OUT_ROWS.append(dict(analysis=analysis, target=target, method=method, n=n,
                         n_clusters=n_clusters, estimate=round(float(estimate), 4),
                         p=(round(float(p), 6) if np.isfinite(p) else np.nan),
                         ci_lo=(round(float(ci_lo), 4) if np.isfinite(ci_lo) else np.nan),
                         ci_hi=(round(float(ci_hi), 4) if np.isfinite(ci_hi) else np.nan),
                         note=note))


# shared data; t1 = max first-half t_period_sec over all passes, before dropping set pieces
def load_shared():
    passes = pd.read_parquet(DATA / "events_pass.parquet")
    t1 = passes[passes.period == 1].groupby("match_id").t_period_sec.max()
    pall = passes[~passes.is_set_piece].copy()
    pall["tc"] = np.where(pall.period == 1, pall.t_period_sec, pall.match_id.map(t1) + pall.t_period_sec)
    pi = {k: g for k, g in pall.groupby(["match_id", "team_id"])}
    netpi = {k: g[g.outcome.isna()] for k, g in pi.items()}   # completed passes only
    b = pd.read_csv(DATA / "substitution_batches.csv").set_index("batch_id")
    sp = pd.read_parquet(DATA / "windows" / "window_specs.parquet")
    st = sp[(sp.kind == "time") & (sp.W_min == 15)].set_index(["batch_id", "side"])
    fl = pd.read_parquet(DATA / "windows" / "batch_flags.parquet")
    nontr = set(fl[(fl.W_min == 15) & ~fl.post_truncated].batch_id)
    ctrl = pd.read_parquet(DATA / "did_controls.parquet")
    ctrl = ctrl[ctrl.control_type != "none"].set_index("batch_id")
    return pall, pi, netpi, b, st, nontr, ctrl


# Part A: group SNR, clustered sign flips (subset=all)
def zbin_vec(x, y):
    zx = np.clip((x / (PL / NX)).astype(int), 0, NX - 1)
    zy = np.clip((y / (PW / NY)).astype(int), 0, NY - 1)
    return zx * NY + zy


def zvec(w, ex_ids=None):
    if ex_ids is not None:
        w = w[~w.player_id.isin(ex_ids) & ~w.recipient_id.isin(ex_ids)]
    if len(w) == 0:
        return np.zeros(24)
    Z = np.zeros((24, 24))
    np.add.at(Z, (zbin_vec(w.x.values, w.y.values), zbin_vec(w.end_x.values, w.end_y.values)), 1.0)
    np.fill_diagonal(Z, 0)
    t = Z / Z.sum() if Z.sum() > 0 else Z
    return t.sum(0) + t.sum(1)


def part_a(netpi, b, st, nontr, ctrl):
    print("\n" + "=" * 70)
    print("Part A: group SNR with clustered sign flips (subset=all)")
    print("=" * 70)

    def winp(mid, tid, lo, hi, lo_open):
        g = netpi.get((mid, tid))
        if g is None:
            return None
        return g[(g.tc > lo) & (g.tc <= hi)] if lo_open else g[(g.tc >= lo) & (g.tc < hi)]

    # same recs construction and iteration order as control_robustness.py, plus cluster labels
    recs = {}
    for bid in nontr:
        if bid not in ctrl.index or (bid, "pre") not in st.index:
            continue
        r = b.loc[bid]
        pre, post = st.loc[(bid, "pre")], st.loc[(bid, "post")]
        wp = winp(r.match_id, r.team_id, pre.t_start, pre.t_end, False)
        wq = winp(r.match_id, r.team_id, post.t_start, post.t_end, True)
        c = ctrl.loc[bid]; tc = c.control_t_center
        wcp = winp(int(c.control_match_id), int(c.control_team_id), tc - 900, tc, False)
        wcq = winp(int(c.control_match_id), int(c.control_team_id), tc, tc + 900, True)
        if any(w is None or len(w) < 10 for w in [wp, wq, wcp, wcq]):
            continue
        swapped = ([int(x) for x in str(r.players_out_id).split("|")]
                   + [int(x) for x in str(r.players_in_id).split("|")])
        dz = (zvec(wq, swapped) - zvec(wp, swapped)) - (zvec(wcq) - zvec(wcp))
        recs[bid] = (r.timing_group, dz, f"{r.match_id}_{r.team_id}", int(r.team_id))

    arrs, tm_lab, team_lab = {}, {}, {}
    for g in GROUPS:
        vals = [(dz, tm, tt) for (gg, dz, tm, tt) in recs.values() if gg == g]
        arrs[g] = np.array([v[0] for v in vals])
        tm_lab[g] = np.array([v[1] for v in vals])
        team_lab[g] = np.array([v[2] for v in vals])
    ns = {g: len(arrs[g]) for g in GROUPS}
    N = min(ns.values())
    print(f"group n = {ns}, common N = {N}")

    # per-batch flips: exact headline replay (one rng stream, strategic -> regular -> late)
    expected = {"strategic": (1.052, 0.39), "regular": (1.561, 0.0025), "late": (1.594, 0.0075)}
    rng = np.random.default_rng(SEED_FLIP)
    obs_store = {}
    print("\n[replay check] per-batch flips vs control_robustness.csv (subset=all):")
    for g in GROUPS:
        arr = arrs[g]
        mags = [np.linalg.norm(arr[rng.integers(len(arr), size=N)].mean(0)) for _ in range(N_BOOT_SNR)]
        nf = []
        for _ in range(N_FLIP):
            idx = rng.integers(len(arr), size=N)
            sgn = rng.choice([1, -1], size=len(arr))
            nf.append(np.linalg.norm((arr[idx] * sgn[idx, None]).mean(0)))
        obs = np.mean(mags)
        snr_v, p_v = obs / np.mean(nf), 1 - (obs > np.array(nf)).mean()
        obs_store[g] = obs
        exp_s, exp_p = expected[g]
        ok = (round(snr_v, 3) == exp_s) and (abs(p_v - exp_p) < 1e-9)
        print(f"  {g:10s}: SNR={snr_v:.3f} (expected {exp_s})  p={p_v:.4f} (expected {exp_p})  "
              f"{'match' if ok else 'MISMATCH'}")
        if not ok:
            print("  WARNING: replay failed; clustered p-values may not be comparable with the headline p-values")
        add_row("A_snr_flip", g, "flip_batch(orig)", ns[g], ns[g], snr_v, p_v,
                note=f"replicates control_robustness: expected SNR={exp_s},p={exp_p}; N={N}; boot800/flip400; seed{SEED_FLIP}")

    # clustered flips: obs unchanged, only the sign unit of the null changes
    for scheme, labs in [("flip_team_match", tm_lab), ("flip_team", team_lab)]:
        rng = np.random.default_rng(SEED_FLIP)   # reset per scheme
        print(f"\n[{scheme}] 400 flips, signs shared within clusters, seed={SEED_FLIP}:")
        for g in GROUPS:
            arr = arrs[g]
            codes, uniq = pd.factorize(labs[g])
            nC = len(uniq)
            nf = []
            for _ in range(N_FLIP):
                idx = rng.integers(len(arr), size=N)
                sgn = rng.choice([1, -1], size=nC)[codes]
                nf.append(np.linalg.norm((arr[idx] * sgn[idx, None]).mean(0)))
            nf = np.array(nf)
            p_c = 1 - (obs_store[g] > nf).mean()
            snr_c = obs_store[g] / nf.mean()
            print(f"  {g:10s}: n={ns[g]:4d} clusters={nC:4d}  SNR_vs_cluster_null={snr_c:.3f}  p={p_c:.4f}"
                  f" {'*' if p_c < .05 else 'ns'}")
            add_row("A_snr_flip", g, scheme, ns[g], nC, snr_c, p_c,
                    note=f"obs identical to per-batch version; p resolution 1/400; seed{SEED_FLIP}")


# Part B: d_terr cluster-robust inference (dterr_audit.csv)
def cr1_se(x, codes, G):
    """CR1 cluster-robust SE of the mean (intercept-only model)."""
    e = x - x.mean()
    S = np.bincount(codes, weights=e, minlength=G)
    V = (G / (G - 1)) * (S ** 2).sum() / len(x) ** 2
    return np.sqrt(V)


def part_b(b):
    print("\n" + "=" * 70)
    print("Part B: territorial shift d_terr, cluster-robust inference (dterr_audit.csv)")
    print("=" * 70)
    M = pd.read_csv(TAB / "dterr_audit.csv")
    M = M.join(b[["match_id", "team_id"]], on="batch_id")
    assert M.match_id.notna().all() and M.team_id.notna().all(), "batch_id join to match/team failed"
    M["tm"] = M.match_id.astype(int).astype(str) + "_" + M.team_id.astype(int).astype(str)
    print(f"N = {len(M)}  team-match clusters = {M.tm.nunique()}  team clusters = {M.team_id.nunique()}")

    subsets = [("all", M)] + [(g, M[M.timing == g]) for g in GROUPS]
    for name, dsub in subsets:
        for col in ["did_terr_excl", "did_terr_incl"]:
            d = dsub.dropna(subset=[col])
            x = d[col].values
            n = len(x)
            target = f"{name}:{col.replace('did_terr_', '')}"

            # original per-batch t-test
            tt, pt = ttest_1samp(x, 0)
            sem = x.std(ddof=1) / np.sqrt(n)
            from scipy.stats import t as tdist
            hw = tdist.ppf(0.975, n - 1) * sem
            add_row("B_dterr", target, "ttest_batch(orig)", n, n, x.mean(), pt,
                    x.mean() - hw, x.mean() + hw, "per-batch one-sample t")

            # (a) team-match cluster bootstrap
            rng = np.random.default_rng(SEED_BOOT)
            codes_tm, uniq_tm = pd.factorize(d.tm)
            nC = len(uniq_tm)
            gidx = [np.where(codes_tm == c)[0] for c in range(nC)]
            boots = np.empty(B_BOOT)
            for i in range(B_BOOT):
                cs = rng.integers(nC, size=nC)
                boots[i] = x[np.concatenate([gidx[c] for c in cs])].mean()
            lo, hi = np.percentile(boots, [2.5, 97.5])
            p_boot = min(1.0, 2 * min(((boots <= 0).sum() + 1) / (B_BOOT + 1),
                                      ((boots >= 0).sum() + 1) / (B_BOOT + 1)))
            add_row("B_dterr", target, "boot_team_match", n, nC, x.mean(), p_boot, lo, hi,
                    f"cluster bootstrap B={B_BOOT} seed{SEED_BOOT}; percentile CI; p=2*min tail, +1 correction")

            # (b) wild cluster bootstrap by team (Rademacher, null imposed)
            rng = np.random.default_rng(SEED_BOOT)
            codes_t, uniq_t = pd.factorize(d.team_id)
            G = len(uniq_t)
            se_obs = cr1_se(x, codes_t, G)
            t_obs = x.mean() / se_obs
            S0 = np.bincount(codes_t, weights=x, minlength=G)     # per-team sums, null residuals = x
            ng = np.bincount(codes_t, minlength=G).astype(float)
            signs = rng.choice([1, -1], size=(B_BOOT, G))
            mean_star = signs @ S0 / n
            Sg_star = signs * S0[None, :] - ng[None, :] * mean_star[:, None]
            se_star = np.sqrt((G / (G - 1)) * (Sg_star ** 2).sum(1)) / n
            t_star = mean_star / se_star
            p_wild = ((np.abs(t_star) >= abs(t_obs)).sum() + 1) / (B_BOOT + 1)
            q95 = np.quantile(np.abs(t_star), 0.95)   # symmetric bootstrap-t CI
            add_row("B_dterr", target, "wild_boot_team", n, G, x.mean(), p_wild,
                    x.mean() - q95 * se_obs, x.mean() + q95 * se_obs,
                    f"wild cluster (Rademacher, H0 imposed) B={B_BOOT} seed{SEED_BOOT}; CR1; 95% symmetric bootstrap-t CI")
            print(f"  {target:16s}: mean={x.mean():+.3f}  p_batch={pt:.4f}  "
                  f"p_bootTM={p_boot:.4f}(C={nC})  p_wildTeam={p_wild:.4f}(G={G})")


# Part C: consequence regression of d_terr, team-level extensions
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
    if len(w) == 0:
        return dict(compl=np.nan, terr=np.nan, prog=np.nan, vol=0)
    compl = w.outcome.isna().mean()
    cw = w[w.outcome.isna()]
    terr = cw.x.mean() if len(cw) else np.nan
    prog = ((cw.end_x - cw.x) > 10).mean() if len(cw) else np.nan
    return dict(compl=compl, terr=terr, prog=prog, vol=len(w))


def part_c(pall, pi, netpi, b, st, nontr):
    print("\n" + "=" * 70)
    print("Part C: d_terr consequence regression, crossed match/team + team-cluster-robust")
    print("=" * 70)
    goals = pd.read_parquet(DATA / "events_goals.parquet")
    gd = {m: g for m, g in goals.groupby("match_id")}
    match_end = {m: g.tc.max() for m, g in pall.groupby("match_id")}
    at = pd.read_parquet(DATA / "analysis_table.parquet").set_index("batch_id")
    mm = pd.read_parquet(DATA / "matches_meta.parquet")
    opp = {}
    pts = {}
    for _, r in mm.iterrows():
        hp = 3 if r.home_score > r.away_score else (1 if r.home_score == r.away_score else 0)
        pts[r.home_team_id] = pts.get(r.home_team_id, 0) + hp
        pts[r.away_team_id] = pts.get(r.away_team_id, 0) + (1 if hp == 1 else 3 - hp)
        opp[(r.match_id, r.home_team_id)] = r.away_team_id
        opp[(r.match_id, r.away_team_id)] = r.home_team_id

    def win(dd, lo, hi, lo_open):
        t = dd.tc.values
        sel = ((t > lo) & (t <= hi)) if lo_open else ((t >= lo) & (t < hi))
        return dd[sel]

    rows = []
    for bid in b.index:
        if bid not in nontr or (bid, "pre") not in st.index or bid not in at.index:
            continue
        r = b.loc[bid]
        tid = r.team_id
        oid = opp.get((r.match_id, tid))
        netg = netpi.get((r.match_id, tid))
        allg = pi.get((r.match_id, tid))
        oallg = pi.get((r.match_id, oid))
        if netg is None or allg is None:
            continue
        pre, post = st.loc[(bid, "pre")], st.loc[(bid, "post")]
        wp = win(netg, pre.t_start, pre.t_end, False)
        wq = win(netg, post.t_start, post.t_end, True)
        if len(wp) < 10 or len(wq) < 10:
            continue
        reorg = np.linalg.norm(zone_share(wq) - zone_share(wp))
        ap = feats(win(allg, pre.t_start, pre.t_end, False))
        aq = feats(win(allg, post.t_start, post.t_end, True))
        t0 = pre.t_end
        a = at.loc[bid]
        rows.append(dict(
            batch_id=bid, match_id=r.match_id, team_id=tid, reorg=reorg,
            npass=min(len(wp), len(wq)),
            tnum={"strategic": 1, "regular": 2, "late": 3}[r.timing_group],
            leading=int(a.score_state == "lead"), trailing=int(a.score_state == "trail"),
            is_home=int(a.is_home), team_str=pts.get(tid), opp_str=pts.get(oid),
            t_remain=(match_end.get(r.match_id, t0) - t0) / 60,
            d_terr=aq["terr"] - ap["terr"]))
    M = pd.DataFrame(rows)
    print(f"replayed sample N = {len(M)} (expected 1490)")

    def zc(s):
        return (s - s.mean()) / s.std()
    M["reorg_z"] = zc(M.reorg)
    npz = zc(M.npass)
    Xv = pd.DataFrame({"a": npz, "b": npz ** 2, "c": zc(1 / np.sqrt(M.npass)),
                       "d": zc(np.log(M.npass)), "const": 1.0})
    M["reorg_resid"] = M.reorg_z - sm.OLS(M.reorg_z, Xv).fit().predict(Xv)
    for c in ["team_str", "opp_str", "t_remain"]:
        M[c + "_z"] = zc(M[c])

    CTRLSTR = "tnum + leading + trailing + is_home + team_str_z + opp_str_z + t_remain_z"
    f = f"d_terr ~ reorg_resid + {CTRLSTR}"
    d = M.dropna(subset=["d_terr", "reorg_resid"]).copy()
    n = len(d)

    # baseline replay: mixedlm groups=match_id
    m0 = smf.mixedlm(f, d, groups=d.match_id).fit(reml=False)
    b0, p0 = m0.params["reorg_resid"], m0.pvalues["reorg_resid"]
    ok = (abs(b0 - 1.5909) < 5e-4) and (abs(p0 - 5.77e-05) / 5.77e-05 < 0.02)
    print(f"[replay check] mixedlm(match): beta={b0:.4f} (expected 1.5909)  p={p0:.3e} (expected 5.77e-05)  "
          f"{'match' if ok else 'MISMATCH'}  converged={m0.converged}")
    add_row("C_consequence", "d_terr", "mixedlm_match(orig)", n, d.match_id.nunique(), b0, p0,
            note=f"replicates consequence_mine.csv; converged={m0.converged}; q_BH ~ p x 11 (11 outcomes, smallest p)")

    # (a1) crossed match + team random effects (one group, two variance components)
    try:
        t0c = time.time()
        d["allone"] = 1
        mx = smf.mixedlm(f, d, groups=d.allone, re_formula="0",
                         vc_formula={"match": "0 + C(match_id)", "team": "0 + C(team_id)"}
                         ).fit(reml=False)
        bx, px = mx.params["reorg_resid"], mx.pvalues["reorg_resid"]
        vc_names = list(mx.model.exog_vc.names) if hasattr(mx.model, "exog_vc") else ["vc1", "vc2"]
        vc_note = "; ".join(f"{k}={v:.3f}" for k, v in zip(vc_names, mx.vcomp))
        # degenerate if fitted team variance far exceeds the raw team-mean dispersion
        raw_team_sd = d.groupby("team_id").d_terr.mean().std()
        vc_team = dict(zip(vc_names, mx.vcomp)).get("team", np.nan)
        degen = np.isfinite(vc_team) and vc_team > 10 * max(raw_team_sd ** 2, 1e-9)
        diag = (f"raw team-mean SD={raw_team_sd:.2f} (var ~{raw_team_sd**2:.1f}, incl. sampling noise)"
                + ("; team vcomp contradicts this -> degenerate, do not cite; use team-only / OLS-CR / wild boot" if degen else ""))
        print(f"mixedlm(crossed match+team): beta={bx:.4f}  p={px:.3e}  [{time.time()-t0c:.0f}s]  "
              f"{vc_note}  converged={mx.converged}  {'WARNING ' + diag if degen else diag}")
        add_row("C_consequence", "d_terr", "mixedlm_crossed_match_team", n,
                d.match_id.nunique() + d.team_id.nunique(), bx, px,
                note=f"single group + vc(match,team); {vc_note}; converged={mx.converged}; {diag}")
    except Exception as e:
        print(f"crossed random effects failed: {e}")
        add_row("C_consequence", "d_terr", "mixedlm_crossed_match_team", n, np.nan, np.nan, np.nan,
                note=f"failed: {e}")

    # (a2) team-only random effects
    mt = smf.mixedlm(f, d, groups=d.team_id).fit(reml=False)
    bt, pt_ = mt.params["reorg_resid"], mt.pvalues["reorg_resid"]
    print(f"mixedlm(team only):     beta={bt:.4f}  p={pt_:.3e}  converged={mt.converged}")
    add_row("C_consequence", "d_terr", "mixedlm_team", n, d.team_id.nunique(), bt, pt_,
            note=f"converged={mt.converged}")

    # (b) OLS + cluster-robust SE
    for lab, gvar in [("ols_cluster_team", d.team_id), ("ols_cluster_match", d.match_id)]:
        mo = smf.ols(f, d).fit(cov_type="cluster", cov_kwds={"groups": gvar}, use_t=True)
        bo, po = mo.params["reorg_resid"], mo.pvalues["reorg_resid"]
        ci = mo.conf_int().loc["reorg_resid"]
        print(f"{lab:22s}: beta={bo:.4f}  p={po:.3e}  CI=[{ci[0]:.3f},{ci[1]:.3f}]  G={gvar.nunique()}")
        add_row("C_consequence", "d_terr", lab, n, gvar.nunique(), bo, po, ci[0], ci[1],
                note="cluster-robust SE, use_t, df ~ G-1")

    # (b+) wild cluster bootstrap by team for the reorg_resid coefficient (G=20)
    import patsy
    ymat, X = patsy.dmatrices(f, d, return_type="dataframe")
    yv = ymat.values.ravel(); Xv2 = X.values
    j = list(X.columns).index("reorg_resid")
    nn, k = Xv2.shape
    codes_t, uniq_t = pd.factorize(d.team_id)
    G = len(uniq_t)
    corr = (G / (G - 1)) * ((nn - 1) / (nn - k))          # statsmodels use_correction
    XtXi = np.linalg.inv(Xv2.T @ Xv2)

    def cr_t(yy):
        beta = XtXi @ (Xv2.T @ yy)
        e = yy - Xv2 @ beta
        Xe = Xv2 * e[:, None]
        S = np.zeros((G, k))
        np.add.at(S, codes_t, Xe)
        V = corr * XtXi @ (S.T @ S) @ XtXi
        return beta[j] / np.sqrt(V[j, j])

    t_obs = cr_t(yv)
    Xr = np.delete(Xv2, j, axis=1)                          # restricted model, H0 imposed
    br = np.linalg.lstsq(Xr, yv, rcond=None)[0]
    yhat_r = Xr @ br
    e_r = yv - yhat_r
    rng = np.random.default_rng(SEED_BOOT)
    t_star = np.empty(B_BOOT)
    for i in range(B_BOOT):
        s = rng.choice([1, -1], size=G)
        t_star[i] = cr_t(yhat_r + e_r * s[codes_t])
    p_wild = ((np.abs(t_star) >= abs(t_obs)).sum() + 1) / (B_BOOT + 1)
    mo_t = smf.ols(f, d).fit(cov_type="cluster", cov_kwds={"groups": d.team_id}, use_t=True)
    se_j = mo_t.bse["reorg_resid"]
    q95 = np.quantile(np.abs(t_star), 0.95)   # symmetric bootstrap-t CI
    bo = mo_t.params["reorg_resid"]
    print(f"wild_boot_team(beta)    : beta={bo:.4f}  t_obs={t_obs:.3f}  p={p_wild:.4f}  "
          f"CI=[{bo - q95 * se_j:.3f},{bo + q95 * se_j:.3f}]  G={G}")
    add_row("C_consequence", "d_terr", "ols_wild_boot_team", n, G, bo, p_wild,
            bo - q95 * se_j, bo + q95 * se_j,
            note=f"wild cluster (Rademacher, H0 imposed) B={B_BOOT} seed{SEED_BOOT}; CR1; 95% symmetric bootstrap-t CI")


def main():
    t_start = time.time()
    pall, pi, netpi, b, st, nontr, ctrl = load_shared()
    part_a(netpi, b, st, nontr, ctrl)
    part_b(b)
    part_c(pall, pi, netpi, b, st, nontr)
    out = pd.DataFrame(OUT_ROWS)
    out.to_csv(TAB / "cluster_flips.csv", index=False)
    print(f"\nsaved {TAB / 'cluster_flips.csv'}  ({len(out)} rows)  total {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
