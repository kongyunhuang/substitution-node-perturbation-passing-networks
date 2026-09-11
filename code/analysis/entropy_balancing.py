"""
Ridge-relaxed entropy balancing of control windows and the weighted re-estimates
(Section 3.4, Appendix A, Table A.1). Per timing group, control weights match 12
pre-window covariates; ridge relaxed until all |SMD| < 0.1 with ESS >= 25%.
Inputs: data/metrics/{metrics_player,metrics_zone}.parquet, data/events_pass.parquet,
        data/substitution_batches.csv, data/windows/{window_specs,batch_flags}.parquet,
        data/did_controls.parquet
Output: results/tables/entropy_balance_smd.csv, results/tables/entropy_balance_outcomes.csv
Run:    python code/analysis/entropy_balancing.py
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
TAB = ROOT / "results" / "tables"
NX, NY = 6, 4
PL, PW = 120, 80
SEED = 20260702
PMETS = ["density", "clustering", "betweenness", "lambda2", "efficiency"]
ZEXTRA = ["vers_P_w10", "vers_Z_w10"]
GROUPS = ["strategic", "regular", "late"]


# covariates
def load_covariates():
    """Treated / control pre-window covariates per batch (5 P-layer + 5 Z-layer 6x4 + 2 versatility)."""
    P = pd.read_parquet(DATA / "metrics" / "metrics_player.parquet")
    Z = pd.read_parquet(DATA / "metrics" / "metrics_zone.parquet")
    P = P[(P.kind == "time") & (P.W_min == 15) & (P.side == "pre")]
    Z = Z[(Z.kind == "time") & (Z.W_min == 15) & (Z.side == "pre") & (Z.tiling == "6x4")]
    out = {}
    for unit in ["treated", "control"]:
        p = P[P.unit == unit].set_index("batch_id")[PMETS].add_prefix("P_")
        z = Z[Z.unit == unit].set_index("batch_id")[PMETS + ZEXTRA].add_prefix("Z_")
        out[unit] = p.join(z, how="inner")
    cols = out["treated"].columns
    both = out["treated"].join(out["control"], how="inner", lsuffix="_t", rsuffix="_c").dropna()
    Xt = both[[f"{c}_t" for c in cols]].to_numpy()
    Xc = both[[f"{c}_c" for c in cols]].to_numpy()
    return both.index.to_numpy(), Xt, Xc, list(cols)


def _eb_solve(Zc, lam):
    """Ridge EB dual: min logsumexp(Zc@theta) + lam*||theta||^2. lam=0 is exact balancing."""
    def obj(th):
        return logsumexp(Zc @ th) + lam * th @ th

    def grad(th):
        a = Zc @ th
        w = np.exp(a - logsumexp(a))
        return Zc.T @ w + 2 * lam * th

    res = minimize(obj, np.zeros(Zc.shape[1]), jac=grad, method="L-BFGS-B",
                   options=dict(maxiter=5000, ftol=1e-14, gtol=1e-12))
    a = Zc @ res.x
    a = a - logsumexp(a)
    return np.exp(a)


def entropy_balance(Xc, target, min_ess_frac=0.25):
    """Approximate entropy balancing (Zubizarreta-style stable weights).
    Start at lam=0 (exact); if weights are non-finite or ESS < min_ess_frac, relax
    along the ridge ladder; return the smallest lam with all |SMD_after| < 0.1 and
    ESS above threshold. Returns (w, lam, ok)."""
    mu, sd = Xc.mean(0), Xc.std(0)
    sd[sd == 0] = 1.0
    Zc = (Xc - mu) / sd - (target - mu) / sd  # constraint sum(w * Zc) = 0
    n = len(Xc)
    for lam in [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]:
        w = _eb_solve(Zc, lam)
        if not np.isfinite(w).all() or w.sum() <= 0:
            continue
        ess = 1.0 / np.sum(w ** 2)
        # weighted residual on the standardized scale (~SMD)
        resid = np.abs(w @ Zc)
        if resid.max() < 0.1 and ess >= min_ess_frac * n:
            return w, lam, True
        if resid.max() < 0.1 and lam >= 1e-1:
            return w, lam, False
    # no rung meets ESS threshold; loosest rung that balances
    for lam in [1e-1, 3e-2, 1e-2, 3e-3, 1e-3]:
        w = _eb_solve(Zc, lam)
        if np.isfinite(w).all() and w.sum() > 0 and np.abs(w @ Zc).max() < 0.1:
            return w, lam, False
    return _eb_solve(Zc, 1.0), 1.0, False


def smd(xt, xc, w=None):
    """Standardized mean difference; w=None for unweighted."""
    mt, vt = xt.mean(), xt.var()
    if w is None:
        mc, vc = xc.mean(), xc.var()
    else:
        mc = np.sum(w * xc)
        vc = np.sum(w * (xc - mc) ** 2)
    sd_ = np.sqrt((vt + vc) / 2)
    return (mt - mc) / sd_ if sd_ > 0 else np.nan


# outcomes
def zbin_vec(x, y):
    zx = np.clip((x / (PL / NX)).astype(int), 0, NX - 1)
    zy = np.clip((y / (PW / NY)).astype(int), 0, NY - 1)
    return zx * NY + zy


def zvec(w, ex_ids=None):
    if ex_ids is not None:
        w = w[~w.player_id.isin(ex_ids) & ~w.recipient_id.isin(ex_ids)]
    if len(w) == 0:
        return np.zeros(NX * NY)
    Z = np.zeros((NX * NY, NX * NY))
    np.add.at(Z, (zbin_vec(w.x.values, w.y.values), zbin_vec(w.end_x.values, w.end_y.values)), 1.0)
    np.fill_diagonal(Z, 0)
    t = Z / Z.sum() if Z.sum() > 0 else Z
    return t.sum(0) + t.sum(1)


def terr(w, ex_ids=None):
    cw = w if ex_ids is None else w[~w.player_id.isin(ex_ids) & ~w.recipient_id.isin(ex_ids)]
    return cw.x.mean() if len(cw) >= 5 else np.nan


def collect_outcomes():
    """Per batch: treated dz (subs excluded) / control dz vectors, treated dterr (subs excluded) / control dterr."""
    passes = pd.read_parquet(DATA / "events_pass.parquet")
    t1 = passes[passes.period == 1].groupby("match_id").t_period_sec.max()
    tc_all = np.where(passes.period == 1, passes.t_period_sec,
                      passes.match_id.map(t1) + passes.t_period_sec)
    passes = passes.assign(tc=tc_all)
    net = passes[passes.outcome.isna() & ~passes.is_set_piece]      # completed passes, zone network
    pall = passes[~passes.is_set_piece]                              # all open-play passes; terr filters to completed
    ni = {k: g for k, g in net.groupby(["match_id", "team_id"])}
    pi = {k: g for k, g in pall.groupby(["match_id", "team_id"])}
    b = pd.read_csv(DATA / "substitution_batches.csv").set_index("batch_id")
    sp = pd.read_parquet(DATA / "windows" / "window_specs.parquet")
    st = sp[(sp.kind == "time") & (sp.W_min == 15)].set_index(["batch_id", "side"])
    fl = pd.read_parquet(DATA / "windows" / "batch_flags.parquet")
    nontr = set(fl[(fl.W_min == 15) & ~fl.post_truncated].batch_id)
    ctrl = pd.read_parquet(DATA / "did_controls.parquet")
    ctrl = ctrl[ctrl.control_type != "none"].set_index("batch_id")

    def win(src, mid, tid, lo, hi, lo_open):
        g = src.get((mid, tid))
        if g is None:
            return None
        return g[(g.tc > lo) & (g.tc <= hi)] if lo_open else g[(g.tc >= lo) & (g.tc < hi)]

    rec = {}
    for bid in nontr:
        if bid not in ctrl.index or (bid, "pre") not in st.index or bid not in b.index:
            continue
        r = b.loc[bid]
        pre, post = st.loc[(bid, "pre")], st.loc[(bid, "post")]
        c = ctrl.loc[bid]
        tcc = c.control_t_center
        wp = win(ni, r.match_id, r.team_id, pre.t_start, pre.t_end, False)
        wq = win(ni, r.match_id, r.team_id, post.t_start, post.t_end, True)
        wcp = win(ni, int(c.control_match_id), int(c.control_team_id), tcc - 900, tcc, False)
        wcq = win(ni, int(c.control_match_id), int(c.control_team_id), tcc, tcc + 900, True)
        if any(w is None or len(w) < 10 for w in [wp, wq, wcp, wcq]):
            continue
        swapped = ([int(x) for x in str(r.players_out_id).split("|")]
                   + [int(x) for x in str(r.players_in_id).split("|")])
        tz = zvec(wq, swapped) - zvec(wp, swapped)          # treated dz, subs excluded
        cz = zvec(wcq) - zvec(wcp)                           # control dz
        # terr windows (completed passes)
        tp = win(pi, r.match_id, r.team_id, pre.t_start, pre.t_end, False)
        tq = win(pi, r.match_id, r.team_id, post.t_start, post.t_end, True)
        tcp = win(pi, int(c.control_match_id), int(c.control_team_id), tcc - 900, tcc, False)
        tcq = win(pi, int(c.control_match_id), int(c.control_team_id), tcc, tcc + 900, True)
        dt = dc = np.nan
        if all(w is not None and len(w) >= 10 for w in [tp, tq, tcp, tcq]):
            tp, tq = tp[tp.outcome.isna()], tq[tq.outcome.isna()]
            tcp, tcq = tcp[tcp.outcome.isna()], tcq[tcq.outcome.isna()]
            dt = terr(tq, swapped) - terr(tp, swapped)       # treated dterr, subs excluded
            dc = terr(tcq) - terr(tcp)                       # control dterr
        rec[bid] = dict(timing=r.timing_group, tz=tz, cz=cz, dterr_t=dt, dterr_c=dc)
    return rec


def snr_weighted(T, C, w, N, rng, n_boot=800, n_null=400):
    """Weighted unpaired DiD SNR: obs=||mean(T_boot) - mean(C_boot~w)||, null=two-sided sign flip."""
    nt, nc = len(T), len(C)
    mags, nulls = [], []
    for _ in range(n_boot):
        ti = rng.integers(nt, size=N)
        ci = rng.choice(nc, size=N, p=w)
        mags.append(np.linalg.norm(T[ti].mean(0) - C[ci].mean(0)))
    for _ in range(n_null):
        ti = rng.integers(nt, size=N)
        ci = rng.choice(nc, size=N, p=w)
        st = rng.choice([1, -1], size=N)[:, None]
        sc = rng.choice([1, -1], size=N)[:, None]
        nulls.append(np.linalg.norm((T[ti] * st).mean(0) - (C[ci] * sc).mean(0)))
    obs = np.mean(mags)
    nulls = np.array(nulls)
    return obs, obs / nulls.mean(), 1 - (obs > nulls).mean()


def main():
    rng = np.random.default_rng(SEED)
    bids, Xt, Xc, cols = load_covariates()
    print(f"Batches with complete covariate pairs: {len(bids)}  (12 covariates: {cols})")

    # 0. collect outcomes first; balancing set must equal analysis set
    print("\nCollecting outcomes (per-batch windows, about 1-2 minutes)...")
    rec = collect_outcomes()
    covidx = {bid: i for i, bid in enumerate(bids)}
    keep = np.array([bid in rec for bid in bids])
    bids, Xt, Xc = np.array(bids)[keep], Xt[keep], Xc[keep]
    timing = pd.Series([rec[b]["timing"] for b in bids], index=bids)
    print(f"Analysis set (outcomes + covariates complete): {len(bids)}")

    # 1. entropy balancing per timing group
    smd_rows, weights = [], {}
    for g in GROUPS:
        m = (timing == g).to_numpy()
        xt, xc = Xt[m], Xc[m]
        w, lam, ok = entropy_balance(xc, xt.mean(0))
        ess = 1.0 / np.sum(w ** 2)
        weights[g] = (np.array(bids)[m], w)
        print(f"[{g:9s}] n={m.sum()}  lam={lam:g}  ESS={ess:.0f} ({ess/m.sum():.0%}) "
              f"{'ok' if ok else 'WARNING: ESS below the 25% threshold'}")
        for j, cname in enumerate(cols):
            smd_rows.append(dict(group=g, metric=cname, n=int(m.sum()), lam=lam,
                                 smd_before=smd(xt[:, j], xc[:, j]),
                                 smd_after=smd(xt[:, j], xc[:, j], w),
                                 ess=ess))
    S = pd.DataFrame(smd_rows)
    S.to_csv(TAB / "entropy_balance_smd.csv", index=False)
    nb = (S.smd_before.abs() > 0.1).sum()
    na = (S.smd_after.abs() > 0.1).sum()
    print(f"\n|SMD|>0.1: before {nb}/{len(S)} -> after {na}/{len(S)}")
    print(f"max|SMD|: before {S.smd_before.abs().max():.3f} -> after {S.smd_after.abs().max():.3f}")

    # 2/3. weighted re-estimates
    out_rows = []

    # A: zone reorg SNR (subs excluded), weighted vs unweighted
    packs = {}
    for g in GROUPS:
        gb, w = weights[g]
        packs[g] = (np.array([rec[b]["tz"] for b in gb]),
                    np.array([rec[b]["cz"] for b in gb]), w)
    N = min(len(p[0]) for p in packs.values())
    print(f"\n=== A. zone reorg SNR (subs excluded, equal N={N}) ===")
    print(f"{'group':9s} | {'unweighted SNR(p)':>18s} | {'balanced SNR(p)':>18s}")
    snr_u, snr_w = {}, {}
    for g in GROUPS:
        T, C, w = packs[g]
        uni = np.full(len(C), 1.0 / len(C))
        _, su, pu = snr_weighted(T, C, uni, N, rng)
        _, sw, pw = snr_weighted(T, C, w, N, rng)
        snr_u[g], snr_w[g] = su, sw
        print(f"{g:9s} | {su:8.2f} (p={pu:.3f}) | {sw:8.2f} (p={pw:.3f})")
        out_rows.append(dict(outcome="zone_reorg_snr_excl", group=g, n=len(T),
                             est_unweighted=su, p_unweighted=pu,
                             est_weighted=sw, p_weighted=pw))
    thr_u = snr_u["late"] > snr_u["strategic"] and snr_u["regular"] > snr_u["strategic"]
    thr_w = snr_w["late"] > snr_w["strategic"] and snr_w["regular"] > snr_w["strategic"]
    print(f"threshold shape (strategic < regular and strategic < late): unweighted {'yes' if thr_u else 'no'} / balanced {'yes' if thr_w else 'no'}")

    # B: d_terr (subs excluded), weighted unpaired DiD + within-group bias correction
    from scipy.stats import ttest_1samp
    print("\n=== B. d_terr (subs excluded, remaining teammates) ===")
    covidx = {bid: i for i, bid in enumerate(bids)}
    bc_all = []  # within-group bias-corrected pairs, pooled for ALL
    for g in GROUPS:
        gb, w = weights[g]
        ok = np.array([not (np.isnan(rec[b]["dterr_t"]) or np.isnan(rec[b]["dterr_c"]))
                       for b in gb])
        gb2, w2 = gb[ok], w[ok] / w[ok].sum()
        dt = np.array([rec[b]["dterr_t"] for b in gb2])
        dc = np.array([rec[b]["dterr_c"] for b in gb2])
        idx = [covidx[b] for b in gb2]
        Xt_a, Xc_a = Xt[idx], Xc[idx]
        est_u = dt.mean() - dc.mean()
        est_w = dt.mean() - np.sum(w2 * dc)
        bs = []
        for _ in range(2000):
            ti = rng.integers(len(dt), size=len(dt))
            ci = rng.choice(len(dc), size=len(dc), p=w2)
            bs.append(dt[ti].mean() - dc[ci].mean())
        bs = np.array(bs)
        p_w = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
        # within-group paired bias correction (OLS of control dterr on X)
        sd_c = Xc_a.std(0); sd_c[sd_c == 0] = 1.0
        A = np.column_stack([np.ones(len(dc)), (Xc_a - Xc_a.mean(0)) / sd_c])
        beta, *_ = np.linalg.lstsq(A, dc, rcond=None)
        mhat = lambda X: (np.column_stack([np.ones(len(X)),
                                           (X - Xc_a.mean(0)) / sd_c]) @ beta)
        did_bc = (dt - dc) - (mhat(Xt_a) - mhat(Xc_a))
        bc_all.append(did_bc)
        t_bc, p_bc = ttest_1samp(did_bc, 0)
        print(f"  [{g:9s}] n={len(dt):4d}  unweighted {est_u:+.3f} | balanced {est_w:+.3f} "
              f"(p={p_w:.3f}) | within-group bias-corrected {did_bc.mean():+.3f} (p={p_bc:.3f})")
        out_rows.append(dict(outcome="dterr_excl", group=g, n=len(dt),
                             est_unweighted=est_u, p_unweighted=np.nan,
                             est_weighted=est_w, p_weighted=p_w,
                             est_biascorr=did_bc.mean(), p_biascorr=p_bc))
    allbc = np.concatenate(bc_all)
    t_a, p_a = ttest_1samp(allbc, 0)
    print(f"  [ALL, within-group corrected, pooled] n={len(allbc)}  {allbc.mean():+.3f} (p={p_a:.3f})")
    out_rows.append(dict(outcome="dterr_excl", group="ALL_withinbc", n=len(allbc),
                         est_unweighted=np.nan, p_unweighted=np.nan,
                         est_weighted=np.nan, p_weighted=np.nan,
                         est_biascorr=allbc.mean(), p_biascorr=p_a))
    pd.DataFrame(out_rows).to_csv(TAB / "entropy_balance_outcomes.csv", index=False)

    print("\nVerdict:")
    ok_bal = na == 0
    print(f"  {'[OK]' if ok_bal else '[WARN]'} balance: all |SMD|<0.1 {'achieved' if ok_bal else 'not fully achieved'}")
    print(f"  {'[OK]' if thr_w else '[FAIL]'} threshold shape {'preserved' if thr_w else 'lost'} after balancing")
    print("Saved entropy_balance_smd.csv and entropy_balance_outcomes.csv")


if __name__ == "__main__":
    main()
