"""Calibration inversion: coupling lambda per timing group with bootstrap CIs.

P1 replays the generator SNR(lambda) curve and inverts the headline targets;
P2 multi-seed curves; P3 batch-level bootstrap of the observed SNR; P4 Monte-Carlo
propagation of both into lambda and the late/early ratio; P5 early-group upper bound.
Inputs:  results/tables/{bias_predictability,artifact_exclusion}.csv, data/{events_pass.parquet,substitution_batches.csv,did_controls.parquet,windows/*}
Outputs: results/tables/inversion_ci.csv
Run: python code/audits/calibration_inversion.py
"""

import importlib.util
import time
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]  # <repo>/code

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[2]
TAB = ROOT / "results" / "tables"
DATA = ROOT / "data"
GROUPS = ["strategic", "regular", "late"]
OLD_TARGETS = {"strategic": 1.05, "regular": 1.57, "late": 1.64}     # previous targets
NEW_TARGETS = {"strategic": 1.052, "regular": 1.561, "late": 1.594}  # control_robustness.csv subset=all
OLD_LAM = {"strategic": 0.0244, "regular": 0.1508, "late": 0.213}    # previous inversion (artifact_exclusion.csv), replay check
SEED_NEW = 20260808
LAM_GRID_FINE = np.round(np.arange(0.0, 0.3001, 0.005), 4)  # 61 points
NSEEDS = 24     # seeds per group, P2/P5
B_BOOT = 200    # batch bootstrap reps, P3
D_MC = 4000     # MC draws, P4


def _load_module(fname, name):
    spec = importlib.util.spec_from_file_location(name, CODE_ROOT / "analysis" / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m47 = _load_module("artifact_exclusion.py", "ae47")   # make_batches / snr_group / K
m54 = _load_module("control_robustness.py", "cr54")   # zvec / zbin_vec


# P1: replay
def invert_47(curve, lam_grid, tgt):
    """Original inversion: searchsorted + two-point interpolation."""
    i = np.searchsorted(curve, tgt)
    if i == 0:
        return 0.0
    if i >= len(curve):
        return np.nan
    return float(np.interp(tgt, curve[i - 1:i + 1], lam_grid[i - 1:i + 1]))


def replay_47():
    """Replay the original rng stream (S1/S2 advance state, then S3 curve); unrounded."""
    rng = np.random.default_rng(m47.SEED)  # 20260702
    bp = pd.read_csv(TAB / "bias_predictability.csv")
    vols = {g: np.clip(bp[bp.timing == g].volume.values.astype(int), 10, None) for g in GROUPS}
    ns = {g: (bp.timing == g).sum() for g in GROUPS}
    N = min(ns.values())
    r_common = {g: rng.dirichlet(np.full(m47.K, 0.5)) for g in GROUPS}
    for tag, lam_by_g, mode in [("S1_null", {g: 0.0 for g in GROUPS}, "idio"),
                                ("S2a_idio_lam0.15", {g: 0.15 for g in GROUPS}, "idio"),
                                ("S2b_common_lam0.04", {g: 0.04 for g in GROUPS}, "common"),
                                ("S2b_common_lam0.06", {g: 0.06 for g in GROUPS}, "common")]:
        for g in GROUPS:
            arr = m47.make_batches(rng, ns[g], vols[g], lam_by_g[g], mode, r_common[g])
            m47.snr_group(arr, N, rng)  # advance rng only
    lam_grid = np.arange(0.0, 0.251, 0.02)
    curves = {}
    for g in GROUPS:
        c = []
        for lam in lam_grid:
            arr = m47.make_batches(rng, ns[g], vols[g], lam, "common", r_common[g])
            c.append(m47.snr_group(arr, N, rng))
        curves[g] = np.array(c)
    return lam_grid, curves, vols, ns, N


# P2/P5: multi-seed response curves
def curve_multiseed(vols_g, n_g, N_boot, seed_key, nseeds=NSEEDS):
    """One SNR(lambda) curve per seed, independent rng and r_common."""
    out = np.empty((nseeds, len(LAM_GRID_FINE)))
    for i in range(nseeds):
        rng = np.random.default_rng([SEED_NEW] + seed_key + [i])
        r_common = rng.dirichlet(np.full(m47.K, 0.5))
        for j, lam in enumerate(LAM_GRID_FINE):
            arr = m47.make_batches(rng, n_g, vols_g, float(lam), "common", r_common)
            out[i, j] = m47.snr_group(arr, N_boot, rng)
    return out


def monotonize(curve):
    """Isotonic fit plus a tiny ramp so np.interp sees a strictly increasing curve."""
    c = IsotonicRegression(increasing=True).fit_transform(LAM_GRID_FINE, curve)
    return c + np.arange(len(c)) * 1e-9


def invert_mono(c_mono, tgt):
    """Invert lambda on a monotone curve; below start -> 0.0, above end -> NaN (censored > 0.30)."""
    if tgt <= c_mono[0]:
        return 0.0
    if tgt > c_mono[-1]:
        return np.nan
    return float(np.interp(tgt, c_mono, LAM_GRID_FINE))


# P3: observation side
def build_recs():
    """Headline data prep; returns {batch_id: (timing, ctype, dz_excl)}."""
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

    def winp(mid, tid, lo, hi, lo_open):
        g = ni.get((mid, tid))
        if g is None:
            return None
        return g[(g.tc > lo) & (g.tc <= hi)] if lo_open else g[(g.tc >= lo) & (g.tc < hi)]

    recs = {}
    for bid in nontr:
        if bid not in ctrl.index or (bid, "pre") not in st.index:
            continue
        r = b.loc[bid]
        pre, post = st.loc[(bid, "pre")], st.loc[(bid, "post")]
        wp = winp(r.match_id, r.team_id, pre.t_start, pre.t_end, False)
        wq = winp(r.match_id, r.team_id, post.t_start, post.t_end, True)
        c = ctrl.loc[bid]
        tc = c.control_t_center
        wcp = winp(int(c.control_match_id), int(c.control_team_id), tc - 900, tc, False)
        wcq = winp(int(c.control_match_id), int(c.control_team_id), tc, tc + 900, True)
        if any(w is None or len(w) < 10 for w in [wp, wq, wcp, wcq]):
            continue
        swapped = ([int(x) for x in str(r.players_out_id).split("|")]
                   + [int(x) for x in str(r.players_in_id).split("|")])
        dz = (m54.zvec(wq, swapped) - m54.zvec(wp, swapped)) - (m54.zvec(wcq) - m54.zvec(wcp))
        recs[bid] = (r.timing_group, c.control_type, dz)
    return recs


def snr54(arr, N, rng, nb=800, nn=400):
    """Headline SNR estimator with explicit rng."""
    mags = [np.linalg.norm(arr[rng.integers(len(arr), size=N)].mean(0)) for _ in range(nb)]
    nf = []
    for _ in range(nn):
        idx = rng.integers(len(arr), size=N)
        sgn = rng.choice([1, -1], size=len(arr))
        nf.append(np.linalg.norm((arr[idx] * sgn[idx, None]).mean(0)))
    obs = np.mean(mags)
    return obs / np.mean(nf), 1 - (obs > np.array(nf)).mean()


def main():
    t_start = time.time()
    rows = []

    # P1: replay + inversion of old and new targets
    print("=" * 78)
    print("P1  replay of the S3 curve (SEED=20260702, S1/S2 rng state included)")
    print("=" * 78)
    t0 = time.time()
    lam_grid47, curves47, vols, ns, N47 = replay_47()
    print(f"  group n={ns}  equal-sample N={N47}  {time.time()-t0:.1f}s")

    # check against the previous curve (3 decimals)
    ae = pd.read_csv(TAB / "artifact_exclusion.csv")
    ok_curve = True
    for g in GROUPS:
        saved = ae[(ae.scenario == "S3_curve") & (ae.timing == g)].sort_values("lam").snr.values
        mine = np.round(curves47[g], 3)
        if not np.allclose(saved, mine):
            ok_curve = False
            print(f"  WARNING {g} curve differs from the previous run: saved={saved} mine={mine}")
    print(f"  S3 curve vs previous run (3 decimals): {'match' if ok_curve else 'MISMATCH'}")

    lam_old, lam_new = {}, {}
    for g in GROUPS:
        lam_old[g] = invert_47(curves47[g], lam_grid47, OLD_TARGETS[g])
        lam_new[g] = invert_47(curves47[g], lam_grid47, NEW_TARGETS[g])
        rows += [dict(block="curve_47replay", timing=g, lam=round(l, 3), snr=s)
                 for l, s in zip(lam_grid47, curves47[g])]
    ok_old = all(round(lam_old[g], 4) == OLD_LAM[g] for g in GROUPS)
    print(f"  old-target inversion: " + " ".join(f"{g}={lam_old[g]:.4f}(previous {OLD_LAM[g]})" for g in GROUPS)
          + f" -> {'match' if ok_old else 'MISMATCH'}")
    ratio_new = lam_new["late"] / lam_new["strategic"] if lam_new["strategic"] > 0 else np.inf
    print(f"  new-target (1.052/1.561/1.594) inversion: "
          + " ".join(f"{g}={lam_new[g]:.4f}" for g in GROUPS)
          + f"  late/early={ratio_new:.2f}")
    for g in GROUPS:
        rows.append(dict(block="inversion_old_replay", timing=g, value=round(lam_old[g], 4),
                         snr=OLD_TARGETS[g], note=f"previous={OLD_LAM[g]}; match={ok_old}"))

    # P2: multi-seed curves (generator-side uncertainty)
    print("\n" + "=" * 78)
    print(f"P2  multi-seed SNR(lambda) curves: lambda in [0,0.30] step 0.005 x {NSEEDS} seeds/group (N={N47})")
    print("=" * 78)
    t0 = time.time()
    seed_curves, iso_curves, mean_curves = {}, {}, {}
    for gi, g in enumerate(GROUPS):
        seed_curves[g] = curve_multiseed(vols[g], ns[g], N47, [gi])
        iso_curves[g] = [monotonize(seed_curves[g][i]) for i in range(NSEEDS)]
        mean_curves[g] = seed_curves[g].mean(0)
        sd = seed_curves[g].std(0, ddof=1)
        for j, lam in enumerate(LAM_GRID_FINE):
            rows.append(dict(block="curve_multiseed", timing=g, lam=float(lam),
                             snr=round(mean_curves[g][j], 4), snr_sd=round(sd[j], 4),
                             n_seeds=NSEEDS, N_boot=N47))
        lam_mean = invert_mono(monotonize(mean_curves[g]), NEW_TARGETS[g])
        print(f"  {g:10s}: between-seed SD [{sd.min():.3f},{sd.max():.3f}]  "
              f"mean-curve inversion lambda={lam_mean:.4f} (P1 point {lam_new[g]:.4f})")
    print(f"  {time.time()-t0:.1f}s")

    # P3: observation side, headline replay + batch bootstrap
    print("\n" + "=" * 78)
    print("P3  observation side: rebuild batch dz -> replay point estimate -> batch bootstrap B=200 (N=73)")
    print("=" * 78)
    t0 = time.time()
    recs = build_recs()
    by = {g: np.array([dz for (gg, ct, dz) in recs.values() if gg == g]) for g in GROUPS}
    ns_obs = {g: len(by[g]) for g in GROUPS}
    N_obs = min(ns_obs.values())
    print(f"  rebuilt batches: n={ns_obs}  equal-sample N={N_obs}  {time.time()-t0:.1f}s")

    # replay subset=all point estimate (first three draws of rng 20260613)
    rng54 = np.random.default_rng(20260613)
    snr_point = {}
    for g in GROUPS:
        s, p = snr54(by[g], N_obs, rng54)
        snr_point[g] = s
    ok54 = all(round(snr_point[g], 3) == NEW_TARGETS[g] for g in GROUPS)
    print(f"  headline point estimate replay: " + " ".join(f"{g}={snr_point[g]:.3f}" for g in GROUPS)
          + f" vs 1.052/1.561/1.594 -> {'match' if ok54 else 'MISMATCH'}")

    t0 = time.time()
    boot = {}
    for gi, g in enumerate(GROUPS):
        rng_b = np.random.default_rng([SEED_NEW, 54, gi])
        n = ns_obs[g]
        vals = np.empty(B_BOOT)
        for bidx in range(B_BOOT):
            arr_b = by[g][rng_b.integers(n, size=n)]
            vals[bidx], _ = snr54(arr_b, N_obs, rng_b)
        boot[g] = vals
        lo, hi = np.percentile(vals, [2.5, 97.5])
        rows.append(dict(block="obs_target_boot", timing=g, value=round(snr_point[g], 4),
                         snr=round(vals.mean(), 4), snr_sd=round(vals.std(ddof=1), 4),
                         lo95=round(lo, 4), hi95=round(hi, 4),
                         note=f"B={B_BOOT}; batch-level resample; N_boot={N_obs}"))
        print(f"  {g:10s}: point={snr_point[g]:.3f}  boot mean={vals.mean():.3f}  "
              f"SD={vals.std(ddof=1):.3f}  95%[{lo:.3f},{hi:.3f}]")
    print(f"  bootstrap {time.time()-t0:.1f}s")

    # P4: two-sided MC propagation
    print("\n" + "=" * 78)
    print(f"P4  two-sided propagation: D={D_MC} draws (random seed curve x random bootstrap target), seed [{SEED_NEW},999]")
    print("=" * 78)
    rng_mc = np.random.default_rng([SEED_NEW, 999])
    lam_draws, cens = {}, {}
    for g in GROUPS:
        si = rng_mc.integers(NSEEDS, size=D_MC)
        ti = rng_mc.integers(B_BOOT, size=D_MC)
        d = np.array([invert_mono(iso_curves[g][si[k]], boot[g][ti[k]]) for k in range(D_MC)])
        cens[g] = np.isnan(d).mean()          # right-censored, lambda > 0.30
        d = np.where(np.isnan(d), 0.30, d)    # censored -> 0.30, share reported in note
        lam_draws[g] = d
        lo, hi = np.percentile(d, [2.5, 97.5])
        pz = (d == 0).mean()
        hi_s = f"{hi:.4f}" if cens[g] <= 0.025 else ">0.30"
        rows.append(dict(block="inversion_new", timing=g, value=round(lam_new[g], 4),
                         snr=NEW_TARGETS[g], lo95=round(lo, 4),
                         hi95=round(hi, 4) if cens[g] <= 0.025 else 0.30,
                         note=f"MC D={D_MC}; P(lam=0)={pz:.3f}; censored>0.30={cens[g]:.3f}"))
        print(f"  {g:10s}: lambda={lam_new[g]:.4f}  95%CI[{lo:.4f},{hi_s}]  "
              f"P(lambda=0)={pz:.3f}  censored(>0.30)={cens[g]:.3f}")

    ratio_draws = np.where(lam_draws["strategic"] > 0,
                           lam_draws["late"] / np.maximum(lam_draws["strategic"], 1e-12), np.inf)
    r_lo, r_med, r_hi = np.percentile(ratio_draws, [2.5, 50, 97.5])
    frac_inf = np.isinf(ratio_draws).mean()
    hi_str = "inf" if np.isinf(r_hi) else f"{r_hi:.2f}"
    rows.append(dict(block="ratio_late_over_early", timing="late/strategic",
                     value=round(ratio_new, 3), lo95=round(r_lo, 3),
                     hi95=(np.inf if np.isinf(r_hi) else round(r_hi, 3)),
                     snr_sd=round(r_med, 3),
                     note=f"median in snr_sd col; P(ratio=inf)={frac_inf:.3f}; "
                          f"lower bound => 'at least {r_lo:.1f}x'"))
    print(f"\n  late/early ratio: point={ratio_new:.2f}  median={r_med:.2f}  "
          f"95%CI[{r_lo:.2f},{hi_str}]  P(inf)={frac_inf:.3f}")
    if np.isinf(r_hi):
        print(f"    -> early draws cover lambda=0, upper bound diverges; report at least {r_lo:.1f}x (95% lower bound)")

    # P5: early-group equivalence bound (N=73)
    print("\n" + "=" * 78)
    print(f"P5  early equivalence bound: strategic volumes, SNR with N=73, {NSEEDS} seeds")
    print("=" * 78)
    t0 = time.time()
    sc73 = curve_multiseed(vols["strategic"], ns["strategic"], N_obs, [73])
    m73, s73 = sc73.mean(0), sc73.std(0, ddof=1)
    for j, lam in enumerate(LAM_GRID_FINE):
        rows.append(dict(block="curve_N73_early", timing="strategic", lam=float(lam),
                         snr=round(m73[j], 4), snr_sd=round(s73[j], 4),
                         n_seeds=NSEEDS, N_boot=N_obs))
    tgt_e = NEW_TARGETS["strategic"]
    lam50 = invert_mono(monotonize(m73), tgt_e)               # mean curve crosses 1.052
    lam975 = invert_mono(monotonize(m73 - 1.96 * s73), tgt_e)  # normal-approx 2.5% quantile crosses 1.052
    frac_le = (sc73 <= tgt_e).mean(0)                          # empirical P(SNR <= 1.052) per lambda
    idx_emp = np.where(frac_le >= 0.025)[0]
    lam975_emp = float(LAM_GRID_FINE[idx_emp[-1]]) if len(idx_emp) else 0.0
    rows.append(dict(block="early_equiv_bound", timing="strategic", value=round(lam50, 4),
                     snr=tgt_e, note=f"lam50: mean-curve crossing; N_boot={N_obs}"))
    rows.append(dict(block="early_equiv_bound", timing="strategic", value=round(lam975, 4),
                     snr=tgt_e, note=f"lam97.5: (mean-1.96SD) crossing, normal approx; "
                                     f"empirical(>=1/{NSEEDS} seeds)={lam975_emp}"))
    print(f"  lambda_50   (P(SNR<={tgt_e}) >= 50%)   = {lam50:.4f}")
    print(f"  lambda_97.5 (2.5% quantile <={tgt_e}, normal approx) = {lam975:.4f}  (empirical grid={lam975_emp:.3f})")
    print(f"  {time.time()-t0:.1f}s")

    # write
    out = pd.DataFrame(rows)
    cols = ["block", "timing", "lam", "snr", "snr_sd", "n_seeds", "N_boot",
            "value", "lo95", "hi95", "note"]
    out = out.reindex(columns=cols)
    out.to_csv(TAB / "inversion_ci.csv", index=False)
    print(f"\nwrote {TAB / 'inversion_ci.csv'} ({len(out)} rows)  total {time.time()-t_start:.0f}s")
    print("\nsummary: new lambda triple = "
          + "/".join(f"{lam_new[g]:.4f}" for g in GROUPS)
          + f"  ratio={ratio_new:.2f} 95%CI[{r_lo:.2f},{hi_str}]"
          + f"  early bound lambda50={lam50:.4f} lambda97.5={lam975:.4f}")


if __name__ == "__main__":
    main()
