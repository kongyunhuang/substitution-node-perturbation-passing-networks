"""Monte-Carlo SE of the headline SNRs and the null noise floor.

Replays the headline SNR estimator, then: (1) SE over 8 seeds, (2) SNR floor
under per-batch random sign flips (200 reps) with empirical type-I error,
(3) occupancy variant keeping same-cell passes.
Inputs:  data/{events_pass.parquet,substitution_batches.csv,did_controls.parquet,windows/*}
Outputs: results/tables/snr_mc_se.csv
Run: python code/audits/snr_monte_carlo_se.py
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
TAB = ROOT / "results" / "tables"
NX, NY, PL, PW = 6, 4, 120, 80
GROUPS = ["strategic", "regular", "late"]

SEED_HEADLINE = 20260613                       # seed of the published values
SEEDS_MC = list(range(20260601, 20260609))     # check 1
SEED_H0 = 20260808                             # check 2: outer sign-flip seed
N_H0_REP = 200                                 # check 2: outer reps
N_BOOT, N_FLIP = 800, 400                      # as in control_robustness.py

# published values: gradient_selfexcl.csv snr_excl; control_robustness.csv subset=all
PUB35 = {"strategic": 1.0485, "regular": 1.5745, "late": 1.6438}
PUB54 = {"strategic": 1.052, "regular": 1.561, "late": 1.594}
HEADLINE_P = {"strategic": 0.39, "regular": 0.0025, "late": 0.0075}
# analytic floor prediction, comparison only
ANALYTIC_FLOOR = {"strategic": 0.93, "regular": 0.92, "late": 0.71}


# next three functions copied from control_robustness.py
def zbin_vec(x, y):
    zx = np.clip((x / (PL / NX)).astype(int), 0, NX - 1)
    zy = np.clip((y / (PW / NY)).astype(int), 0, NY - 1)
    return zx * NY + zy


def zvec(w, ex_ids=None):
    """Reference zvec, diagonal removed."""
    if ex_ids is not None:
        w = w[~w.player_id.isin(ex_ids) & ~w.recipient_id.isin(ex_ids)]
    if len(w) == 0:
        return np.zeros(24)
    Z = np.zeros((24, 24))
    np.add.at(Z, (zbin_vec(w.x.values, w.y.values), zbin_vec(w.end_x.values, w.end_y.values)), 1.0)
    np.fill_diagonal(Z, 0)
    t = Z / Z.sum() if Z.sum() > 0 else Z
    return t.sum(0) + t.sum(1)


def snr_loop(arr, N, rng):
    """Headline snr() with explicit rng; same RNG call order."""
    mags = [np.linalg.norm(arr[rng.integers(len(arr), size=N)].mean(0)) for _ in range(N_BOOT)]
    nf = []
    for _ in range(N_FLIP):
        idx = rng.integers(len(arr), size=N)
        sgn = rng.choice([1, -1], size=len(arr))
        nf.append(np.linalg.norm((arr[idx] * sgn[idx, None]).mean(0)))
    obs = np.mean(mags)
    return obs / np.mean(nf), 1 - (obs > np.array(nf)).mean()


def zvec_diag(w, ex_ids=None):
    """zvec without fill_diagonal, same-cell passes kept."""
    if ex_ids is not None:
        w = w[~w.player_id.isin(ex_ids) & ~w.recipient_id.isin(ex_ids)]
    if len(w) == 0:
        return np.zeros(24)
    Z = np.zeros((24, 24))
    np.add.at(Z, (zbin_vec(w.x.values, w.y.values), zbin_vec(w.end_x.values, w.end_y.values)), 1.0)
    # diagonal kept
    t = Z / Z.sum() if Z.sum() > 0 else Z
    return t.sum(0) + t.sum(1)


def snr_fast(arr, N, rng):
    """Vectorised snr_loop for the check-2 outer loop; signs drawn per batch,
    take_along_axis == sgn[idx] so repeated batches share a sign."""
    n = len(arr)
    idx = rng.integers(n, size=(N_BOOT, N))
    mags = np.linalg.norm(arr[idx].mean(axis=1), axis=1)
    idx2 = rng.integers(n, size=(N_FLIP, N))
    sgn = rng.choice([1, -1], size=(N_FLIP, n))
    nf = np.linalg.norm((arr[idx2] * np.take_along_axis(sgn, idx2, axis=1)[:, :, None]).mean(axis=1), axis=1)
    obs = mags.mean()
    return obs / nf.mean(), 1 - (obs > nf).mean()


def build_by_group(zfun):
    """Headline data load + batch DiD vectors; zfun picks the occupancy variant.
    Returns {timing_group: ndarray(n_g, 24)}, subset=all."""
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
        c = ctrl.loc[bid]; tc = c.control_t_center
        wcp = winp(int(c.control_match_id), int(c.control_team_id), tc - 900, tc, False)
        wcq = winp(int(c.control_match_id), int(c.control_team_id), tc, tc + 900, True)
        if any(w is None or len(w) < 10 for w in [wp, wq, wcp, wcq]):
            continue
        swapped = ([int(x) for x in str(r.players_out_id).split("|")]
                   + [int(x) for x in str(r.players_in_id).split("|")])
        dz = (zfun(wq, swapped) - zfun(wp, swapped)) - (zfun(wcq) - zfun(wcp))
        recs[bid] = (r.timing_group, dz)
    return {g: np.array([dz for (gg, dz) in recs.values() if gg == g]) for g in GROUPS}


def main():
    rows = []

    print("== loading data, DiD vectors (reference zvec) ==")
    by = build_by_group(zvec)
    ns = {g: len(by[g]) for g in GROUPS}
    N = min(ns.values())
    print(f"group n={ns}, common N={N}  (expected 425/404/73, N=73)")

    # check 0: reproduce headline
    print("\n== check 0: SEED=20260613 must reproduce the published values ==")
    rng = np.random.default_rng(SEED_HEADLINE)
    for g in GROUPS:
        s, p = snr_loop(by[g], N, rng)
        ok = (round(s, 3) == PUB54[g]) and (round(p, 4) == HEADLINE_P[g])
        print(f"  {g:10s}: SNR={s:.4f} (round3={round(s,3)}, published={PUB54[g]}) "
              f"p={p:.4f} (published={HEADLINE_P[g]})  {'MATCH' if ok else 'MISMATCH'}")
        rows.append(dict(part="repro_check", timing=g, seed=SEED_HEADLINE, n=ns[g],
                         N_resample=N, snr_excl=round(s, 4), p=round(p, 4),
                         headline_snr=PUB54[g], headline_p=HEADLINE_P[g],
                         note="MATCH" if ok else "MISMATCH"))

    # check 1: MC SE over 8 seeds
    print("\n== check 1: MC SE (8 seeds 20260601..20260608) ==")
    mc = {g: [] for g in GROUPS}
    for seed in SEEDS_MC:
        rng = np.random.default_rng(seed)
        for g in GROUPS:
            s, p = snr_loop(by[g], N, rng)
            mc[g].append(s)
            rows.append(dict(part="mc_seed", timing=g, seed=seed, n=ns[g],
                             N_resample=N, snr_excl=round(s, 4), p=round(p, 4)))
        print(f"  seed={seed}: " + "  ".join(f"{g}={mc[g][-1]:.4f}" for g in GROUPS))
    print("  -- summary --")
    for g in GROUPS:
        v = np.array(mc[g])
        m, sd = v.mean(), v.std(ddof=1)
        z35 = abs(PUB35[g] - m) / sd
        z54 = abs(PUB54[g] - m) / sd
        print(f"  {g:10s}: MC mean={m:.4f}  SD={sd:.4f}  "
              f"pub35 {PUB35[g]} at {z35:.2f} SD | pub54 {PUB54[g]} at {z54:.2f} SD")
        rows.append(dict(part="mc_summary", timing=g, seed="20260601-20260608", n=ns[g],
                         N_resample=N, mc_mean=round(m, 4), mc_sd=round(sd, 4),
                         pub35=PUB35[g], sd_from_mc_pub35=round(z35, 2),
                         pub54=PUB54[g], sd_from_mc_pub54=round(z54, 2)))

    # check 2: SNR floor under H0
    print(f"\n== check 2: H0 noise floor (per-batch sign flips x{N_H0_REP}, seed {SEED_H0}) ==")
    rng = np.random.default_rng(SEED_H0)   # one stream for flips and inner resampling
    for g in GROUPS:
        arr = by[g]
        snrs, ps = [], []
        for _ in range(N_H0_REP):
            sgn0 = rng.choice([1, -1], size=len(arr))
            s, p = snr_fast(arr * sgn0[:, None], N, rng)
            snrs.append(s); ps.append(p)
        snrs = np.array(snrs); ps = np.array(ps)
        lo, hi = np.percentile(snrs, [2.5, 97.5])
        t1e = (ps < 0.05).mean()
        print(f"  {g:10s}: H0 SNR mean={snrs.mean():.4f}  2.5%={lo:.4f}  97.5%={hi:.4f}  "
              f"type-I(p<.05)={t1e:.3f}  [analytic ~{ANALYTIC_FLOOR[g]}]")
        rows.append(dict(part="h0_floor", timing=g, seed=SEED_H0, n=ns[g],
                         N_resample=N, n_rep=N_H0_REP,
                         h0_mean=round(snrs.mean(), 4), h0_q025=round(lo, 4),
                         h0_q975=round(hi, 4), type1_err=round(t1e, 4),
                         analytic_floor=ANALYTIC_FLOOR[g]))

    # check 3: occupancy variant
    print(f"\n== check 3: occupancy variant (no fill_diagonal, seed {SEED_HEADLINE}) ==")
    by_d = build_by_group(zvec_diag)
    ns_d = {g: len(by_d[g]) for g in GROUPS}
    N_d = min(ns_d.values())
    print(f"  group n={ns_d}, N={N_d} (batch set should match the reference)")
    rng = np.random.default_rng(SEED_HEADLINE)
    for g in GROUPS:
        s, p = snr_loop(by_d[g], N_d, rng)
        print(f"  {g:10s}: SNR={s:.4f}  p={p:.4f}  (headline: SNR={PUB54[g]}, p={HEADLINE_P[g]})")
        rows.append(dict(part="diag_variant", timing=g, seed=SEED_HEADLINE, n=ns_d[g],
                         N_resample=N_d, snr_excl=round(s, 4), p=round(p, 4),
                         headline_snr=PUB54[g], headline_p=HEADLINE_P[g]))

    out = TAB / "snr_mc_se.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nsaved {out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
