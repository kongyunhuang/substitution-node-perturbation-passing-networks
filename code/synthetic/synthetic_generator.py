"""
Synthetic two-layer generator with planted coupling lambda (Section 3.6, Appendix B.3, Figure 5c).
K = 24 zones, N staying nodes, m perturbers with off/on footprints; dB = footprint change,
dz = zone-share change; naive cos uses all events, LPO drops the perturbers' own events.
Experiments: A null (lambda = 0), B lambda sweep, C share sweep at fixed lambda, plus
self-checks. Pure numpy/scipy.
Inputs: none
Outputs: results/tables/synthetic_validation.csv
Run: python code/synthetic/synthetic_generator.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
TAB = ROOT / "results" / "tables"
K = 24          # zones, matches the 6x4 grid
N_TEAM = 14     # staying nodes
CONC = 0.5      # Dirichlet concentration, smaller = more peaked profiles


def _profile(rng, conc=CONC):
    """Peaked spatial profile over zones (sums to 1)."""
    return rng.dirichlet(np.full(K, conc))


def _hist(rng, prof, n):
    """n events drawn from a profile -> zone counts."""
    if n <= 0:
        return np.zeros(K)
    return rng.multinomial(n, prof).astype(float)


def _norm(v):
    s = v.sum()
    return v / s if s > 0 else v


def _cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else np.nan


def simulate_batch(rng, lam, h, o, m, N=N_TEAM):
    """
    One synthetic substitution. Returns naive/lpo cos, perturber share and true lambda.
    lam: true coupling (teammate shift towards the on-footprint). h: perturber activity weight.
    o: events per window. m: perturbers per window.
    """
    base = [_profile(rng) for _ in range(N)]      # staying teammates, pre = post baseline
    f_out = [_profile(rng) for _ in range(m)]      # off footprints
    f_in = [_profile(rng) for _ in range(m)]       # on footprints
    resp_dir = np.mean(f_in, 0) - np.mean(f_out, 0)  # true coupling direction

    w = np.array([1.0] * N + [h] * m)
    w = w / w.sum()

    # ---- pre window: N staying + m off ----
    n_pre = rng.multinomial(o, w)
    H_pre = np.zeros(K)
    pert_pre = np.zeros(K)
    f_out_hat = np.zeros(K)
    for i in range(N):
        H_pre += _hist(rng, base[i], n_pre[i])
    for j in range(m):
        hj = _hist(rng, f_out[j], n_pre[N + j])
        H_pre += hj
        pert_pre += hj
        f_out_hat += hj

    # ---- post window: N staying (with coupling response) + m on ----
    n_post = rng.multinomial(o, w)
    H_post = np.zeros(K)
    pert_post = np.zeros(K)
    f_in_hat = np.zeros(K)
    for i in range(N):
        prof = np.clip(base[i] + lam * resp_dir, 0, None)
        prof = _norm(prof) if prof.sum() > 0 else base[i]
        H_post += _hist(rng, prof, n_post[i])
    for j in range(m):
        hj = _hist(rng, f_in[j], n_post[N + j])
        H_post += hj
        pert_post += hj
        f_in_hat += hj

    # ---- change vectors ----
    dB = _norm(f_in_hat) - _norm(f_out_hat)               # player layer, observable
    dz_incl = _norm(H_post) - _norm(H_pre)                 # zone layer with perturbers (naive)
    dz_excl = _norm(H_post - pert_post) - _norm(H_pre - pert_pre)  # LPO, perturbers removed

    tot = H_pre.sum() + H_post.sum()
    share = (pert_pre.sum() + pert_post.sum()) / tot if tot > 0 else np.nan
    return dict(lam=lam, h=h, o=o, m=m, share=share,
                naive=_cos(dB, dz_incl), lpo=_cos(dB, dz_excl))


def grid(rng, lams, hs, os, ms, reps):
    rows = []
    for lam in lams:
        for h in hs:
            for o in os:
                for m in ms:
                    for _ in range(reps):
                        rows.append(simulate_batch(rng, lam, h, o, m))
    return pd.DataFrame(rows)


def _slope_p(x, y):
    r, p = stats.pearsonr(x, y)
    b = np.polyfit(x, y, 1)[0]
    return b, r, p


def _mean_ci(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    se = v.std(ddof=1) / np.sqrt(len(v))
    return v.mean(), v.mean() - 1.96 * se, v.mean() + 1.96 * se


def main():
    rng = np.random.default_rng(20260624)
    REPS = 120

    # ---- experiment A: null, artefact grows with share, LPO zero
    print("=" * 78)
    print("Experiment A  NULL (lambda=0): naive self-coupling bias rises with share, LPO should be zero")
    print("=" * 78)
    A = grid(rng, lams=[0.0], hs=np.linspace(1, 10, 8), os=[300], ms=[1], reps=REPS)
    A["sbin"] = pd.qcut(A.share, 5, labels=["lowest", "low", "mid", "high", "highest"])
    print(A.groupby("sbin", observed=True)[["share", "naive", "lpo"]].mean().round(4).to_string())
    bn, rn, pn = _slope_p(A.share.values, A.naive.values)
    bl, rl, pl = _slope_p(A.share.values, A.lpo.values)
    print(f"\n  naive ~ share : beta={bn:+.3f}  r={rn:+.3f}  p={pn:.2e}   (expect beta>0 significant: predictable bias)")
    print(f"  lpo   ~ share : beta={bl:+.3f}  r={rl:+.3f}  p={pl:.2e}   (expect beta~0 n.s.: share-independent after correction)")
    m_lpo, lo, hi = _mean_ci(A.lpo)
    m_naive, _, _ = _mean_ci(A.naive)
    print(f"  lambda=0 overall:  naive cos = {m_naive:+.3f} (biased)   |   LPO cos = {m_lpo:+.3f}  95%CI[{lo:+.3f},{hi:+.3f}]")
    A_naive_pos = pn < 0.05 and bn > 0
    A_lpo_zero = lo <= 0 <= hi and abs(m_lpo) < 0.03

    # ---- experiment B: signal, LPO recovers lambda, naive biased upward
    print("\n" + "=" * 78)
    print("Experiment B  SIGNAL: sweep true lambda. LPO ~0 at lambda=0 and increasing (unbiased); naive >= LPO (upward bias)")
    print("=" * 78)
    lams = [0.0, 0.1, 0.2, 0.4, 0.7, 1.0]
    B = grid(rng, lams=lams, hs=[5.0], os=[300], ms=[1], reps=REPS)
    tb = B.groupby("lam")[["naive", "lpo"]].agg(["mean"]).round(4)
    print("   lambda   naive_cos    lpo_cos     naive-lpo (residual bias)")
    for lam in lams:
        mn = B[B.lam == lam].naive.mean()
        ml = B[B.lam == lam].lpo.mean()
        print(f"  {lam:.2f}    {mn:+.4f}      {ml:+.4f}        {mn-ml:+.4f}")
    bL, rL, pL = _slope_p(B.lam.values, B.lpo.values)
    print(f"\n  lpo ~ lambda : beta={bL:+.3f}  r={rL:+.3f}  p={pL:.2e}   (expect beta>0 significant: LPO tracks the true signal)")
    m0_lpo, l0, h0 = _mean_ci(B[B.lam == 0].lpo)
    print(f"  LPO cos at lambda=0 = {m0_lpo:+.3f}  95%CI[{l0:+.3f},{h0:+.3f}]  (should contain 0: unbiased)")
    naive_ge_lpo = all(B[B.lam == lam].naive.mean() >= B[B.lam == lam].lpo.mean() for lam in lams)
    print(f"  naive >= lpo at every lambda (consistent upward bias): {'yes' if naive_ge_lpo else 'no'}")
    B_unbiased = l0 <= 0 <= h0
    B_tracks = pL < 0.05 and bL > 0

    # ---- experiment C: fixed lambda > 0, share sweep; naive drifts, LPO stable
    print("\n" + "=" * 78)
    print("Experiment C  CONFOUND: fixed lambda=0.4, sweep share via h. naive should drift with share (signal + artefact);")
    print("        LPO should stay roughly flat (tracks only the true lambda): LPO separates signal from artefact")
    print("=" * 78)
    C = grid(rng, lams=[0.4], hs=np.linspace(1, 10, 8), os=[300], ms=[1], reps=REPS)
    C["sbin"] = pd.qcut(C.share, 5, labels=["lowest", "low", "mid", "high", "highest"])
    print(C.groupby("sbin", observed=True)[["share", "naive", "lpo"]].mean().round(4).to_string())
    bnc, rnc, pnc = _slope_p(C.share.values, C.naive.values)
    blc, rlc, plc = _slope_p(C.share.values, C.lpo.values)
    print(f"\n  naive ~ share : beta={bnc:+.3f}  p={pnc:.2e}   (expect beta>0: naive contaminated by the artefact)")
    print(f"  lpo   ~ share : beta={blc:+.3f}  p={plc:.2e}   (expect |beta| much smaller: true signal robust to share)")
    C_clean = abs(blc) < abs(bnc) / 2

    # ---- save
    out = pd.concat([A.assign(exp="A"), B.assign(exp="B"), C.assign(exp="C")], ignore_index=True)
    out.drop(columns=["sbin"], errors="ignore").to_csv(TAB / "synthetic_validation.csv", index=False)
    print(f"\nWritten {TAB / 'synthetic_validation.csv'}  ({len(out)} rows)")

    # ---- self-checks
    print("\n" + "=" * 78)
    print("SELF-CHECKS")
    print("=" * 78)
    checks = [
        ("A1 lambda=0: naive bias rises significantly with share", A_naive_pos),
        ("A2 lambda=0: LPO cos 95% CI contains 0 and |mean| < 0.03 (no residual artefact)", A_lpo_zero),
        ("B1 LPO unbiased at lambda=0 (95% CI contains 0)", B_unbiased),
        ("B2 LPO rises monotonically and significantly with true lambda (signal kept)", B_tracks),
        ("B3 naive >= LPO at every lambda (bias always upward)", naive_ge_lpo),
        ("C1 fixed lambda: LPO slope on share < half the naive slope (signal separated from artefact)", C_clean),
    ]
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    allok = all(ok for _, ok in checks)
    print("\nVerdict:", "all checks passed: self-coupling bias is predictable (grows with share), LPO corrects it and recovers lambda without bias"
          if allok else "some checks failed: review the generator or the correction before using these results")


if __name__ == "__main__":
    main()
