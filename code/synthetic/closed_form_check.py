"""
Closed-form expectation of the naive alignment vs controlled simulation (Appendix B, Table B.1).
Null model: teammate profile q, footprints f_out / f_in, share s, window size m.
E[dz] = s dB exactly; E[cos] ~ s b / sqrt(s^2 b^2 + V_perp), V_perp = tr(C) - u'Cu,
C = Sigma(z_post)/m_post + Sigma(z_pre)/m_pre, Sigma(p) = diag(p) - pp'; LPO mean ~ 0.
Sweeps (s, m, K) and reports relative errors.
Inputs: none
Outputs: results/tables/mvp_closedform.csv
Run: python code/synthetic/closed_form_check.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TAB = ROOT / "results" / "tables"
SEED = 20260702
rng = np.random.default_rng(SEED)


def sigma(p):
    return np.diag(p) - np.outer(p, p)


def theory_cos(q, fi, fo, s, m_pre, m_post):
    dB = fi - fo
    b = np.linalg.norm(dB)
    u = dB / b
    zpost = s * fi + (1 - s) * q
    zpre = s * fo + (1 - s) * q
    C = sigma(zpost) / m_post + sigma(zpre) / m_pre
    Vtot = np.trace(C)
    Vpar = u @ C @ u
    Vperp = Vtot - Vpar
    a = s * b
    return a / np.sqrt(a * a + Vperp), b, Vperp


def simulate(q, fi, fo, s, m_pre, m_post, reps):
    K = len(q)
    dB = fi - fo
    b = np.linalg.norm(dB)
    mix_post = s * fi + (1 - s) * q
    mix_pre = s * fo + (1 - s) * q
    cos_n, cos_l = [], []
    for _ in range(reps):
        # naive: draw from the mixture (same as per-event binomial assignment)
        zq = rng.multinomial(m_post, mix_post) / m_post
        zp = rng.multinomial(m_pre, mix_pre) / m_pre
        dz = zq - zp
        nz = np.linalg.norm(dz)
        if nz > 0:
            cos_n.append(dz @ dB / (nz * b))
        # LPO: teammate events only, (1 - s) m draws from q
        mq = max(int(round((1 - s) * m_post)), 1)
        mp = max(int(round((1 - s) * m_pre)), 1)
        zq2 = rng.multinomial(mq, q) / mq
        zp2 = rng.multinomial(mp, q) / mp
        dz2 = zq2 - zp2
        nz2 = np.linalg.norm(dz2)
        if nz2 > 0:
            cos_l.append(dz2 @ dB / (nz2 * b))
    return np.mean(cos_n), np.std(cos_n) / np.sqrt(len(cos_n)), np.mean(cos_l), np.std(cos_l) / np.sqrt(len(cos_l))


def main():
    rows = []
    REPS = 3000
    for K in [8, 24]:
        # one fixed profile set per K
        q = rng.dirichlet(np.full(K, 0.5))
        fi = rng.dirichlet(np.full(K, 0.5))
        fo = rng.dirichlet(np.full(K, 0.5))
        for m in [40, 100, 300]:
            for s in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]:
                th, b, Vperp = theory_cos(q, fi, fo, s, m, m)
                sim, se, lpo, lpo_se = simulate(q, fi, fo, s, m, m, REPS)
                rel = abs(sim - th) / max(abs(sim), 1e-9)
                rows.append(dict(K=K, m=m, s=s, b=round(b, 3),
                                 snr_signal=round(s * b / np.sqrt(Vperp), 2),
                                 theory=round(th, 4), sim=round(sim, 4),
                                 sim_se=round(se, 4), rel_err=round(rel, 4),
                                 lpo=round(lpo, 4), lpo_se=round(lpo_se, 4)))
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "mvp_closedform.csv", index=False)
    print(df.to_string(index=False))

    # verdict
    core = df[df.snr_signal >= 0.3]  # signal not swamped by noise
    ok_fit = (core.rel_err < 0.10).mean()
    ok_lpo = (df.lpo.abs() < 3 * df.lpo_se + 0.02).mean()
    print(f"\nMain region (signal/noise >= 0.3, n={len(core)}): share with rel_err < 10% = {ok_fit:.0%}, "
          f"max rel_err={core.rel_err.max():.3f}")
    print(f"LPO ~ 0 (tolerance 3 se + 0.02): {ok_lpo:.0%}")
    # small s: theory ~ (b / sqrt(Vperp)) s, so theory / s should be near constant
    print("\nSmall-signal linearity (theory/s, should be near constant):")
    for (K, m), g in df[df.s <= 0.15].groupby(["K", "m"]):
        r = (g.theory / g.s)
        print(f"  K={K} m={m}: theory/s = {[round(x,2) for x in r]}  (max/min={r.max()/r.min():.2f})")
    verdict = "closed form holds" if ok_fit > 0.9 and ok_lpo > 0.9 else "deviations present, see table"
    print(f"\nVerdict: {verdict}")


if __name__ == "__main__":
    main()
