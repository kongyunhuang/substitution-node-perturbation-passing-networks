"""
Closed form of the shared-activity bias at three levels vs simulation (Appendix B.2, Table B.1).
L1 unequal shares: mu = s_bar dB + ds (f_bar - q). L2 dB estimated from the same events:
exact identity E<dB_hat, dz_hat> = s E|dB_hat|^2 with E|dB_hat|^2 = b^2 + (1 - |f_in|^2)/n_in
+ (1 - |f_out|^2)/n_out, so the bias stays positive even at b = 0.
Inputs: none
Outputs: results/tables/closedform_full.csv
Run: python code/synthetic/closed_form_verification.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TAB = ROOT / "results" / "tables"
K = 24
SEED = 20260702
rng = np.random.default_rng(SEED)


def sigma(p):
    return np.diag(p) - np.outer(p, p)


# ---------------- L1: unequal shares, known profiles ----------------
def l1_theory(q, fi, fo, si, so, m):
    dB = fi - fo
    b = np.linalg.norm(dB)
    u = dB / b
    fbar = (fi + fo) / 2
    mu = (si + so) / 2 * dB + (si - so) * (fbar - q)
    zpost = si * fi + (1 - si) * q
    zpre = so * fo + (1 - so) * q
    C = sigma(zpost) / m + sigma(zpre) / m
    a_par = u @ mu
    mu_perp2 = mu @ mu - a_par ** 2
    Vtot = np.trace(C)
    Vpar = u @ C @ u
    return a_par / np.sqrt(a_par ** 2 + mu_perp2 + (Vtot - Vpar))


def l1_sim(q, fi, fo, si, so, m, reps):
    dB = fi - fo
    b = np.linalg.norm(dB)
    mix_post = si * fi + (1 - si) * q
    mix_pre = so * fo + (1 - so) * q
    cs = []
    for _ in range(reps):
        dz = rng.multinomial(m, mix_post) / m - rng.multinomial(m, mix_pre) / m
        nz = np.linalg.norm(dz)
        if nz > 0:
            cs.append(dz @ dB / (nz * b))
    return np.mean(cs)


# ---------------- L2: dB estimated from the same events ----------------
def l2_theory(q, fi, fo, s, m):
    n_i = max(int(round(s * m)), 2)
    n_o = n_i
    b2 = float((fi - fo) @ (fi - fo))
    B2 = b2 + (1 - fi @ fi) / n_i + (1 - fo @ fo) / n_o   # E|dB_hat|^2
    # total variance of dz_hat, perturber + teammates, both windows
    zpost = s * fi + (1 - s) * q
    zpre = s * fo + (1 - s) * q
    Vtot = np.trace(sigma(zpost)) / m + np.trace(sigma(zpre)) / m
    a = s * B2  # E<dB_hat, dz_hat> = s E|dB_hat|^2, exact
    # leading order: cos ~ a / (sqrt(B2) sqrt(s^2 B2 + Vperp)), Vperp ~ Vtot - s^2 B2
    Vperp = max(Vtot - s * s * B2, 1e-12)
    return a / (np.sqrt(B2) * np.sqrt(s * s * B2 + Vperp)), B2


def l2_sim(q, fi, fo, s, m, reps):
    cs, covs, B2s = [], [], []
    for _ in range(reps):
        n_i = rng.binomial(m, s)
        n_o = rng.binomial(m, s)
        if n_i < 2 or n_o < 2:
            continue
        h_in = rng.multinomial(n_i, fi)
        h_out = rng.multinomial(n_o, fo)
        hR_post = rng.multinomial(m - n_i, q)
        hR_pre = rng.multinomial(m - n_o, q)
        fhat_i = h_in / n_i
        fhat_o = h_out / n_o
        dB = fhat_i - fhat_o                       # estimated footprint change, shares h_in with z_post
        dz = (h_in + hR_post) / m - (h_out + hR_pre) / m
        nb, nz = np.linalg.norm(dB), np.linalg.norm(dz)
        if nb > 0 and nz > 0:
            cs.append(dB @ dz / (nb * nz))
            covs.append(dB @ dz)
            B2s.append(nb * nb)
    return np.mean(cs), np.mean(covs), np.mean(B2s)


def main():
    rows = []
    REPS = 4000
    q = rng.dirichlet(np.full(K, 0.5))
    fi = rng.dirichlet(np.full(K, 0.5))
    fo = rng.dirichlet(np.full(K, 0.5))

    print("===== L1 unequal shares (known profiles) =====")
    for m in [100, 300]:
        for si, so in [(0.10, 0.20), (0.20, 0.10), (0.05, 0.25), (0.30, 0.15)]:
            th = l1_theory(q, fi, fo, si, so, m)
            sm = l1_sim(q, fi, fo, si, so, m, REPS)
            rel = abs(sm - th) / max(abs(sm), 1e-9)
            rows.append(dict(level="L1", m=m, s_in=si, s_out=so, b0=0,
                             theory=round(th, 4), sim=round(sm, 4), rel_err=round(rel, 4)))
            print(f"  m={m} s_in={si} s_out={so}: theory={th:+.3f} sim={sm:+.3f} rel={rel:.3f}")

    print("\n===== L2 dB estimated from the same events (circular bias) =====")
    for m in [60, 150, 400]:
        for s in [0.08, 0.15, 0.25]:
            th, B2 = l2_theory(q, fi, fo, s, m)
            sm, cov_sim, B2_sim = l2_sim(q, fi, fo, s, m, REPS)
            # identity check: E<dB_hat, dz_hat> == s E|dB_hat|^2
            id_lhs, id_rhs = cov_sim, s * B2_sim
            id_err = abs(id_lhs - id_rhs) / abs(id_rhs)
            rel = abs(sm - th) / max(abs(sm), 1e-9)
            rows.append(dict(level="L2", m=m, s_in=s, s_out=s, b0=0,
                             theory=round(th, 4), sim=round(sm, 4), rel_err=round(rel, 4),
                             ident_err=round(id_err, 4)))
            print(f"  m={m} s={s}: cos theory={th:.3f} sim={sm:.3f} rel={rel:.3f} | "
                  f"identity E<dB,dz>={id_lhs:.5f} vs s*E|dB|^2={id_rhs:.5f} (err={id_err:.3f})")

    print("\n===== L2 special case: b=0 (f_in=f_out) still gives positive bias =====")
    for m in [60, 150, 400]:
        for s in [0.08, 0.15, 0.25]:
            th, _ = l2_theory(q, fi, fi, s, m)
            sm, cov_sim, _ = l2_sim(q, fi, fi, s, m, REPS)
            rows.append(dict(level="L2_b0", m=m, s_in=s, s_out=s, b0=1,
                             theory=round(th, 4), sim=round(sm, 4),
                             rel_err=round(abs(sm - th) / max(abs(sm), 1e-9), 4)))
            print(f"  m={m} s={s}: theory={th:.3f} sim={sm:.3f} "
                  f"{'bias>0' if sm > 0.02 else '~0'}")

    df = pd.DataFrame(rows)
    df.to_csv(TAB / "closedform_full.csv", index=False)
    ok1 = (df[df.level == "L1"].rel_err < 0.10).all()
    ok2 = (df[df.level == "L2"].rel_err < 0.15).all()
    okid = (df[df.level == "L2"].ident_err < 0.05).all()
    okb0 = (df[df.level == "L2_b0"].sim > 0).all()
    print(f"\nVerdict: L1 closed form {'OK' if ok1 else 'WARN'} | L2 closed form {'OK' if ok2 else 'WARN'} | "
          f"identity (exact) {'OK' if okid else 'FAIL'} | b=0 positive bias {'OK' if okb0 else 'FAIL'}")
    print("Table saved: closedform_full.csv")


if __name__ == "__main__":
    main()
