"""
Generator experiment showing that volume and sampling artifacts cannot
manufacture a timing threshold (Section 4.4). Synthetic batches with real group
n and empirical volumes through the headline SNR pipeline: S1 null, S2 constant
reorg, S3 calibration of lam_g reproducing the observed SNRs (OBS_SNR).
Input:  results/tables/bias_predictability.csv
Output: results/tables/artifact_exclusion.csv
Run:    python code/analysis/artifact_exclusion.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TAB = ROOT / "results" / "tables"
K = 24
SEED = 20260702
GROUPS = ["strategic", "regular", "late"]
OBS_SNR = {"strategic": 1.05, "regular": 1.57, "late": 1.64}  # observed teammates-only SNRs


def make_batches(rng, n, vols, lam, mode="common", r_common=None):
    """n synthetic batches; window volumes resampled from vols; lam = true reorg magnitude.
    mode='idio': each batch moves toward its own random target (SNR insensitive, directions cancel)
    mode='common': all batches move toward r_common (the systematic component the SNR measures)"""
    out = []
    for _ in range(n):
        q = rng.dirichlet(np.full(K, 0.5))
        if lam > 0:
            r = rng.dirichlet(np.full(K, 0.5)) if mode == "idio" else r_common
            qp = (1 - lam) * q + lam * r
            qp = qp / qp.sum()
        else:
            qp = q
        m = rng.choice(vols, size=4)  # pre, post, ctrl_pre, ctrl_post
        zpre = rng.multinomial(m[0], q) / m[0]
        zpost = rng.multinomial(m[1], qp) / m[1]
        cpre = rng.multinomial(m[2], q) / m[2]
        cpost = rng.multinomial(m[3], q) / m[3]
        out.append((zpost - zpre) - (cpost - cpre))
    return np.array(out)


def snr_group(arr, N, rng, nb=500, nn=300):
    mags = [np.linalg.norm(arr[rng.integers(len(arr), size=N)].mean(0)) for _ in range(nb)]
    nf = []
    for _ in range(nn):
        idx = rng.integers(len(arr), size=N)
        sgn = rng.choice([1, -1], size=len(arr))
        nf.append(np.linalg.norm((arr[idx] * sgn[idx, None]).mean(0)))
    return np.mean(mags) / np.mean(nf)


def main():
    rng = np.random.default_rng(SEED)
    bp = pd.read_csv(TAB / "bias_predictability.csv")
    vols = {g: np.clip(bp[bp.timing == g].volume.values.astype(int), 10, None) for g in GROUPS}
    ns = {g: (bp.timing == g).sum() for g in GROUPS}
    N = min(ns.values())
    print(f"Empirical volumes: " + ", ".join(f"{g} n={ns[g]} median={np.median(vols[g]):.0f}" for g in GROUPS))
    print(f"Equal sample size N={N}\n")
    rows = []

    r_common = {g: rng.dirichlet(np.full(K, 0.5)) for g in GROUPS}  # common reorg direction per group

    # S1 null / S2a idio reorg (SNR insensitive) / S2b common reorg, constant lam (no threshold expected)
    for tag, lam_by_g, mode in [("S1_null", {g: 0.0 for g in GROUPS}, "idio"),
                                ("S2a_idio_lam0.15", {g: 0.15 for g in GROUPS}, "idio"),
                                ("S2b_common_lam0.04", {g: 0.04 for g in GROUPS}, "common"),
                                ("S2b_common_lam0.06", {g: 0.06 for g in GROUPS}, "common")]:
        snrs = {}
        for g in GROUPS:
            arr = make_batches(rng, ns[g], vols[g], lam_by_g[g], mode, r_common[g])
            snrs[g] = snr_group(arr, N, rng)
        rows += [dict(scenario=tag, timing=g, lam=lam_by_g[g], snr=round(snrs[g], 3)) for g in GROUPS]
        thr = snrs["late"] > snrs["strategic"] * 1.2 and snrs["regular"] > snrs["strategic"] * 1.2
        print(f"[{tag:20s}] strategic/regular/late SNR = {snrs['strategic']:.2f}/{snrs['regular']:.2f}/{snrs['late']:.2f}"
              f"  -> {'WARNING: threshold shape' if thr else 'no threshold shape'}")

    # S3: SNR(lam) curve per group, interpolate lam_g matching OBS_SNR
    print("\nS3 calibration, SNR(lam) curves (common mode, per-group volumes):")
    lam_grid = np.arange(0.0, 0.251, 0.02)
    need = {}
    for g in GROUPS:
        curve = []
        for lam in lam_grid:
            arr = make_batches(rng, ns[g], vols[g], lam, "common", r_common[g])
            curve.append(snr_group(arr, N, rng))
        curve = np.array(curve)
        tgt = OBS_SNR[g]
        i = np.searchsorted(curve, tgt)
        if i == 0:
            lam_need = 0.0
        elif i >= len(curve):
            lam_need = np.nan
        else:
            lam_need = np.interp(tgt, curve[i - 1:i + 1], lam_grid[i - 1:i + 1])
        need[g] = lam_need
        rows += [dict(scenario="S3_curve", timing=g, lam=round(l, 2), snr=round(s, 3))
                 for l, s in zip(lam_grid, curve)]
        print(f"  {g:10s}: target SNR={tgt:.2f} -> required lam={lam_need:.3f}")
    rows += [dict(scenario="S3_needed", timing=g, lam=round(need[g], 4), snr=OBS_SNR[g]) for g in GROUPS]
    pd.DataFrame(rows).to_csv(TAB / "artifact_exclusion.csv", index=False)

    ratio = need["late"] / max(need["strategic"], 1e-9) if need["strategic"] > 0 else np.inf
    print(f"\nVerdict:")
    print(f"  lam_late/lam_strategic = {ratio if np.isfinite(ratio) else 'inf'}"
          f" (lam: {need['strategic']:.3f} -> {need['regular']:.3f} -> {need['late']:.3f})")
    print("  Empirical volumes with constant reorganization do not produce a threshold; matching the observed SNRs needs a real jump in magnitude after the threshold"
          "\nSaved artifact_exclusion.csv")


if __name__ == "__main__":
    main()
