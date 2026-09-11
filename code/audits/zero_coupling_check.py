"""Does a zero-coupling generator reproduce the residual share slope after correction?

Feeds the empirical (share, volume, n_subs, match_id) per batch into the generator
with lam=0, then regresses lpo (and naive) on share_z + logvol_z, mixedlm by match,
same spec as bias_predictability.py. Seeds 20260808..20260810, plus a pooled fit.
Inputs:  results/tables/bias_predictability.csv, data/substitution_batches.csv
Outputs: results/tables/lambda0_share.csv
Run: python code/audits/zero_coupling_check.py
"""

import importlib.util
import warnings
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]  # <repo>/code

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
TAB = ROOT / "results" / "tables"
SEEDS = [20260808, 20260809, 20260810]
SPECS = {"o=2*vol(touch)": 2, "o=vol(pass)": 1}  # event count: touches (main) / passes (sensitivity)

# synthetic generator
_spec = importlib.util.spec_from_file_location(
    "gen40", str(CODE_ROOT / "synthetic" / "synthetic_generator.py"))
gen40 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen40)
N_TEAM = gen40.N_TEAM  # 14


def zc(s):
    return (s - s.mean()) / s.std()


def fit_38_spec(df, dv):
    """bias_predictability spec: mixedlm dv ~ share_z + logvol_z, groups=match_id, ML; OLS too."""
    d = df.dropna(subset=[dv]).copy()
    d["share_z"] = zc(d["share_obs"])
    d["logvol_z"] = zc(np.log(d["volume"]))
    mm = smf.mixedlm(f"{dv} ~ share_z + logvol_z", d, groups=d["match_id"]).fit(reml=False)
    ols = smf.ols(f"{dv} ~ share_z + logvol_z", d).fit()
    return dict(n=len(d),
                beta_mixed=mm.params["share_z"], se_mixed=mm.bse["share_z"], p_mixed=mm.pvalues["share_z"],
                beta_logvol=mm.params["logvol_z"], p_logvol=mm.pvalues["logvol_z"],
                beta_ols=ols.params["share_z"], p_ols=ols.pvalues["share_z"])


def main():
    # empirical (s, volume, m, match_id) carried per batch, no parametrisation
    M = pd.read_csv(TAB / "bias_predictability.csv").dropna(subset=["naive"])
    B = pd.read_csv(DATA / "substitution_batches.csv").set_index("batch_id")
    M["n_subs"] = B.loc[M.batch_id, "n_subs"].values
    M = M.rename(columns={"perturber_share": "share_obs"})
    print(f"empirical N={len(M)} | share median={M.share_obs.median():.3f} | "
          f"volume median={M.volume.median():.0f} | n_subs: "
          f"{M.n_subs.value_counts().sort_index().to_dict()}")

    rows = []
    # empirical reference rows, expect beta ~ +0.080 / +0.010
    for dv in ["naive", "lpo"]:
        r = fit_38_spec(M, dv)
        rows.append(dict(source="empirical", spec="LaLiga", seed=np.nan, estimator=dv, **r))
        print(f"[empirical] {dv:5s}: beta_mixed={r['beta_mixed']:+.4f} (p={r['p_mixed']:.2e})  "
              f"beta_OLS={r['beta_ols']:+.4f} (p={r['p_ols']:.2e})  n={r['n']}")
    emp_lpo_beta = rows[-1]["beta_mixed"]

    # lam=0 synthesis: per seed x spec, one synthetic batch per empirical batch
    for spec_name, mult in SPECS.items():
        pooled = []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            recs = []
            for t in M.itertuples():
                s, vol, m = float(t.share_obs), int(t.volume), int(t.n_subs)
                h = s * N_TEAM / ((1.0 - s) * m)          # solve E[share]=s
                o = int(mult * vol)                        # event count
                r = gen40.simulate_batch(rng, lam=0.0, h=h, o=o, m=m)
                recs.append(dict(match_id=t.match_id, seed=seed, volume=vol,
                                 share_obs=r["share"], naive=r["naive"], lpo=r["lpo"]))
            S = pd.DataFrame(recs)
            pooled.append(S)
            for dv in ["naive", "lpo"]:
                r = fit_38_spec(S, dv)
                rows.append(dict(source="synthetic_lam0", spec=spec_name, seed=seed,
                                 estimator=dv, **r))
                print(f"[{spec_name:15s} seed={seed}] {dv:5s}: "
                      f"beta_mixed={r['beta_mixed']:+.4f} (p={r['p_mixed']:.2e})  "
                      f"beta_OLS={r['beta_ols']:+.4f} (p={r['p_ols']:.2e})")
        # pool seeds, groups = match_id x seed
        P = pd.concat(pooled, ignore_index=True)
        P["match_id"] = P.match_id.astype(str) + "_s" + P.seed.astype(str)
        for dv in ["naive", "lpo"]:
            r = fit_38_spec(P, dv)
            rows.append(dict(source="synthetic_lam0", spec=spec_name, seed=-1,
                             estimator=dv, **r))
            print(f"[{spec_name:15s} POOLED  ] {dv:5s}: beta_mixed={r['beta_mixed']:+.4f} "
                  f"(p={r['p_mixed']:.2e})  n={r['n']}")

    out = pd.DataFrame(rows).round(6)
    out.to_csv(TAB / "lambda0_share.csv", index=False)
    print(f"\nwrote {TAB / 'lambda0_share.csv'}  ({len(out)} rows)")

    # verdict, pre-specified criteria
    print("\n" + "=" * 78)
    print("verdict (main spec o=2*vol, pooled corrected beta vs empirical +0.0104)")
    print("=" * 78)
    main_pool = out[(out.source == "synthetic_lam0") & (out.spec == "o=2*vol(touch)")
                    & (out.seed == -1) & (out.estimator == "lpo")].iloc[0]
    ctrl_pool = out[(out.source == "synthetic_lam0") & (out.spec == "o=2*vol(touch)")
                    & (out.seed == -1) & (out.estimator == "naive")].iloc[0]
    syn_beta, syn_se = main_pool.beta_mixed, main_pool.se_mixed
    gate_naive = ctrl_pool.beta_mixed > 0.04 and ctrl_pool.p_mixed < 1e-10
    print(f"  positive control naive: synthetic beta={ctrl_pool.beta_mixed:+.4f} vs empirical +0.0795 "
          f"-> {'generator reproduces the naive artefact under empirical (s,m)' if gate_naive else 'calibration failed, lpo verdict not interpretable'}")
    reproduced = (syn_beta > 0) and (0.5 * emp_lpo_beta <= syn_beta <= 2.0 * emp_lpo_beta)
    near_zero = abs(syn_beta) < 0.5 * emp_lpo_beta
    print(f"  corrected lpo : synthetic beta={syn_beta:+.4f} +/- {syn_se:.4f} (p={main_pool.p_mixed:.2e}) "
          f"vs empirical {emp_lpo_beta:+.4f}")
    if not gate_naive:
        verdict = "WARNING: positive control failed; (s,m) -> generator calibration off, experiment cannot decide"
    elif reproduced:
        verdict = ("REPRODUCED: with lam=0 (coupling and teammate response exactly zero) the synthetic beta matches the empirical +0.010 in sign and magnitude "
                   "-> residual slope is a finite-sample artefact (O(1/m) higher-order terms of the normalisation plus asymmetric teammate sample sizes "
                   "from unequal perturber event counts across windows); drop the measurement-precision explanation")
    elif near_zero:
        verdict = ("NOT REPRODUCED: with lam=0 the synthetic beta ~ 0, not the empirical +0.010 -> residual slope not attributable to "
                   "a sampling artefact; list as a limitation of the correction")
    else:
        verdict = ("PARTIAL: same sign, magnitude off by >2x; sampling artefact explains part but not all of the residual slope, "
                   "report as a limitation with these numbers")
    print(f"\n  {verdict}")


if __name__ == "__main__":
    main()
