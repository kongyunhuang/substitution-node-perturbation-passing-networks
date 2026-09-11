"""
Invariance of the timing profile under high/low splits of every candidate factor (Figure 4a, 4b).

Volume-residualized reorganization versus substitution minute in equal-count bins, for the full
sample and for each factor split at its median. ratio = RMS gap between the halves over RMS
noise; ratio_cen is the same after mean-removing each half (the interaction test). Also writes
the pass-volume artifact and mediation-plane tables read by fig4.py.
Inputs: design matrix from fig4_mechanism_nulls.build_design, results/tables/mechanism_mine.csv
Outputs: results/tables/{invariance_curves,invariance_collapse,volume_artifact,volume_artifact_points,mediation_plane}.csv
Run: python code/figures/fig4_invariance_curves.py
"""

import importlib.util
import warnings
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]  # <repo>/code

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TAB = ROOT / "results" / "tables"
NB = 6                      # equal-count minute bins


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, CODE_ROOT / {"fig4_mechanism_nulls.py": "figures", "mechanism_screen.py": "analysis"}[file] / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    m71 = load("m71", "fig4_mechanism_nulls.py")
    m25 = load("m25", "mechanism_screen.py")
    frozen = pd.read_csv(TAB / "mechanism_mine.csv")
    PRED = list(frozen.predictor)
    M = m71.build_design(m25)

    # volume-residualized reorganization, as in mechanism_screen.py
    npz = m71.zc(M.npass)
    Xv = np.column_stack([np.ones(len(M)), npz, npz ** 2,
                          m71.zc(1 / np.sqrt(M.npass)), m71.zc(np.log(M.npass))])
    y = m71.zc(M.reorg)
    M["resp"] = y - Xv @ np.linalg.lstsq(Xv, y, rcond=None)[0]

    # equal-count minute bins
    M["mbin"] = pd.qcut(M.minute, NB, labels=False, duplicates="drop")
    binmid = M.groupby("mbin").minute.median()
    print(f"batches {len(M)} | minute bins (equal-count, medians): "
          f"{', '.join(f'{v:.0f}' for v in binmid.values)}")
    print(f"per-bin n: {', '.join(str(int(v)) for v in M.groupby('mbin').size().values)}\n")

    rows, coll = [], []
    # overall curve
    for b_, g in M.groupby("mbin"):
        rows.append(dict(factor="__all__", split="all", bin=int(b_),
                         minute_mid=float(binmid[b_]), n=len(g),
                         mean=float(g.resp.mean()),
                         se=float(g.resp.std(ddof=1) / np.sqrt(len(g)))))
    over = pd.DataFrame(rows)
    print("overall response by minute:")
    for _, r in over.iterrows():
        print(f"  {r.minute_mid:>4.0f}'  n={int(r.n):>4}  mean {r['mean']:+.3f} ± {r.se:.3f}")
    span = over["mean"].max() - over["mean"].min()
    print(f"  span across bins = {span:.3f} SD, mean SE = {over.se.mean():.3f}\n")

    for p in PRED:
        v = M[p].astype(float)
        if v.dropna().nunique() <= 2:                      # binary factor: split on value
            hi = v > v.dropna().median()
        else:
            hi = v > v.median()
        gaps, refs = [], []
        for lab, mask in [("high", hi), ("low", ~hi)]:
            for b_, g in M[mask.fillna(False)].groupby("mbin"):
                if len(g) < 15:
                    continue
                rows.append(dict(factor=p, split=lab, bin=int(b_),
                                 minute_mid=float(binmid[b_]), n=len(g),
                                 mean=float(g.resp.mean()),
                                 se=float(g.resp.std(ddof=1) / np.sqrt(len(g)))))
        d = pd.DataFrame([r for r in rows if r["factor"] == p])
        for b_ in d.bin.unique():
            s = d[d.bin == b_]
            if len(s) == 2:
                gaps.append(float(s["mean"].iloc[0] - s["mean"].iloc[1]))
                refs.append(float(np.sqrt((s.se ** 2).sum())))
        if gaps:
            grms, rrms = float(np.sqrt(np.mean(np.square(gaps)))), float(np.sqrt(np.mean(np.square(refs))))
            # mean-removed version: shape only (interaction = 0)
            dd = d.copy()
            for lab in ("high", "low"):
                mk = dd.split == lab
                dd.loc[mk, "cen"] = dd.loc[mk, "mean"] - dd.loc[mk, "mean"].mean()
            gc, rc = [], []
            for b_ in dd.bin.unique():
                s_ = dd[dd.bin == b_]
                if len(s_) == 2:
                    gc.append(float(s_.cen.iloc[0] - s_.cen.iloc[1]))
                    rc.append(float(np.sqrt((s_.se ** 2).sum())))
            gcr = float(np.sqrt(np.mean(np.square(gc)))) if gc else np.nan
            rcr = float(np.sqrt(np.mean(np.square(rc)))) if rc else np.nan
            coll.append(dict(factor=p, gap_rms=grms, ref_rms=rrms, ratio=grms / rrms,
                             gap_cen=gcr, ref_cen=rcr, ratio_cen=gcr / rcr))

    # pass-volume artifact: reorganization vs npass
    R2 = 1 - np.var(M.resp) / np.var(m71.zc(M.reorg))
    M["vbin"] = pd.qcut(M.npass, 10, labels=False, duplicates="drop")
    va = M.groupby("vbin").agg(npass_med=("npass", "median"),
                               reorg_mean=("reorg", "mean"),
                               reorg_se=("reorg", lambda s: s.std(ddof=1) / np.sqrt(len(s))),
                               n=("reorg", "size")).reset_index()
    va["r2_nonlinear"] = R2
    va.to_csv(TAB / "volume_artifact.csv", index=False)
    # per-batch points
    M[["batch_id", "npass", "reorg"]].to_csv(TAB / "volume_artifact_points.csv", index=False)
    print(f"\n[volume artifact] R^2 of the nonlinear volume control = {R2:.3f}; "
          f"mean reorg from {va.reorg_mean.iloc[0]:.3f} (npass~{va.npass_med.iloc[0]:.0f}) "
          f"to {va.reorg_mean.iloc[-1]:.3f} (npass~{va.npass_med.iloc[-1]:.0f})")

    # mediation plane: correlation with minute x own effect
    from scipy.stats import norm as _norm
    med = []
    for p_ in PRED:
        v = M[p_].astype(float)
        ok_ = np.isfinite(v)
        fr_ = frozen[frozen.predictor == p_].iloc[0]
        med.append(dict(predictor=p_,
                        corr_minute=float(np.corrcoef(v[ok_], M.minute[ok_])[0, 1]),
                        z_main=float(_norm.isf(fr_.p_main / 2) * np.sign(fr_.beta_main)),
                        q_main=float(fr_.q_main)))
    MED = pd.DataFrame(med)
    MED.to_csv(TAB / "mediation_plane.csv", index=False)
    cand = MED[(MED.corr_minute.abs() > 0.2) & (MED.z_main.abs() > 1.96) &
               (~MED.predictor.isin(["minute", "t_remain"]))]
    print(f"\n[mediation plane] median |corr_minute| {MED.corr_minute.abs().median():.3f}; "
          f"factors in the candidate corner (|r|>0.2 and |z|>1.96, timing itself excluded): "
          f"{', '.join(cand.predictor) if len(cand) else 'none'}"
          f"{' (q=' + f'{cand.q_main.iloc[0]:.2f}' + ', not significant after FDR)' if len(cand) else ''}")

    C = pd.DataFrame(coll).sort_values("ratio", ascending=False)
    pd.DataFrame(rows).to_csv(TAB / "invariance_curves.csv", index=False)
    C.to_csv(TAB / "invariance_collapse.csv", index=False)
    print("collapse ratio (gap/noise; <1 below noise, >2 stratified), top 8:")
    print(C.head(8).round(3).to_string(index=False))
    print(f"\n  factors with ratio < 1: {(C.ratio < 1).sum()}/{len(C)} | "
          f"1-2: {((C.ratio >= 1) & (C.ratio < 2)).sum()} | >2: {(C.ratio >= 2).sum()}")
    print(f"  median ratio = {C.ratio.median():.2f}")
    print(f"\n  mean-removed ratio_cen (interaction): <1 {(C.ratio_cen < 1).sum()}/{len(C)} | "
          f"1-2 {((C.ratio_cen >= 1) & (C.ratio_cen < 2)).sum()} | >=2 {(C.ratio_cen >= 2).sum()} | "
          f"median {C.ratio_cen.median():.2f} | max {C.ratio_cen.max():.2f} "
          f"({C.sort_values('ratio_cen').factor.iloc[-1]})")
    print("saved invariance_curves.csv + invariance_collapse.csv")


if __name__ == "__main__":
    main()
