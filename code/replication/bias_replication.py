"""Naive, null and perturber-excluded cross-layer alignment in 11 competitions.

Per batch: naive cos(dB, dz), random-teammate-pair null (15 draws), and dz with
the substituted players' passes removed. La Liga 2023/24 rows come from the frozen
bias_predictability.csv. Recomputed naive means must match the reference table.
Inputs:  data/opendata/*.parquet, results/tables/{bias_predictability,duallayer_opendata_reference}.csv
Outputs: results/tables/bias_replication.csv
Run: python code/replication/bias_replication.py
"""

import importlib.util
import warnings
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]  # <repo>/code

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TAB = ROOT / "results" / "tables"
ARCH = TAB / "duallayer_opendata_reference.csv"  # reference naive-alignment table from an earlier run, used only as a consistency check
W = 900.0


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def per_comp3(m33, od, cid, sid, rng):
    """One competition: per-batch naive / null / perturber-excluded alignment."""
    df = od.fetch(cid, sid)
    net, _ = od.prep(df)
    ni = {k: g for k, g in net.groupby(["match_id", "team"])}
    match_end = {m: g.minute.max() * 60 + g.second.max() for m, g in df.groupby("match_id")}
    B = m33.batches_named(df)
    reals, nulls, corr = [], [], []
    for b in B:
        key, t0 = (b["match_id"], b["team"]), b["t0"]
        if t0 + W > match_end.get(b["match_id"], 0):
            continue
        g = ni.get(key)
        if g is None:
            continue
        wp = g[(g.tc >= t0 - W) & (g.tc < t0)]
        wq = g[(g.tc > t0) & (g.tc <= t0 + W)]
        if len(wp) < 10 or len(wq) < 10:
            continue
        din = [m33.pdist(wq, p) for p in b["in"]]
        dout = [m33.pdist(wp, p) for p in b["out"]]
        din = [d for d in din if d is not None]
        dout = [d for d in dout if d is not None]
        if not din or not dout:
            continue
        dB = np.mean(din, 0) - np.mean(dout, 0)
        dz = od.zvec(wq) - od.zvec(wp)
        if dz is None:
            continue
        reals.append(m33.cos(dB, dz))
        # null: random pair of unsubstituted teammates, 15 draws
        others = [p for p in wp.player.unique() if p not in b["out"] and p in wq.player.values]
        if len(others) >= 2:
            cs = []
            for _ in range(15):
                a, c = rng.choice(others, 2, replace=False)
                da, dc = m33.pdist(wp, a), m33.pdist(wq, c)
                if da is not None and dc is not None:
                    cs.append(m33.cos(dc - da, dz))
            nulls.append(np.nanmean(cs) if cs else np.nan)
        else:
            nulls.append(np.nan)
        # corrected: drop every pass involving a substituted player
        sw = set(b["out"]) | set(b["in"])
        wpx = wp[~wp.player.isin(sw) & ~wp.pass_recipient.isin(sw)]
        wqx = wq[~wq.player.isin(sw) & ~wq.pass_recipient.isin(sw)]
        if len(wpx) < 10 or len(wqx) < 10:
            corr.append(np.nan)
            continue
        zp, zq = od.zvec(wpx), od.zvec(wqx)
        corr.append(m33.cos(dB, zq - zp) if (zp is not None and zq is not None) else np.nan)
    return np.array(reals), np.array(nulls), np.array(corr)


def main():
    od = load("od30", CODE_ROOT / "replication" / "fetch_open_data.py")
    m33 = load("m33", CODE_ROOT / "replication" / "open_data_two_layer.py")
    rng = np.random.default_rng(20260613)          # same seed as open_data_two_layer.py
    rows = []
    print(f"{'comp':10s}{'n':>5s}{'naive':>9s}{'null':>9s}{'corrected':>9s}")
    for name, (cid, sid) in m33.COMPS.items():
        r, nu, c = per_comp3(m33, od, cid, sid, rng)
        if len(r) < 30:
            print(f"{name:10s} insufficient sample ({len(r)})")
            continue
        cc = c[np.isfinite(c)]
        rows.append(dict(comp=name, n=len(r),
                         naive=float(np.nanmean(r)),
                         naive_se=float(np.nanstd(r, ddof=1) / np.sqrt(np.isfinite(r).sum())),
                         null=float(np.nanmean(nu)),
                         null_se=float(np.nanstd(nu, ddof=1) / np.sqrt(np.isfinite(nu).sum())),
                         corrected=float(cc.mean()),
                         corrected_se=float(cc.std(ddof=1) / np.sqrt(len(cc))),
                         p_corrected=float(ttest_1samp(cc, 0).pvalue),
                         source="open data (recomputed 2026-07-26)"))
        print(f"{name:10s}{len(r):5d}{rows[-1]['naive']:+9.3f}{rows[-1]['null']:+9.3f}"
              f"{rows[-1]['corrected']:+9.3f}")

    # La Liga 2023/24: frozen table, not recomputed
    bp = pd.read_csv(TAB / "bias_predictability.csv")
    rows.append(dict(comp="LaLiga23", n=len(bp),
                     naive=float(bp.naive.mean()),
                     naive_se=float(bp.naive.std(ddof=1) / np.sqrt(len(bp))),
                     null=np.nan, null_se=np.nan,
                     corrected=float(bp.lpo.mean()),
                     corrected_se=float(bp.lpo.std(ddof=1) / np.sqrt(len(bp))),
                     p_corrected=float(ttest_1samp(bp.lpo, 0).pvalue),
                     source="frozen bias_predictability.csv (main dataset)"))
    print(f"{'LaLiga23':10s}{len(bp):5d}{rows[-1]['naive']:+9.3f}{'--':>9s}"
          f"{rows[-1]['corrected']:+9.3f}")

    R = pd.DataFrame(rows)

    # gate: naive means must match the reference table
    old = pd.read_csv(ARCH).set_index("comp")
    d = R[R.comp != "LaLiga23"].set_index("comp")
    diff = (d.naive - old.real).abs()
    assert diff.max() < 0.01, \
        f"naive alignment differs from the reference table (max diff {diff.max():.4f}):\n{(d.naive - old.real).round(4)}"
    print(f"\n  gate passed: naive alignment matches the reference table in 10 competitions (max diff {diff.max():.4f})")
    print(f"  naive positive in {(R.naive > 0).sum()}/{len(R)} competitions (mean {R.naive.mean():+.3f}); "
          f"corrected indistinguishable from zero in {(R.p_corrected > 0.05).sum()}/{len(R)} competitions")
    R.to_csv(TAB / "bias_replication.csv", index=False)
    print("saved bias_replication.csv")


if __name__ == "__main__":
    main()
