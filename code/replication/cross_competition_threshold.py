"""Cross-competition timing threshold under the teammates-only exclusion.

Per competition: batch-level zone DiD vs a substitution-free control window,
with and without the substituted players' own passes; equal-sample bootstrap
SNR per timing group; binomial sign test of late > early across competitions.
Inputs:  data/opendata/{cid}_{sid}.parquet
Outputs: results/tables/cross_gradient_clean.csv
Run: python code/replication/cross_competition_threshold.py
"""

import importlib.util
import warnings
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]  # <repo>/code

import numpy as np
import pandas as pd
from scipy.stats import binomtest

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
OD = ROOT / "data" / "opendata"
TAB = ROOT / "results" / "tables"
NX, NY = 6, 4
PL, PW = 120, 80
W = 900.0

spec = importlib.util.spec_from_file_location("od33", CODE_ROOT / "replication" / "open_data_two_layer.py")
m33 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m33)
od = m33.od

COMPS = {"WC22": (43, 106), "Euro24": (55, 282), "Euro20": (55, 43), "ISL": (1238, 108),
         "WSL": (37, 90), "Bund23": (9, 281), "Bund15": (9, 27), "LaLiga15": (11, 27),
         "PL15": (2, 27), "SerieA15": (12, 27)}


def zvec_excl(w, names):
    if names is not None:
        w = w[~w.player.isin(names) & ~w.pass_recipient.isin(names)]
    return od.zvec(w)


def did_lists(cid, sid):
    df = od.fetch(cid, sid)
    net, _ = od.prep(df)
    ni = {k: g.sort_values("tc") for k, g in net.groupby(["match_id", "team"])}
    keys = list(ni.keys())
    sub_times = {k: np.sort(g.minute.values * 60.0)
                 for k, g in df[df.type == "Substitution"].groupby(["match_id", "team"])}
    match_end = {m: g.minute.max() * 60 + g.second.max() for m, g in df.groupby("match_id")}
    B = m33.batches_named(df)
    rng = np.random.default_rng(od.SEED)
    incl = {"strategic": [], "regular": [], "late": []}
    excl = {"strategic": [], "regular": [], "late": []}
    for b in B:
        key, t0 = (b["match_id"], b["team"]), b["t0"]
        if t0 + W > match_end.get(b["match_id"], 0):
            continue
        g = ni.get(key)
        if g is None:
            continue
        wp = g[(g.tc >= t0 - W) & (g.tc < t0)]
        wq = g[(g.tc > t0) & (g.tc <= t0 + W)]
        swapped = list(b["out"]) + list(b["in"])
        zip_, ziq = od.zvec(wp), od.zvec(wq)
        zep, zeq = zvec_excl(wp, swapped), zvec_excl(wq, swapped)
        if any(v is None for v in (zip_, ziq, zep, zeq)):
            continue
        # matched control window
        cands = [k for k in keys if k != key]
        rng.shuffle(cands)
        ctrl = None
        for ck in cands[:200]:
            for off in [0, 60, -60, 120, -120]:
                tc = t0 + off
                if tc - W < 0 or tc + W > match_end.get(ck[0], 0):
                    continue
                if not od.clean(sub_times, ck, tc - W, tc + W):
                    continue
                gc = ni.get(ck)
                cp = gc[(gc.tc >= tc - W) & (gc.tc < tc)]
                cq = gc[(gc.tc > tc) & (gc.tc <= tc + W)]
                if od.zvec(cp) is not None and od.zvec(cq) is not None:
                    ctrl = od.zvec(cq) - od.zvec(cp); break
            if ctrl is not None:
                break
        if ctrl is None:
            continue
        tg = "strategic" if b["minute"] <= 60 else ("regular" if b["minute"] <= 75 else "late")
        incl[tg].append((ziq - zip_) - ctrl)
        excl[tg].append((zeq - zep) - ctrl)
    return ({g: np.array(incl[g]) for g in incl}, {g: np.array(excl[g]) for g in excl})


def snr_group(arr, N, rng):
    if len(arr) < 10:
        return np.nan, np.nan
    mags = [np.linalg.norm(arr[rng.integers(len(arr), size=N)].mean(0)) for _ in range(500)]
    nf = []
    for _ in range(300):
        idx = rng.integers(len(arr), size=N)
        sgn = rng.choice([1, -1], size=len(arr))
        nf.append(np.linalg.norm((arr[idx] * sgn[idx, None]).mean(0)))
    obs = np.mean(mags)
    return obs / np.mean(nf), 1 - (obs > np.array(nf)).mean()


def main():
    rng = np.random.default_rng(od.SEED)
    rows = []
    late_gt_incl, late_gt_excl = [], []
    print(f"{'comp':10s} | {'incl early/mid/late SNR':>22s} | {'excl early/mid/late SNR':>22s} | late>early")
    for name, (cid, sid) in COMPS.items():
        if not (OD / f"{cid}_{sid}.parquet").exists():
            continue
        incl, excl = did_lists(cid, sid)
        ns = {g: len(incl[g]) for g in incl}
        N = min(v for v in ns.values() if v >= 10) if all(ns[g] >= 5 for g in ns) else min(ns.values())
        if N < 10:
            print(f"{name:10s} insufficient sample ({ns})"); continue
        si = {g: snr_group(incl[g], N, rng)[0] for g in incl}
        se = {g: snr_group(excl[g], N, rng)[0] for g in excl}
        for g in incl:
            rows.append(dict(comp=name, timing=g, n=ns[g], snr_incl=si[g], snr_excl=se[g]))
        print(f"{name:10s} | {si['strategic']:.2f}/{si['regular']:.2f}/{si['late']:.2f}        "
              f"   | {se['strategic']:.2f}/{se['regular']:.2f}/{se['late']:.2f}           | "
              f"incl {'Y' if si['late']>si['strategic'] else 'N'} excl {'Y' if se['late']>se['strategic'] else 'N'}")
        late_gt_incl.append(si["late"] > si["strategic"]); late_gt_excl.append(se["late"] > se["strategic"])
    pd.DataFrame(rows).to_csv(TAB / "cross_gradient_clean.csv", index=False)
    n = len(late_gt_incl)
    print(f"\n[incl] late>early: {sum(late_gt_incl)}/{n} (p={binomtest(sum(late_gt_incl), n).pvalue:.4f})")
    print(f"[excl] late>early: {sum(late_gt_excl)}/{n} (p={binomtest(sum(late_gt_excl), n).pvalue:.4f})")
    print("saved cross_gradient_clean.csv")


if __name__ == "__main__":
    main()
