"""Replication without extra-time substitutions, plus W=12 and W=20 min windows.

Part 1 drops batches after minute 90 and repeats the cross-competition sign test.
Part 2 adds the W12_6x4 and W20_6x4 scans to the multi-scale table.
Inputs:  data/opendata/{cid}_{sid}.parquet, results/tables/mvp_multiscale.csv
Outputs: results/tables/cross_gradient_no_overtime.csv, results/tables/mvp_multiscale.csv
Run: python code/replication/overtime_exclusion.py
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
TAB = ROOT / "results" / "tables"
spec = importlib.util.spec_from_file_location("mvp46", CODE_ROOT / "replication" / "grid_window_robustness.py")
m46 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m46)
od, COMPS, make_zvec, snr_group = m46.od, m46.COMPS, m46.make_zvec, m46.snr_group


def did_lists(cid, sid, W, zvec, max_minute=None):
    df = od.fetch(cid, sid)
    net, _ = od.prep(df)
    ni = {k: g.sort_values("tc") for k, g in net.groupby(["match_id", "team"])}
    keys = list(ni.keys())
    sub_times = {k: np.sort(g.minute.values * 60.0)
                 for k, g in df[df.type == "Substitution"].groupby(["match_id", "team"])}
    match_end = {m: g.minute.max() * 60 + g.second.max() for m, g in df.groupby("match_id")}
    B = m46.m33.batches_named(df)
    rng = np.random.default_rng(od.SEED)
    excl = {"strategic": [], "regular": [], "late": []}
    for b in B:
        if max_minute is not None and b["minute"] > max_minute:
            continue  # drop extra-time subs
        key, t0 = (b["match_id"], b["team"]), b["t0"]
        if t0 + W > match_end.get(b["match_id"], 0):
            continue
        g = ni.get(key)
        if g is None:
            continue
        wp = g[(g.tc >= t0 - W) & (g.tc < t0)]
        wq = g[(g.tc > t0) & (g.tc <= t0 + W)]
        swapped = list(b["out"]) + list(b["in"])
        zep, zeq = zvec(wp, swapped), zvec(wq, swapped)
        if zep is None or zeq is None:
            continue
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
                a, bb = zvec(cp), zvec(cq)
                if a is not None and bb is not None:
                    ctrl = bb - a
                    break
            if ctrl is not None:
                break
        if ctrl is None:
            continue
        tg = "strategic" if b["minute"] <= 60 else ("regular" if b["minute"] <= 75 else "late")
        excl[tg].append((zeq - zep) - ctrl)
    return {g: np.array(excl[g]) for g in excl}


def run(tag, W, nx, ny, max_minute=None):
    zvec = make_zvec(nx, ny)
    rng = np.random.default_rng(od.SEED)
    rows, lg, rg = [], [], []
    print(f"\n===== {tag} (W={W/60:.0f}min {nx}x{ny}{' no extra time' if max_minute else ''}) =====")
    for name, (cid, sid) in COMPS.items():
        if not (ROOT / "data" / "opendata" / f"{cid}_{sid}.parquet").exists():
            continue
        excl = did_lists(cid, sid, W, zvec, max_minute)
        ns = {g: len(excl[g]) for g in excl}
        good = [n for n in ns.values() if n >= 10]
        if len(good) < 3:
            print(f"{name:9s} insufficient sample {ns}"); continue
        N = min(good)
        se = {g: snr_group(excl[g], N, rng)[0] for g in excl}
        for g in excl:
            rows.append(dict(config=tag, comp=name, timing=g, n=ns[g], snr_excl=round(se[g], 3)))
        lg.append(se["late"] > se["strategic"]); rg.append(se["regular"] > se["strategic"])
        print(f"{name:9s} early/mid/late={se['strategic']:.2f}/{se['regular']:.2f}/{se['late']:.2f}")
    n = len(lg)
    pl = binomtest(sum(lg), n).pvalue if n else np.nan
    pr = binomtest(sum(rg), n).pvalue if n else np.nan
    print(f"[{tag}] late>early {sum(lg)}/{n} (p={pl:.4f}) | mid>early {sum(rg)}/{n} (p={pr:.4f})")
    return rows


def main():
    # part 1: no extra time (W15 6x4, minute <= 90)
    r_noot = run("W15_6x4_noOT", 900.0, 6, 4, max_minute=90)
    pd.DataFrame(r_noot).to_csv(TAB / "cross_gradient_no_overtime.csv", index=False)
    # part 2: W12 / W20 rows
    all_rows = []
    all_rows += run("W12_6x4", 720.0, 6, 4)
    all_rows += run("W20_6x4", 1200.0, 6, 4)
    if all_rows:
        old = pd.read_csv(TAB / "mvp_multiscale.csv")
        old = old[~old.config.isin(["W12_6x4", "W20_6x4"])]  # idempotent
        pd.concat([old, pd.DataFrame(all_rows)]).to_csv(TAB / "mvp_multiscale.csv", index=False)
    print("\nsaved cross_gradient_no_overtime.csv and mvp_multiscale.csv (W12/W20 rows)")


if __name__ == "__main__":
    main()
