"""Threshold robustness to window length and pitch grid across competitions.

Reruns the teammates-only cross-competition gradient (DiD, equal-sample SNR,
sign test) at W=10 min / 6x4 and W=15 min / 4x3; reference is W=15 min / 6x4.
Inputs:  data/opendata/{cid}_{sid}.parquet
Outputs: results/tables/mvp_multiscale.csv
Run: python code/replication/grid_window_robustness.py
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
PL, PW = 120, 80

spec = importlib.util.spec_from_file_location("od33", CODE_ROOT / "replication" / "open_data_two_layer.py")
m33 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m33)
od = m33.od

COMPS = {"WC22": (43, 106), "Euro24": (55, 282), "Euro20": (55, 43), "ISL": (1238, 108),
         "WSL": (37, 90), "Bund23": (9, 281), "Bund15": (9, 27), "LaLiga15": (11, 27),
         "PL15": (2, 27), "SerieA15": (12, 27)}


def make_zvec(nx, ny):
    K = nx * ny

    def zvec(w, names=None):
        if names is not None:
            w = w[~w.player.isin(names) & ~w.pass_recipient.isin(names)]
        if len(w) == 0:
            return None
        zx = np.clip((w.x / (PL / nx)).astype(int), 0, nx - 1)
        zy = np.clip((w.y / (PW / ny)).astype(int), 0, ny - 1)
        ex = np.clip((w.ex / (PL / nx)).astype(int), 0, nx - 1)
        ey = np.clip((w.ey / (PW / ny)).astype(int), 0, ny - 1)
        Z = np.zeros((K, K))
        np.add.at(Z, ((zx * ny + zy).values, (ex * ny + ey).values), 1.0)
        np.fill_diagonal(Z, 0)
        if Z.sum() == 0:
            return None
        t = Z / Z.sum()
        return t.sum(0) + t.sum(1)
    return zvec


def did_lists(cid, sid, W, zvec):
    df = od.fetch(cid, sid)
    net, _ = od.prep(df)
    ni = {k: g.sort_values("tc") for k, g in net.groupby(["match_id", "team"])}
    keys = list(ni.keys())
    sub_times = {k: np.sort(g.minute.values * 60.0)
                 for k, g in df[df.type == "Substitution"].groupby(["match_id", "team"])}
    match_end = {m: g.minute.max() * 60 + g.second.max() for m, g in df.groupby("match_id")}
    B = m33.batches_named(df)
    rng = np.random.default_rng(od.SEED)
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


def run_config(tag, W, nx, ny):
    zvec = make_zvec(nx, ny)
    rng = np.random.default_rng(od.SEED)
    rows, late_gt, reg_gt = [], [], []
    print(f"\n===== {tag}: W={W/60:.0f}min, grid {nx}x{ny} (teammates-only) =====")
    for name, (cid, sid) in COMPS.items():
        if not (OD / f"{cid}_{sid}.parquet").exists():
            continue
        excl = did_lists(cid, sid, W, zvec)
        ns = {g: len(excl[g]) for g in excl}
        if min(ns.values()) < 10:
            print(f"{name:10s} insufficient sample {ns}")
            continue
        N = min(ns.values())
        se = {g: snr_group(excl[g], N, rng)[0] for g in excl}
        for g in excl:
            rows.append(dict(config=tag, comp=name, timing=g, n=ns[g], snr_excl=se[g]))
        late_gt.append(se["late"] > se["strategic"])
        reg_gt.append(se["regular"] > se["strategic"])
        print(f"{name:10s} early/mid/late = {se['strategic']:.2f}/{se['regular']:.2f}/{se['late']:.2f} "
              f"{'late>early Y' if se['late']>se['strategic'] else 'late>early N'} "
              f"{'mid>early Y' if se['regular']>se['strategic'] else 'mid>early N'}")
    n = len(late_gt)
    p_l = binomtest(sum(late_gt), n).pvalue
    p_r = binomtest(sum(reg_gt), n).pvalue
    print(f"[{tag}] late>early {sum(late_gt)}/{n} (p={p_l:.4f}) | mid>early {sum(reg_gt)}/{n} (p={p_r:.4f})")
    return rows, (tag, sum(late_gt), sum(reg_gt), n, p_l, p_r)


def main():
    all_rows, verdicts = [], []
    for tag, W, nx, ny in [("W10_6x4", 600.0, 6, 4), ("W15_4x3", 900.0, 4, 3)]:
        rows, v = run_config(tag, W, nx, ny)
        all_rows += rows
        verdicts.append(v)
    pd.DataFrame(all_rows).to_csv(TAB / "mvp_multiscale.csv", index=False)
    print("\n===== verdict =====")
    ok = all(v[4] < 0.05 and v[5] < 0.05 for v in verdicts)
    for v in verdicts:
        print(f"  {v[0]}: late>early {v[1]}/{v[3]} p={v[4]:.4f}; mid>early {v[2]}/{v[3]} p={v[5]:.4f}")
    print("threshold robust across scales" if ok else "threshold not robust at some scale; report per scale")


if __name__ == "__main__":
    main()
