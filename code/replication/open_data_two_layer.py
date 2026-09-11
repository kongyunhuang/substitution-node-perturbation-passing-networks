"""
Two-layer construction and cross-layer alignment on the open-data competitions
(shared conventions used by the other replication scripts). Per batch, align =
cos(dB, dz) with dB the subs' zone-footprint change and dz the team's zone
flow-share change; null from random non-substituted teammate pairs; sign test.
Inputs: data/opendata/{cid}_{sid}.parquet for COMPS (run fetch_open_data.py first)
Output: results/tables/duallayer_opendata.csv
Run:    python code/replication/open_data_two_layer.py
"""

import importlib.util
import warnings
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]  # <repo>/code

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, binomtest

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
OD = ROOT / "data" / "opendata"
TAB = ROOT / "results" / "tables"
NX, NY = 6, 4
PL, PW = 120, 80
W = 900.0

spec = importlib.util.spec_from_file_location("od30", CODE_ROOT / "replication" / "fetch_open_data.py")
od = importlib.util.module_from_spec(spec)
spec.loader.exec_module(od)

COMPS = {"WC22": (43, 106), "Euro24": (55, 282), "Euro20": (55, 43), "ISL": (1238, 108),
         "WSL": (37, 90), "Bund23": (9, 281), "Bund15": (9, 27), "LaLiga15": (11, 27),
         "PL15": (2, 27), "SerieA15": (12, 27)}


def zbin(x, y):
    return int(np.clip(int(x / (PL / NX)), 0, NX - 1)) * NY + int(np.clip(int(y / (PW / NY)), 0, NY - 1))


def pdist(w, name):
    """Zone distribution of player `name` in a window (pass origins + reception end points)."""
    v = np.zeros(24)
    s = w[w.player == name]
    for x, y in zip(s.x, s.y):
        if pd.notna(x):
            v[zbin(x, y)] += 1
    r = w[w.pass_recipient == name]
    for x, y in zip(r.ex, r.ey):
        if pd.notna(x):
            v[zbin(x, y)] += 1
    tot = v.sum()
    return v / tot if tot > 0 else None


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else np.nan


def batches_named(df):
    """Substitution batches with outgoing / incoming player names."""
    subs = df[df.type == "Substitution"].copy()
    subs["tc"] = subs.minute * 60.0 + subs.second
    sb2 = subs[(subs.period == 2) & (subs.position != "Goalkeeper")
               & (subs.substitution_outcome != "Injury")]
    out = []
    for (mid, team), g in sb2.sort_values("tc").groupby(["match_id", "team"]):
        cur = None
        for _, r in g.iterrows():
            if cur and r.tc - cur["last"] <= od.MERGE_SEC:
                cur["last"] = r.tc; cur["out"].append(r.player); cur["in"].append(r.substitution_replacement)
            else:
                if cur:
                    out.append(cur)
                cur = {"match_id": mid, "team": team, "t0": r.tc, "last": r.tc, "minute": r.minute,
                       "out": [r.player], "in": [r.substitution_replacement]}
        if cur:
            out.append(cur)
    return out


def per_comp(cid, sid, rng):
    df = od.fetch(cid, sid)
    net, _ = od.prep(df)
    ni = {k: g for k, g in net.groupby(["match_id", "team"])}
    match_end = {m: g.minute.max() * 60 + g.second.max() for m, g in df.groupby("match_id")}
    B = batches_named(df)
    reals, nulls = [], []
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
        din = [pdist(wq, p) for p in b["in"]]; dout = [pdist(wp, p) for p in b["out"]]
        din = [d for d in din if d is not None]; dout = [d for d in dout if d is not None]
        if not din or not dout:
            continue
        dB = np.mean(din, 0) - np.mean(dout, 0)
        dz = od.zvec(wq) - od.zvec(wp)
        if dz is None:
            continue
        reals.append(cos(dB, dz))
        # null: random pairs of non-substituted teammates present in both windows
        others = [p for p in wp.player.unique() if p not in b["out"] and p in wq.player.values]
        if len(others) >= 2:
            cs = []
            for _ in range(15):
                a, c = rng.choice(others, 2, replace=False)
                da, dc = pdist(wp, a), pdist(wq, c)
                if da is not None and dc is not None:
                    cs.append(cos(dc - da, dz))
            nulls.append(np.nanmean(cs) if cs else np.nan)
        else:
            nulls.append(np.nan)
    return np.array(reals), np.array(nulls)


def main():
    rng = np.random.default_rng(20260613)
    print(f"{'comp':10s} {'n':>4s} {'real':>9s} {'null':>7s} {'delta':>8s}  real>null")
    rows = []
    real_gt = []
    for name, (cid, sid) in COMPS.items():
        if not (OD / f"{cid}_{sid}.parquet").exists():
            continue
        reals, nulls = per_comp(cid, sid, rng)
        if len(reals) < 15:
            print(f"{name:10s} insufficient sample"); continue
        rm = np.nanmean(reals)
        paired = ~np.isnan(reals) & ~np.isnan(nulls)
        nm = np.nanmean(nulls[paired]) if paired.sum() else np.nan
        delta = rm - nm
        print(f"{name:10s} {len(reals):4d} {rm:+9.3f} {nm:+7.3f} {delta:+8.3f}   {'yes' if delta > 0 else 'no'}")
        rows.append(dict(comp=name, n=len(reals), real=rm, null=nm, delta=delta))
        real_gt.append(delta > 0)
    R = pd.DataFrame(rows)
    R.to_csv(TAB / "duallayer_opendata.csv", index=False)
    n = len(real_gt)
    print(f"\nReal alignment > teammate null: {sum(real_gt)}/{n} competitions "
          f"(sign test p={binomtest(sum(real_gt), n).pvalue:.4f})")
    print(f"Mean real align: {R.real.mean():+.3f}; mean null: {R.null.mean():+.3f}; "
          f"mean delta: {R.delta.mean():+.3f}")
    print("Primary-dataset reference: real +0.284, null +0.208, delta +0.076")
    print("If most competitions show real > null, player-to-zone propagation generalizes across competitions.")


if __name__ == "__main__":
    main()
