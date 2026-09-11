"""
Fetch the ten StatsBomb open-data competitions from GitHub, cache them as
parquet, and replicate the timing threshold per competition. Second-half
tactical non-GK subs merged within 90 s; 15-min windows; cross-match no-sub
control; 6x4 zone DiD by timing (<=60 / 61-75 / >75 min); equal-N bootstrap SNR.
Inputs: StatsBomb open data (network) or cached data/opendata/{cid}_{sid}.parquet
Output: data/opendata/{cid}_{sid}.parquet, data/opendata/{cid}_{sid}_perbatch_did.npz
Run:    python code/replication/fetch_open_data.py --cid 43 --sid 106   # World Cup 2022
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OD = DATA / "opendata"
NX, NY = 6, 4
PL, PW = 120, 80
SET_PIECE = {"Kick Off", "Free Kick", "Corner", "Throw-in", "Goal Kick"}
MERGE_SEC = 90.0
SEED = 20260613


RAW = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"


def _fetch_match(mid):
    """Fetch one match's events JSON from GitHub raw; return Pass / Substitution rows, flat schema."""
    import urllib.request
    url = f"{RAW}/events/{mid}.json"
    with urllib.request.urlopen(url, timeout=60) as r:
        events = __import__("json").loads(r.read().decode("utf-8"))
    rows = []
    for e in events:
        tp = e.get("type", {}).get("name")
        if tp not in ("Pass", "Substitution"):
            continue
        loc = e.get("location") or [np.nan, np.nan]
        pas = e.get("pass", {})
        end = pas.get("end_location") or [np.nan, np.nan]
        sub = e.get("substitution", {})
        rows.append(dict(
            match_id=mid, team=e.get("team", {}).get("name"),
            player=(e.get("player") or {}).get("name"),
            period=e.get("period"), minute=e.get("minute"), second=e.get("second"), type=tp,
            x=loc[0], y=loc[1], ex=end[0], ey=end[1],
            pass_recipient=(pas.get("recipient") or {}).get("name"),
            pass_outcome=(pas.get("outcome") or {}).get("name"),
            pass_type=(pas.get("type") or {}).get("name"),
            substitution_replacement=(sub.get("replacement") or {}).get("name"),
            substitution_outcome=(sub.get("outcome") or {}).get("name"),
            position=(e.get("position") or {}).get("name")))
    return rows


def fetch(cid, sid):
    """Fetch all matches of a competition/season in parallel (16 threads); cache as parquet."""
    import json
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor
    OD.mkdir(parents=True, exist_ok=True)
    cache = OD / f"{cid}_{sid}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    with urllib.request.urlopen(f"{RAW}/matches/{cid}/{sid}.json", timeout=60) as r:
        matches = json.loads(r.read().decode("utf-8"))
    mids = [m["match_id"] for m in matches]
    rows = []
    done = 0
    with ThreadPoolExecutor(max_workers=16) as ex:
        for res in ex.map(_fetch_match, mids):
            rows.extend(res)
            done += 1
            if done % 50 == 0:
                print(f"  ... {done}/{len(mids)} matches", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(cache, index=False)
    print(f"  cached {cache.name}: {len(df)} events, {len(mids)} matches")
    return df


def prep(df):
    """Continuous clock + network-pass selection."""
    p = df[df.type == "Pass"].copy()
    p1end = p[p.period == 1].groupby("match_id").apply(
        lambda g: g.minute.max() * 60 + g.second.max(), include_groups=False)
    df["tc"] = df.minute * 60 + df.second  # seconds; StatsBomb minute is cumulative across periods
    df.loc[df.period == 2, "tc"] = df.minute * 60 + df.second  # no offset needed for period 2
    net = df[(df.type == "Pass") & df.pass_outcome.isna()
             & ~df.pass_type.isin(SET_PIECE)].copy()
    return net, df[df.type == "Substitution"].copy()


def zvec(w):
    if w is None or len(w) < 10:
        return None
    zx = np.clip((w.x / (PL / NX)).astype(int), 0, NX - 1)
    zy = np.clip((w.y / (PW / NY)).astype(int), 0, NY - 1)
    ex = np.clip((w.ex / (PL / NX)).astype(int), 0, NX - 1)
    ey = np.clip((w.ey / (PW / NY)).astype(int), 0, NY - 1)
    W = np.zeros((24, 24))
    np.add.at(W, ((zx * NY + zy).values, (ex * NY + ey).values), 1.0)
    np.fill_diagonal(W, 0)
    t = W / W.sum() if W.sum() > 0 else W
    return t.sum(0) + t.sum(1)


def build(cid, sid):
    df = fetch(cid, sid)
    net, subs = prep(df)
    net_idx = {k: g.sort_values("tc") for k, g in net.groupby(["match_id", "team"])}
    sub_times = {k: np.sort(g.minute.values * 60.0) for k, g in subs.groupby(["match_id", "team"])}
    match_end = {m: g.minute.max() * 60 + g.second.max() for m, g in df.groupby("match_id")}

    # batches: second half, non-GK, non-injury; same team within 90 s merged
    sb2 = subs[(subs.period == 2) & (subs.position != "Goalkeeper")
               & (subs.substitution_outcome != "Injury")].copy()
    sb2["tc"] = sb2.minute * 60.0 + sb2.second
    batches = []
    for (mid, team), g in sb2.sort_values("tc").groupby(["match_id", "team"]):
        cur = None
        for _, r in g.iterrows():
            if cur and r.tc - cur["last"] <= MERGE_SEC:
                cur["last"] = r.tc
            else:
                if cur:
                    batches.append(cur)
                cur = {"match_id": mid, "team": team, "t0": r.tc, "last": r.tc, "minute": r.minute}
        if cur:
            batches.append(cur)
    B = pd.DataFrame(batches)
    B["timing"] = np.where(B.minute <= 60, "strategic", np.where(B.minute <= 75, "regular", "late"))
    return net_idx, sub_times, match_end, B


def win(net_idx, key, lo, hi, lo_open):
    g = net_idx.get(key)
    if g is None:
        return None
    t = g.tc.values
    sel = ((t > lo) & (t <= hi)) if lo_open else ((t >= lo) & (t < hi))
    return g[sel]


def clean(sub_times, key, lo, hi):
    arr = sub_times.get(key)
    if arr is None:
        return True
    i = np.searchsorted(arr, lo)
    return i >= len(arr) or arr[i] > hi


def perbatch_did(cid, sid):
    net_idx, sub_times, match_end, B = build(cid, sid)
    keys = list(net_idx.keys())
    rng = np.random.default_rng(SEED)
    W = 900.0
    out = {"strategic": [], "regular": [], "late": []}
    n_noctrl = 0
    for _, b in B.iterrows():
        key = (b.match_id, b.team)
        t0 = b.t0
        if t0 + W > match_end.get(b.match_id, 0):
            continue  # post-window truncated
        wp = win(net_idx, key, t0 - W, t0, False)
        wq = win(net_idx, key, t0, t0 + W, True)
        if zvec(wp) is None or zvec(wq) is None:
            continue
        # cross-match control: same clock +-2 min, no sub in window
        cands = [k for k in keys if k != key]
        rng.shuffle(cands)
        ctrl = None
        for ck in cands[:200]:
            for off in [0, 60, -60, 120, -120]:
                tc = t0 + off
                if tc - W < 0 or tc + W > match_end.get(ck[0], 0):
                    continue
                if not clean(sub_times, ck, tc - W, tc + W):
                    continue
                cp = win(net_idx, ck, tc - W, tc, False)
                cq = win(net_idx, ck, tc, tc + W, True)
                if zvec(cp) is not None and zvec(cq) is not None:
                    ctrl = (zvec(cq) - zvec(cp))
                    break
            if ctrl is not None:
                break
        if ctrl is None:
            n_noctrl += 1
            continue
        did = (zvec(wq) - zvec(wp)) - ctrl
        out[b.timing].append(did)
    for g in out:
        out[g] = np.array(out[g])
    print(f"Valid batches: {{'strategic':{len(out['strategic'])},'regular':{len(out['regular'])},"
          f"'late':{len(out['late'])}}}; dropped without control {n_noctrl}")
    np.savez(OD / f"{cid}_{sid}_perbatch_did.npz", **out)
    return out


def verify(out, label=""):
    rng = np.random.default_rng(SEED)
    grps = [g for g in ["strategic", "regular", "late"] if len(out[g]) >= 15]
    if len(grps) < 2:
        print("  insufficient sample, verification skipped"); return
    N = min(len(out[g]) for g in grps)
    print(f"  Equal sample size N={N}: pooled DiD magnitude + permutation significance")
    for g in grps:
        mags = [np.linalg.norm(out[g][rng.integers(len(out[g]), size=N)].mean(0)) for _ in range(800)]
        nf = []
        for _ in range(400):
            signs = rng.choice([1, -1], size=len(out[g]))
            idx = rng.integers(len(out[g]), size=N)
            nf.append(np.linalg.norm((out[g][idx] * signs[idx, None]).mean(0)))
        obs = np.mean(mags)
        p = 1 - (obs > np.array(nf)).mean()
        print(f"    {g:10s}(n={len(out[g])}): magnitude {obs:.4f}, SNR {obs/np.mean(nf):.2f}, "
              f"p={p:.3f} {'* signal' if p<0.05 else 'ns'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cid", type=int, required=True)
    ap.add_argument("--sid", type=int, required=True)
    a = ap.parse_args()
    npz = OD / f"{a.cid}_{a.sid}_perbatch_did.npz"
    if npz.exists():
        d = np.load(npz)
        out = {g: d[g] for g in d.files}
    else:
        out = perbatch_did(a.cid, a.sid)
    verify(out, f"{a.cid}_{a.sid}")


if __name__ == "__main__":
    main()
