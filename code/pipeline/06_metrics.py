"""
Pipeline step 7: all network metrics on treated windows (batch x {pre,post} x 6 time windows
+ 50-pass) and on the matched controls (W = 15 and 50-pass), multi-process. Layer metrics:
density, Fagiolo clustering, betweenness, lambda2, efficiency; cross-layer PageRank versatility
with omega in {0.5,1,2}. Zero-length post windows give NaN; < 15 active zone edges sets
low_conf; players without passes are dropped from the supra matrix (czero_dropped).
Inputs: data/{events_pass,events_substitution,player_intervals,did_controls}.parquet, data/substitution_batches.csv, data/windows/window_specs.parquet
Outputs: data/metrics/metrics_player.parquet, data/metrics/metrics_zone.parquet
Run: python code/pipeline/06_metrics.py [--smoke 20] [--workers 8]
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics_core import (
    betweenness_inv_w,
    build_supra,
    density_rate,
    efficiency_inv_w,
    fagiolo_clustering,
    lambda2_sym,
    pagerank_versatility,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = DATA / "metrics"

TILINGS = [(3, 2), (4, 3), (6, 4), (7, 7), (10, 7)]
OMEGAS = [0.5, 1.0, 2.0]
LOW_CONF_EDGES = 15
W_MAIN_SEC = 900.0

# ---------------- worker globals, loaded once per process ----------------
G = {}


def parse_ts_sec(ts):
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def worker_init():
    passes = pd.read_parquet(DATA / "events_pass.parquet")
    subs = pd.read_parquet(DATA / "events_substitution.parquet")
    subs["t_sec"] = subs.timestamp.map(parse_ts_sec)
    t1_end = pd.concat([
        passes.query("period == 1").groupby("match_id").t_period_sec.max(),
        subs.query("period == 1").groupby("match_id").t_sec.max()], axis=1).max(axis=1)
    t2_end = pd.concat([
        passes.query("period == 2").groupby("match_id").t_period_sec.max(),
        subs.query("period == 2").groupby("match_id").t_sec.max()], axis=1).max(axis=1)
    G["match_end"] = (t1_end + t2_end).to_dict()
    passes = passes.assign(t_cum=np.where(passes.period == 1, passes.t_period_sec,
                                          passes.match_id.map(t1_end) + passes.t_period_sec))
    net = passes[passes.outcome.isna() & ~passes.is_set_piece]

    team_data = {}
    for key, g in net.groupby(["match_id", "team_id"]):
        g = g.sort_values("t_cum")
        d = dict(t=g.t_cum.values, p=g.player_id.values, r=g.recipient_id.values)
        for nx_, ny_ in TILINGS:
            zx = np.clip((g.x.values / (120 / nx_)).astype(int), 0, nx_ - 1)
            zy = np.clip((g.y.values / (80 / ny_)).astype(int), 0, ny_ - 1)
            ex = np.clip((g.end_x.values / (120 / nx_)).astype(int), 0, nx_ - 1)
            ey = np.clip((g.end_y.values / (80 / ny_)).astype(int), 0, ny_ - 1)
            d[f"src{nx_}{ny_}"] = zx * ny_ + zy
            d[f"dst{nx_}{ny_}"] = ex * ny_ + ey
        team_data[key] = d
    G["team"] = team_data

    iv = pd.read_parquet(DATA / "player_intervals.parquet")
    G["iv"] = {k: (g.player_id.values, g.t_on.values, g.t_off.values)
               for k, g in iv.groupby(["match_id", "team_id"])}
    G["specs"] = pd.read_parquet(DATA / "windows" / "window_specs.parquet")
    G["batches"] = pd.read_csv(DATA / "substitution_batches.csv")


def window_metrics(mid, tid, lo, hi, lo_open, hi_open, minutes, unit, batch_id,
                   kind, W_min, side):
    """All metrics of one window -> (player_row, [zone_rows])."""
    d = G["team"][(mid, tid)]
    t = d["t"]
    sel = ((t > lo) if lo_open else (t >= lo)) & ((t < hi) if hi_open else (t <= hi))
    pids, rids = d["p"][sel], d["r"][sel]
    n_pass = int(sel.sum())

    piv, t_on, t_off = G["iv"][(mid, tid)]
    on = (t_on < hi) & (t_off > lo)
    node_ids = piv[on]
    # dedupe in case a player has several intervals
    node_ids = pd.unique(node_ids)
    idx = {p: i for i, p in enumerate(node_ids)}
    n_p = len(node_ids)

    bad_min = (minutes is not None and minutes <= 1e-9)
    A_P = np.zeros((n_p, n_p))
    dropped = 0
    for p, r in zip(pids, rids):
        if p in idx and r in idx:
            A_P[idx[p], idx[r]] += 1
        else:
            dropped += 1

    def structural(W, mins):
        if bad_min or W.shape[0] < 3 or W.sum() == 0:
            return dict(density=np.nan, clustering=np.nan, betweenness=np.nan,
                        lambda2=np.nan, efficiency=np.nan)
        return dict(density=density_rate(W, mins),
                    clustering=float(np.mean(fagiolo_clustering(W))),
                    betweenness=float(np.mean(betweenness_inv_w(W, normalized=True))),
                    lambda2=lambda2_sym(W),
                    efficiency=efficiency_inv_w(W))

    base = dict(unit=unit, batch_id=batch_id, kind=kind, W_min=W_min, side=side)
    prow = dict(**base, n_nodes=n_p, n_passes=n_pass, dropped_passes=dropped,
                minutes=minutes if minutes is not None else np.nan,
                **structural(A_P, minutes))

    # involvement (initiated + received) -> zero C rows, tiling independent
    inv = A_P.sum(axis=1) + A_P.sum(axis=0)
    keep = inv > 0
    czero = int(n_p - keep.sum())
    prow["czero_dropped"] = czero

    zrows = []
    for nx_, ny_ in TILINGS:
        n_z = nx_ * ny_
        src, dst = d[f"src{nx_}{ny_}"][sel], d[f"dst{nx_}{ny_}"][sel]
        A_Z = np.zeros((n_z, n_z))
        np.add.at(A_Z, (src, dst), 1.0)
        A_Zh = A_Z.copy()
        np.fill_diagonal(A_Zh, 0.0)  # structural metrics without the diagonal
        active_edges = int((A_Zh > 0).sum())
        zr = dict(**base, tiling=f"{nx_}x{ny_}", active_edges=active_edges,
                  low_conf=active_edges < LOW_CONF_EDGES,
                  **structural(A_Zh, minutes))
        # versatility on the retained nodes
        if bad_min or n_pass == 0 or keep.sum() < 2:
            for om in OMEGAS:
                tag = str(om).replace(".", "")
                zr[f"vers_P_w{tag}"] = np.nan
                zr[f"vers_Z_w{tag}"] = np.nan
        else:
            C = np.zeros((n_p, n_z))
            for p, sz in zip(pids, src):
                if p in idx:
                    C[idx[p], sz] += 1
            for r, dz in zip(rids, dst):
                if r in idx:
                    C[idx[r], dz] += 1
            A_Pk, Ck = A_P[np.ix_(keep, keep)], C[keep]
            for om in OMEGAS:
                S = build_supra(A_Pk, A_Zh, Ck, omega=om)
                v = pagerank_versatility(S)
                tag = str(om).replace(".", "")
                zr[f"vers_P_w{tag}"] = float(v[:Ck.shape[0]].mean())
                zr[f"vers_Z_w{tag}"] = float(v[Ck.shape[0]:].mean())
        zrows.append(zr)
    return prow, zrows


def process_treated(batch_ids):
    if not G:
        worker_init()
    specs = G["specs"]
    bm = G["batches"].set_index("batch_id")
    sp = specs[specs.batch_id.isin(batch_ids)]
    prows, zrows = [], []
    for _, s in sp.iterrows():
        b = bm.loc[s.batch_id]
        lo_open = s.side == "post"  # post: (lo,hi]; pre: [lo,hi)
        if s.kind == "pass50" and (pd.isna(s.t_start) or pd.isna(s.t_end)):
            continue  # empty pass50 window
        minutes = None if s.kind == "pass50" else float(s.actual_len_min)
        p, z = window_metrics(b.match_id, b.team_id, float(s.t_start), float(s.t_end),
                              lo_open, not lo_open, minutes, "treated",
                              int(s.batch_id), s.kind, int(s.W_min), s.side)
        p["truncated"] = bool(s.truncated)
        prows.append(p)
        zrows.extend(z)
    return prows, zrows


def process_controls(ctrl_records):
    if not G:
        worker_init()
    prows, zrows = [], []
    for c in ctrl_records:
        mid, tid, tc, bid = c["control_match_id"], c["control_team_id"], \
            c["control_t_center"], c["batch_id"]
        mend = G["match_end"][mid]
        # W=15: pre [tc-900, tc), post (tc, min(tc+900, end)]
        post_e = min(tc + W_MAIN_SEC, mend)
        for side, lo, hi, lo_open, mins in [
                ("pre", tc - W_MAIN_SEC, tc, False, 15.0),
                ("post", tc, post_e, True, (post_e - tc) / 60.0)]:
            p, z = window_metrics(mid, tid, lo, hi, lo_open, not lo_open,
                                  mins, "control", bid, "time", 15, side)
            p["truncated"] = bool(side == "post" and post_e < tc + W_MAIN_SEC)
            prows.append(p)
            zrows.extend(z)
        # pass50: positional slice
        t = G["team"][(mid, tid)]["t"]
        pre_t = t[t < tc][-50:]
        post_t = t[t > tc][:50]
        for side, arr in [("pre", pre_t), ("post", post_t)]:
            if len(arr) == 0:
                continue
            lo, hi = (arr[0], tc) if side == "pre" else (tc, arr[-1])
            p, z = window_metrics(mid, tid, lo, hi, side == "post", side != "post",
                                  None, "control", bid, "pass50", 0, side)
            p["truncated"] = len(arr) < 50
            prows.append(p)
            zrows.extend(z)
    return prows, zrows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    batches = pd.read_csv(DATA / "substitution_batches.csv")
    ctrl = pd.read_parquet(DATA / "did_controls.parquet")
    ctrl = ctrl[ctrl.control_type != "none"]
    bids = batches.batch_id.tolist()
    crecs = ctrl[["batch_id", "control_match_id", "control_team_id",
                  "control_t_center"]].astype(
        dict(control_match_id=int, control_team_id=int)).to_dict("records")
    if args.smoke:
        bids = bids[: args.smoke]
        crecs = crecs[: args.smoke]

    chunks_b = [bids[i::args.workers * 4] for i in range(args.workers * 4)]
    chunks_c = [crecs[i::args.workers * 4] for i in range(args.workers * 4)]
    prows, zrows = [], []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=worker_init) as ex:
        for i, (p, z) in enumerate(ex.map(process_treated, chunks_b), 1):
            prows += p
            zrows += z
            print(f"treated chunk {i}/{len(chunks_b)} done", flush=True)
        for i, (p, z) in enumerate(ex.map(process_controls, chunks_c), 1):
            prows += p
            zrows += z
            print(f"control chunk {i}/{len(chunks_c)} done", flush=True)

    P = pd.DataFrame(prows)
    Z = pd.DataFrame(zrows)
    P.to_parquet(OUT / "metrics_player.parquet", index=False)
    Z.to_parquet(OUT / "metrics_zone.parquet", index=False)

    print("\n===== Run statistics =====")
    print(f"player rows: {P.shape}; zone rows: {Z.shape}")
    print(f"treated rows: {(P.unit == 'treated').sum()}, control rows: {(P.unit == 'control').sum()}")
    print(f"dropped passes (outside the node set): {P.dropped_passes.sum()} "
          f"({P.dropped_passes.sum() / max(P.n_passes.sum(), 1):.3%})")
    warn = P[(P.unit == 'treated') & (P.czero_dropped > 2)]
    print(f"windows with > 2 all-zero C rows: {len(warn)}; czero distribution: "
          f"{P.czero_dropped.value_counts().sort_index().head(6).to_dict()}")
    print(f"windows with NaN player density: {(P.density.isna()).sum()} "
          f"(zero length, < 3 nodes or no passes)")
    print(f"low-confidence zone windows by tiling:")
    print(Z[Z.unit == 'treated'].groupby('tiling').low_conf.mean().round(3).to_string())


if __name__ == "__main__":
    main()
