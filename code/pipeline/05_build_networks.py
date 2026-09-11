"""
Pipeline step 6: two-layer network builders and on-pitch player intervals. Player layer nodes =
players on the pitch during the window (starting XI, subs, red cards); zone layer on 5 tilings
{3x2,4x3,6x4,7x7,10x7}, diagonal stored and removed for structural metrics; coupling C = passes
initiated + received per zone. Run as a script: cache the intervals and run the acceptance
checks (manual pass counts, C row sums, supra blocks, tiling figure).
Inputs: raw events (Starting XI, cards), data/events_pass.parquet, data/events_substitution.parquet
Outputs: data/player_intervals.parquet, results/figures/qc/grid_tilings.{pdf,png}
Run: python code/pipeline/05_build_networks.py
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(
    os.environ.get("STATSBOMB_EVENTS_DIR", "path/to/statsbomb_event_data/11_281_2023_24")
)
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

TILINGS = [(3, 2), (4, 3), (6, 4), (7, 7), (10, 7)]  # (nx, ny)
LOW_CONF_EDGES = 15  # min active zone edges (diagonal removed) before low_conf


def parse_ts_sec(ts: str) -> float:
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


# ------------------------------------------------------------- intervals
def extract_player_intervals() -> pd.DataFrame:
    """On-pitch interval [t_on, t_off) per player and match, continuous clock.
    t_on: 0 for starters, sub-on time otherwise; t_off: sub-off or red card, else match end."""
    cache = DATA / "player_intervals.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    passes = pd.read_parquet(DATA / "events_pass.parquet",
                             columns=["match_id", "period", "t_period_sec"])
    subs = pd.read_parquet(DATA / "events_substitution.parquet")
    subs["t_sec"] = subs.timestamp.map(parse_ts_sec)
    t1_end = pd.concat([
        passes.query("period == 1").groupby("match_id").t_period_sec.max(),
        subs.query("period == 1").groupby("match_id").t_sec.max()],
        axis=1).max(axis=1)
    t2_end = pd.concat([
        passes.query("period == 2").groupby("match_id").t_period_sec.max(),
        subs.query("period == 2").groupby("match_id").t_sec.max()],
        axis=1).max(axis=1)
    match_end = (t1_end + t2_end).to_dict()
    t1d = t1_end.to_dict()

    rows, red_rows = [], []
    for fp in sorted(RAW_DIR.glob("*_events.json")):
        with open(fp) as f:
            events = json.load(f)
        mid = int(fp.stem.split("_")[0])
        for e in events:
            etype = e.get("type", {}).get("name")
            if etype == "Starting XI":
                for p in e["tactics"]["lineup"]:
                    rows.append(dict(match_id=mid, team_id=e["team"]["id"],
                                     player_id=p["player"]["id"],
                                     player_name=p["player"]["name"],
                                     t_on=0.0, t_off=match_end[mid], starter=True))
            else:
                card = None
                if etype == "Foul Committed":
                    card = (e.get("foul_committed", {}).get("card") or {}).get("name")
                elif etype == "Bad Behaviour":
                    card = (e.get("bad_behaviour", {}).get("card") or {}).get("name")
                if card in ("Red Card", "Second Yellow"):
                    t = parse_ts_sec(e["timestamp"])
                    t_cum = t if e["period"] == 1 else t1d[mid] + t
                    red_rows.append(dict(match_id=mid, player_id=e["player"]["id"],
                                         t_red=t_cum))
    iv = pd.DataFrame(rows)

    # sub-on intervals; truncate players subbed off
    subs = subs.assign(t_cum=np.where(subs.period == 1, subs.t_sec,
                                      subs.match_id.map(t1_end) + subs.t_sec))
    sub_in = subs[["match_id", "team_id", "player_in_id", "player_in_name", "t_cum"]]
    iv_in = pd.DataFrame(dict(match_id=sub_in.match_id, team_id=sub_in.team_id,
                              player_id=sub_in.player_in_id,
                              player_name=sub_in.player_in_name,
                              t_on=sub_in.t_cum,
                              t_off=sub_in.match_id.map(match_end), starter=False))
    iv = pd.concat([iv, iv_in], ignore_index=True)
    out_map = subs.set_index(["match_id", "player_out_id"]).t_cum
    key = pd.MultiIndex.from_frame(iv[["match_id", "player_id"]])
    t_out = out_map.reindex(key).values
    iv["t_off"] = np.where(pd.notna(t_out) & (t_out > iv.t_on), t_out, iv.t_off)
    # red-card truncation
    reds = pd.DataFrame(red_rows)
    if len(reds):
        red_map = reds.set_index(["match_id", "player_id"]).t_red
        t_red = red_map.reindex(key).values
        iv["t_off"] = np.where(pd.notna(t_red) & (t_red > iv.t_on),
                               np.minimum(iv.t_off, np.where(pd.isna(t_red), np.inf, t_red)),
                               iv.t_off)
    iv.to_parquet(cache, index=False)
    print(f"[intervals] {len(iv)} intervals (starters {iv.starter.sum()}, substitutes {len(iv) - iv.starter.sum()}); "
          f"red cards {len(reds)}")
    return iv


def players_on_pitch(iv_team: pd.DataFrame, t_start: float, t_end: float) -> pd.DataFrame:
    """Players whose interval intersects [t_start, t_end); subs on/off included."""
    return iv_team[(iv_team.t_on < t_end) & (iv_team.t_off > t_start)]


# ------------------------------------------------------------- builders
def player_adjacency(win: pd.DataFrame, node_ids: list) -> np.ndarray:
    """A_P, node order = node_ids. Passes with passer or recipient outside node_ids
    are skipped and counted."""
    idx = {p: i for i, p in enumerate(node_ids)}
    W = np.zeros((len(node_ids), len(node_ids)))
    dropped = 0
    for p, r in zip(win.player_id.values, win.recipient_id.values):
        if p in idx and r in idx:
            W[idx[p], idx[r]] += 1
        else:
            dropped += 1
    return W, dropped


def zone_adjacency(win: pd.DataFrame, nx_: int, ny_: int) -> np.ndarray:
    """A_Z on an nx_ x ny_ grid (x-major), diagonal kept."""
    zx = np.clip((win.x.values / (120 / nx_)).astype(int), 0, nx_ - 1)
    zy = np.clip((win.y.values / (80 / ny_)).astype(int), 0, ny_ - 1)
    ex = np.clip((win.end_x.values / (120 / nx_)).astype(int), 0, nx_ - 1)
    ey = np.clip((win.end_y.values / (80 / ny_)).astype(int), 0, ny_ - 1)
    n = nx_ * ny_
    W = np.zeros((n, n))
    np.add.at(W, (zx * ny_ + zy, ex * ny_ + ey), 1.0)
    return W


def coupling_matrix(win: pd.DataFrame, node_ids: list, nx_: int, ny_: int,
                    mode: str = "both") -> np.ndarray:
    """C (N_p x N_z): w_ik = initiated + received per zone (mode 'both'); 'init' / 'recv' for sensitivity."""
    idx = {p: i for i, p in enumerate(node_ids)}
    n_z = nx_ * ny_
    C = np.zeros((len(node_ids), n_z))
    zx = np.clip((win.x.values / (120 / nx_)).astype(int), 0, nx_ - 1)
    zy = np.clip((win.y.values / (80 / ny_)).astype(int), 0, ny_ - 1)
    ex = np.clip((win.end_x.values / (120 / nx_)).astype(int), 0, nx_ - 1)
    ey = np.clip((win.end_y.values / (80 / ny_)).astype(int), 0, ny_ - 1)
    src_z, dst_z = zx * ny_ + zy, ex * ny_ + ey
    for p, r, sz, dz in zip(win.player_id.values, win.recipient_id.values, src_z, dst_z):
        if mode in ("both", "init") and p in idx:
            C[idx[p], sz] += 1
        if mode in ("both", "recv") and r in idx:
            C[idx[r], dz] += 1
    return C


# ------------------------------------------------------------- acceptance checks
def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    iv = extract_player_intervals()

    # check 0: 11 starters per team, 760 team-matches
    st = iv[iv.starter].groupby(["match_id", "team_id"]).size()
    print(f"[check 0] 11 starters per team: {(st == 11).all()} ({len(st)} groups, expected 760)")

    # check 1: one random batch, manual pass count of one player = A_P row sum
    b = pd.read_csv(DATA / "substitution_batches.csv")
    specs = pd.read_parquet(DATA / "windows" / "window_specs.parquet")
    passes = pd.read_parquet(DATA / "events_pass.parquet")
    subs = pd.read_parquet(DATA / "events_substitution.parquet")
    subs["t_sec"] = subs.timestamp.map(parse_ts_sec)
    t1_end = pd.concat([
        passes.query("period == 1").groupby("match_id").t_period_sec.max(),
        subs.query("period == 1").groupby("match_id").t_sec.max()], axis=1).max(axis=1)
    passes = passes.assign(t_cum=np.where(passes.period == 1, passes.t_period_sec,
                                          passes.match_id.map(t1_end) + passes.t_period_sec))
    net = passes[passes.outcome.isna() & ~passes.is_set_piece]

    rng = np.random.default_rng(55)
    r = b[b.batch_id == rng.choice(b.batch_id.values)].iloc[0]
    s = specs[(specs.batch_id == r.batch_id) & (specs.kind == "time")
              & (specs.W_min == 15) & (specs.side == "pre")].iloc[0]
    g = net[(net.match_id == r.match_id) & (net.team_id == r.team_id)]
    win = g[(g.t_cum >= s.t_start) & (g.t_cum < s.t_end)]
    iv_team = iv[(iv.match_id == r.match_id) & (iv.team_id == r.team_id)]
    nodes = players_on_pitch(iv_team, s.t_start, s.t_end)
    node_ids = nodes.player_id.tolist()
    A_P, dropped = player_adjacency(win, node_ids)
    pid = win.player_id.value_counts().index[0]  # most passes in the window
    manual_out = (win.player_id == pid).sum()
    i = node_ids.index(pid)
    print(f"[check 1] batch {r.batch_id} ({r.team_name}) W15 pre: {len(node_ids)} nodes, "
          f"dropped passes {dropped}; player {pid} manual passes made {manual_out} = A_P row sum {A_P[i].sum():.0f}: "
          f"{manual_out == A_P[i].sum()}")

    # check 2: C row sum = initiated + received
    C = coupling_matrix(win, node_ids, 6, 4)
    manual_total = manual_out + (win.recipient_id == pid).sum()
    print(f"[check 2] C row sum ({pid}) {C[i].sum():.0f} = initiated {manual_out} + received "
          f"{(win.recipient_id == pid).sum()}: {C[i].sum() == manual_total}")

    # check 3: supra dimensions and block sums
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from metrics_core import build_supra
    A_Z = zone_adjacency(win, 6, 4)
    zero_rows = int((C.sum(axis=1) == 0).sum())
    S = build_supra(A_P, A_Z, C, omega=1.0)
    print(f"[check 3] supra shape {S.shape} = ({len(node_ids)}+24); "
          f"block sums: P {S[:len(node_ids), :len(node_ids)].sum():.3f}, "
          f"Z {S[len(node_ids):, len(node_ids):].sum():.3f}, "
          f"C {S[:len(node_ids), len(node_ids):].sum():.3f} (each expected 1.0); "
          f"all-zero C rows {zero_rows}")

    # check 4: tiling figure
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.2))
    for ax, (nx_, ny_) in zip(axes, TILINGS):
        for x in np.linspace(0, 120, nx_ + 1):
            ax.axvline(x, color="k", lw=0.5)
        for y in np.linspace(0, 80, ny_ + 1):
            ax.axhline(y, color="k", lw=0.5)
        ax.set_xlim(0, 120); ax.set_ylim(0, 80)
        ax.set_title(f"{nx_}×{ny_} = {nx_*ny_} zones")
        ax.set_xticks([0, 60, 120]); ax.set_yticks([0, 40, 80])
    fig.suptitle("Grid tilings (attacking left→right, StatsBomb 120×80)")
    fig.tight_layout()
    qc = ROOT / "results" / "figures" / "qc"
    for ext in ["pdf", "png"]:
        fig.savefig(qc / f"grid_tilings.{ext}", dpi=300)
    print(f"[check 4] tiling figure saved to {qc}/grid_tilings.[pdf|png] (equal partition, visual confirmation only)")


if __name__ == "__main__":
    main()
