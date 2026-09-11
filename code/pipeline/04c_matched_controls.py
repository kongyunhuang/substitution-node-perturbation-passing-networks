"""
Pipeline step 5: matched no-substitution control windows for the DiD design (W = 15 min).
Priority 1 same-match opponent with no substitution in [t0 - W, t0 + W]; priority 2 another
(match, team) at the same second-half time (+-2 min), matched on score state, home/away and
ranking-difference tertile, no substitution in the span, random draw with fixed seed; else
control_type "none". Uses all 3528 substitutions. Goal timeline reconciled with final scores.
Inputs: data/{substitution_batches.csv, events_substitution.parquet, events_pass.parquet, matches_meta.parquet, windows/batch_flags.parquet}, raw events (goals only, cached)
Outputs: data/did_controls.parquet, data/events_goals.parquet
Run: python code/pipeline/04c_matched_controls.py
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

W_SEC = 15 * 60.0  # main window
SEED = 20260611
CENTER_OFFSETS = [0, 30, -30, 60, -60, 90, -90, 120, -120]  # +-2 min tolerance


def parse_ts_sec(ts: str) -> float:
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def load_goals(t1_end: pd.Series) -> pd.DataFrame:
    """Goal timeline from raw events, reconciled with the final score of all 380 matches."""
    cache = DATA / "events_goals.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    rows = []
    for fp in sorted(RAW_DIR.glob("*_events.json")):
        with open(fp) as f:
            events = json.load(f)
        mid = int(fp.stem.split("_")[0])
        for e in events:
            etype = e.get("type", {}).get("name")
            scoring_team = None
            if etype == "Shot" and e.get("shot", {}).get("outcome", {}).get("name") == "Goal":
                scoring_team = e["team"]["id"]
            elif etype == "Own Goal For":  # benefiting team
                scoring_team = e["team"]["id"]
            if scoring_team is not None:
                rows.append(dict(match_id=mid, period=e["period"],
                                 t_sec=parse_ts_sec(e["timestamp"]),
                                 team_id=scoring_team))
    g = pd.DataFrame(rows)
    g["t_cum"] = np.where(g.period == 1, g.t_sec, g.match_id.map(t1_end) + g.t_sec)
    # reconcile with final scores
    meta = pd.read_parquet(DATA / "matches_meta.parquet")
    cnt = g.groupby(["match_id", "team_id"]).size()
    bad = []
    for _, r in meta.iterrows():
        h = cnt.get((r.match_id, r.home_team_id), 0)
        a = cnt.get((r.match_id, r.away_team_id), 0)
        if h != r.home_score or a != r.away_score:
            bad.append((r.match_id, h, r.home_score, a, r.away_score))
    if bad:
        raise AssertionError(f"goal timeline disagrees with the metadata score in {len(bad)} matches, e.g. {bad[:3]}")
    print(f"[goals] {len(g)} goals; final scores of all 380 matches reconciled (Own Goal For included)")
    g.to_parquet(cache, index=False)
    return g


def main():
    batches = pd.read_csv(DATA / "substitution_batches.csv")
    subs_all = pd.read_parquet(DATA / "events_substitution.parquet")
    passes = pd.read_parquet(DATA / "events_pass.parquet")
    meta = pd.read_parquet(DATA / "matches_meta.parquet")
    subs_all["t_sec"] = subs_all.timestamp.map(parse_ts_sec)

    t1_end = (pd.concat([
        passes.loc[passes.period == 1, ["match_id", "t_period_sec"]],
        subs_all.loc[subs_all.period == 1, ["match_id", "t_sec"]]
        .rename(columns={"t_sec": "t_period_sec"})])
        .groupby("match_id").t_period_sec.max())
    t2_end = (pd.concat([
        passes.loc[passes.period == 2, ["match_id", "t_period_sec"]],
        subs_all.loc[subs_all.period == 2, ["match_id", "t_sec"]]
        .rename(columns={"t_sec": "t_period_sec"})])
        .groupby("match_id").t_period_sec.max())
    match_end = t1_end + t2_end

    subs_all["t_cum"] = np.where(subs_all.period == 1, subs_all.t_sec,
                                 subs_all.match_id.map(t1_end) + subs_all.t_sec)
    sub_times = {k: np.sort(v.t_cum.values)
                 for k, v in subs_all.groupby(["match_id", "team_id"])}

    goals = load_goals(t1_end)
    goal_times = {k: g.sort_values("t_cum") for k, g in goals.groupby("match_id")}

    def clean(mid, tid, lo, hi):
        arr = sub_times.get((mid, tid))
        if arr is None:
            return True
        i = np.searchsorted(arr, lo)
        return i >= len(arr) or arr[i] > hi

    def score_state(mid, tid, t):
        g = goal_times.get(mid)
        if g is None:
            return "draw"
        before = g[g.t_cum < t]
        diff = (before.team_id == tid).sum() - (before.team_id != tid).sum()
        return "lead" if diff > 0 else ("trail" if diff < 0 else "draw")

    # ---- season ranking (points, goal difference, goals for) and rank-difference tertiles
    pts = {}
    for _, r in meta.iterrows():
        hw = r.home_score > r.away_score
        dr = r.home_score == r.away_score
        for tid, sf, sa, win in [(r.home_team_id, r.home_score, r.away_score, hw),
                                 (r.away_team_id, r.away_score, r.home_score, (not hw and not dr))]:
            p = pts.setdefault(tid, dict(pts=0, gd=0, gf=0))
            p["pts"] += 3 if win else (1 if dr else 0)
            p["gd"] += sf - sa
            p["gf"] += sf
    standings = sorted(pts, key=lambda t: (-pts[t]["pts"], -pts[t]["gd"], -pts[t]["gf"]))
    rank = {tid: i + 1 for i, tid in enumerate(standings)}
    print(f"[rank] top: {meta.set_index('home_team_id').home_team_name.get(standings[0], standings[0])}, "
          f"points {pts[standings[0]]['pts']}")

    m_home = meta.set_index("match_id").home_team_id.to_dict()
    m_away = meta.set_index("match_id").away_team_id.to_dict()

    def opp_of(mid, tid):
        return m_away[mid] if m_home[mid] == tid else m_home[mid]

    batches = batches.assign(
        t0=batches.match_id.map(t1_end) + batches.t_sec_first,
        is_home=[m_home[m] == t for m, t in zip(batches.match_id, batches.team_id)])
    batches["rank_diff"] = [abs(rank[t] - rank[opp_of(m, t)])
                            for m, t in zip(batches.match_id, batches.team_id)]
    tertile_edges = batches.rank_diff.quantile([1 / 3, 2 / 3]).values
    def tertile(rd):
        return int(np.searchsorted(tertile_edges, rd, side="right"))
    batches["rank_tertile"] = batches.rank_diff.map(tertile)
    batches["score_state"] = [score_state(m, t, t0) for m, t, t0
                              in zip(batches.match_id, batches.team_id, batches.t0)]
    batches["post_len"] = np.minimum(W_SEC, batches.match_id.map(match_end) - batches.t_last_cum
                                     if "t_last_cum" in batches else np.nan)
    # t_last_cum is not in the csv, recompute
    batches["t_last_cum_"] = batches.match_id.map(t1_end) + batches.t_sec_last
    batches["post_len"] = np.minimum(W_SEC, batches.match_id.map(match_end) - batches.t_last_cum_)

    # cross-match candidate pool keyed by (home/away, rank tertile)
    all_mt = [(m, m_home[m]) for m in m_home] + [(m, m_away[m]) for m in m_away]
    pool = {}
    for mid, tid in all_mt:
        key = (m_home[mid] == tid, tertile(abs(rank[tid] - rank[opp_of(mid, tid)])))
        pool.setdefault(key, []).append((mid, tid))

    rng = np.random.default_rng(SEED)
    rows = []
    for _, b in batches.iterrows():
        opp = opp_of(b.match_id, b.team_id)
        # priority 1: same-match opponent
        if clean(b.match_id, opp, b.t0 - W_SEC, b.t0 + W_SEC):
            rows.append(dict(batch_id=b.batch_id, control_type="same_match",
                             control_match_id=b.match_id, control_team_id=opp,
                             control_t_center=b.t0, n_candidates=1))
            continue
        # priority 2: cross-match, four matching variables
        cands = []
        for mid, tid in pool.get((b.is_home, b.rank_tertile), []):
            if mid == b.match_id:
                continue
            for off in CENTER_OFFSETS:
                tc = t1_end[mid] + b.t_sec_first + off
                if tc + b.post_len > match_end[mid] or tc - W_SEC < 0:
                    continue
                if not clean(mid, tid, tc - W_SEC, tc + W_SEC):
                    continue
                if score_state(mid, tid, tc) != b.score_state:
                    continue
                cands.append((mid, tid, tc))
                break  # first feasible centre per (match, team)
        if cands:
            mid, tid, tc = cands[rng.integers(len(cands))]
            rows.append(dict(batch_id=b.batch_id, control_type="cross_match",
                             control_match_id=mid, control_team_id=tid,
                             control_t_center=tc, n_candidates=len(cands)))
        else:
            rows.append(dict(batch_id=b.batch_id, control_type="none",
                             control_match_id=np.nan, control_team_id=np.nan,
                             control_t_center=np.nan, n_candidates=0))

    ctrl = pd.DataFrame(rows).merge(
        batches[["batch_id", "timing_group", "minute_first"]], on="batch_id")
    ctrl.to_parquet(DATA / "did_controls.parquet", index=False)

    # ---- report
    print("\n[1] Control type x timing group")
    print(ctrl.groupby(["timing_group", "control_type"]).size().unstack(fill_value=0).to_string())

    print("\n[2] Feasibility by match segment (10-min segments)")
    ctrl["segment"] = pd.cut(ctrl.minute_first, [44, 55, 65, 75, 85, 130],
                             labels=["45-55", "56-65", "66-75", "76-85", "86+"])
    print(ctrl.groupby(["segment", "control_type"], observed=True).size().unstack(fill_value=0).to_string())

    print("\n[3] Cross-match candidate counts:",
          ctrl[ctrl.control_type == "cross_match"].n_candidates.describe()[["min", "25%", "50%", "max"]].to_dict())

    # ---- check
    flags = pd.read_parquet(DATA / "windows" / "batch_flags.parquet")
    clean15 = flags[(flags.W_min == 15) & ~flags.post_contaminated].merge(
        batches[["batch_id", "timing_group"]], on="batch_id")
    n_clean_late = (clean15.timing_group == "late").sum()
    n_did_late = ((ctrl.timing_group == "late") & (ctrl.control_type != "none")).sum()
    g2 = n_clean_late < 150 or n_did_late < 150
    print(f"\n[G2] late group: clean subsample = {n_clean_late}, usable DiD controls = {n_did_late} "
          f"(minimum 150 each) -> {'G2 TRIGGERED, stop' if g2 else 'not triggered'}")
    print(f"\nSaved: did_controls.parquet {ctrl.shape}")


if __name__ == "__main__":
    main()
