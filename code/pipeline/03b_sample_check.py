"""
Independent re-implementation of the step 3 batch partition (plain dicts and loops, no pandas
groupby/diff/cumsum): exclusion chain E1/E2/E3, chain merge at tau*, batch member sets.
Passes when the batch count and every event_uuid member set match substitution_batches.csv.
Inputs: data/events_substitution.parquet, data/events_pass.parquet,
        data/events_injury_stoppage.parquet, data/substitution_batches.csv
Outputs: verdict on stdout
Run: python code/pipeline/03b_sample_check.py --tau 88.0
"""

import argparse
from collections import defaultdict

import pandas as pd

DATA = __import__("pathlib").Path(__file__).resolve().parents[2] / "data"


def ts_to_sec(ts: str) -> float:
    parts = ts.split(":")
    return float(parts[0]) * 3600.0 + float(parts[1]) * 60.0 + float(parts[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, required=True)
    args = ap.parse_args()

    subs = pd.read_parquet(DATA / "events_substitution.parquet").to_dict("records")
    inj_stop = pd.read_parquet(DATA / "events_injury_stoppage.parquet").to_dict("records")

    # period 1 length for the continuous clock, scanned per match
    pp = pd.read_parquet(DATA / "events_pass.parquet",
                         columns=["match_id", "period", "t_period_sec"])
    t1_end = {}
    for mid, g in pp[pp.period == 1].groupby("match_id"):
        t1_end[mid] = g.t_period_sec.max()
    for r in inj_stop:
        if r["period"] == 1:
            t1_end[r["match_id"]] = max(t1_end.get(r["match_id"], 0.0), r["t_sec"])

    def cum(mid, period, t):
        return t if period == 1 else t1_end[mid] + t

    # injury stoppages: (match, team) -> [(t_cum, player_id)]
    is_idx = defaultdict(list)
    for r in inj_stop:
        is_idx[(r["match_id"], r["team_id"])].append(
            (cum(r["match_id"], r["period"], r["t_sec"]), r["player_id"]))

    # ---- exclusion chain
    kept = []
    n_e1 = n_e2 = n_e3 = 0
    for s in subs:
        if s["period"] == 1:  # E1
            n_e1 += 1
            continue
        t = ts_to_sec(s["timestamp"])
        t_c = cum(s["match_id"], s["period"], t)
        is_injury = s["outcome"] == "Injury"
        if not is_injury:  # E2 strict proxy: same player, stoppage within 120 s before
            for t_is, pid in is_idx.get((s["match_id"], s["team_id"]), []):
                if 0 <= t_c - t_is <= 120.0 and pid == s["player_out_id"]:
                    is_injury = True
                    break
        if is_injury:
            n_e2 += 1
            continue
        if s["position_out"] == "Goalkeeper":  # E3
            n_e3 += 1
            continue
        kept.append((s["match_id"], s["team_id"], s["period"], t, s["event_uuid"]))

    print(f"[03b] exclusion chain: E1={n_e1}, E2={n_e2}, E3={n_e3}, kept {len(kept)}")

    # ---- chain merge
    by_group = defaultdict(list)
    for mid, tid, per, t, uuid in kept:
        by_group[(mid, tid, per)].append((t, uuid))

    batches_b = []
    for key in sorted(by_group):
        evs = sorted(by_group[key])
        cur = [evs[0][1]]
        for (t_prev, _), (t_now, uuid_now) in zip(evs, evs[1:]):
            if t_now - t_prev <= args.tau:
                cur.append(uuid_now)
            else:
                batches_b.append(frozenset(cur))
                cur = [uuid_now]
        batches_b.append(frozenset(cur))

    print(f"[03b] independent implementation batches: {len(batches_b)}")

    # ---- compare with the main implementation
    main_df = pd.read_csv(DATA / "substitution_batches.csv")
    batches_a = [frozenset(s.split("|")) for s in main_df.event_uuids]
    set_a, set_b = set(batches_a), set(batches_b)
    print(f"[03b] main implementation batches: {len(batches_a)}")
    only_a, only_b = set_a - set_b, set_b - set_a
    if not only_a and not only_b and len(batches_a) == len(batches_b):
        print("[03b] PASS: batch count and all batch member sets agree")
    else:
        print(f"[03b] FAIL: {len(only_a)} batches only in main, {len(only_b)} only in independent")
        for fs in list(only_a)[:5]:
            print("   main only:", sorted(fs))
        for fs in list(only_b)[:5]:
            print("   independent only:", sorted(fs))


if __name__ == "__main__":
    main()
