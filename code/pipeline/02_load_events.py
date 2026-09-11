"""
Pipeline step 2: parse the 380 StatsBomb event files into pass, substitution and match-metadata
tables; print an integrity report (passes per match, 3528 substitutions, missing rates, coordinate
clamping to 120x80, timestamp order). Missing pass.outcome = completed, missing pass.type = open
play, is_set_piece is the only set-piece flag (Recovery / Interception are open play).
Inputs: $STATSBOMB_EVENTS_DIR/<match_id>_events.json, $STATSBOMB_EVENTS_DIR/00_info_all_matches_11_281.json
Outputs: data/events_pass.parquet, data/events_substitution.parquet, data/matches_meta.parquet,
         results/figures/qc/passes_per_match_hist.{pdf,png}
Run: python code/pipeline/02_load_events.py
"""

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

DATA_DIR = Path(
    os.environ.get("STATSBOMB_EVENTS_DIR", "path/to/statsbomb_event_data/11_281_2023_24")
)
ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data"
QC_DIR = ROOT / "results" / "figures" / "qc"

X_MAX, Y_MAX = 120.0, 80.0
OUTLIER_LOW, OUTLIER_HIGH = 300, 1500  # passes per match outlier bounds

# set-piece pass types; Recovery / Interception are open play and stay out
SET_PIECE_TYPES = {"Kick Off", "Free Kick", "Corner", "Throw-in", "Goal Kick"}


def parse_timestamp_sec(ts: str) -> float:
    """'HH:MM:SS.mmm' -> seconds within the period."""
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def extract_match(events_path: Path) -> tuple[list[dict], list[dict], list[dict]]:
    """One match -> (pass rows, substitution rows, timestamp-order violations)."""
    with open(events_path) as f:
        events = json.load(f)
    match_id = int(events_path.stem.split("_")[0])

    passes, subs, violations = [], [], []
    last_ts = {}  # period -> previous event timestamp (s)
    for e in events:
        etype = e.get("type", {}).get("name")
        period = e.get("period")
        ts = e.get("timestamp")
        if ts is not None and period is not None:
            tsec = parse_timestamp_sec(ts)
            if period in last_ts and tsec < last_ts[period] - 1e-9:
                violations.append(
                    {
                        "match_id": match_id,
                        "period": period,
                        "event_uuid": e.get("id"),
                        "timestamp": ts,
                        "prev_sec": last_ts[period],
                    }
                )
            last_ts[period] = tsec

        if etype == "Pass":
            p = e.get("pass", {})
            loc = e.get("location") or [None, None]
            end = p.get("end_location") or [None, None]
            passes.append(
                {
                    "match_id": match_id,
                    "event_uuid": e.get("id"),
                    "event_index": e.get("index"),
                    "period": period,
                    "timestamp": ts,
                    "t_period_sec": parse_timestamp_sec(ts) if ts else None,
                    "minute": e.get("minute"),
                    "second": e.get("second"),
                    "team_id": e.get("team", {}).get("id"),
                    "team_name": e.get("team", {}).get("name"),
                    "player_id": e.get("player", {}).get("id"),
                    "player_name": e.get("player", {}).get("name"),
                    "recipient_id": (p.get("recipient") or {}).get("id"),
                    "recipient_name": (p.get("recipient") or {}).get("name"),
                    "x": loc[0],
                    "y": loc[1],
                    "end_x": end[0],
                    "end_y": end[1],
                    "pass_type": (p.get("type") or {}).get("name"),  # None = open play
                    "is_set_piece": (p.get("type") or {}).get("name") in SET_PIECE_TYPES,
                    "outcome": (p.get("outcome") or {}).get("name"),  # None = Complete
                }
            )
        elif etype == "Substitution":
            sub = e.get("substitution", {})
            subs.append(
                {
                    "match_id": match_id,
                    "event_uuid": e.get("id"),
                    "event_index": e.get("index"),
                    "period": period,
                    "timestamp": ts,
                    "minute": e.get("minute"),
                    "second": e.get("second"),
                    "team_id": e.get("team", {}).get("id"),
                    "team_name": e.get("team", {}).get("name"),
                    "player_out_id": e.get("player", {}).get("id"),
                    "player_out_name": e.get("player", {}).get("name"),
                    "player_in_id": (sub.get("replacement") or {}).get("id"),
                    "player_in_name": (sub.get("replacement") or {}).get("name"),
                    "outcome": (sub.get("outcome") or {}).get("name"),
                    "position_out": (e.get("position") or {}).get("name"),
                }
            )
    return passes, subs, violations


def clamp_coords(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clamp coordinates to the pitch; return clamp counts."""
    stats = {}
    for col, hi in [("x", X_MAX), ("y", Y_MAX), ("end_x", X_MAX), ("end_y", Y_MAX)]:
        valid = df[col].notna()
        out = valid & ((df[col] < 0) | (df[col] > hi))
        stats[col] = int(out.sum())
        df.loc[out, col] = df.loc[out, col].clip(0, hi)
    stats["total_clamped_events"] = int(
        (df[["x", "y", "end_x", "end_y"]].notna().any(axis=1)).sum() and sum(stats.values())
    )
    return df, stats


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    QC_DIR.mkdir(parents=True, exist_ok=True)

    event_files = sorted(DATA_DIR.glob("*_events.json"))
    assert len(event_files) == 380, f"{len(event_files)} event files, expected 380"

    all_passes, all_subs, all_violations = [], [], []
    for i, fp in enumerate(event_files, 1):
        ps, ss, vs = extract_match(fp)
        all_passes.extend(ps)
        all_subs.extend(ss)
        all_violations.extend(vs)
        if i % 50 == 0:
            print(f"  ... {i}/380 matches parsed")

    df_pass = pd.DataFrame(all_passes)
    df_sub = pd.DataFrame(all_subs)

    # match metadata
    with open(DATA_DIR / "00_info_all_matches_11_281.json") as f:
        matches = json.load(f)
    df_meta = pd.DataFrame(
        {
            "match_id": m["match_id"],
            "match_date": m["match_date"],
            "match_week": m.get("match_week"),
            "home_team_id": m["home_team"]["home_team_id"],
            "home_team_name": m["home_team"]["home_team_name"],
            "away_team_id": m["away_team"]["away_team_id"],
            "away_team_name": m["away_team"]["away_team_name"],
            "home_score": m["home_score"],
            "away_score": m["away_score"],
        }
        for m in matches
    )

    print("\n" + "=" * 60)
    print("Integrity report")
    print("=" * 60)

    # (1) row counts
    per_match = df_pass.groupby("match_id").size()
    print(f"\n[1] Pass events: {len(df_pass)}; per match mean {per_match.mean():.1f}, "
          f"median {per_match.median():.0f}, min {per_match.min()}, max {per_match.max()}")
    outliers = per_match[(per_match < OUTLIER_LOW) | (per_match > OUTLIER_HIGH)]
    print(f"    Outlier matches (<{OUTLIER_LOW} or >{OUTLIER_HIGH}): "
          f"{outliers.to_dict() if len(outliers) else 'none'}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(per_match.values, bins=40, edgecolor="white")
    ax.axvline(OUTLIER_LOW, color="red", ls="--", lw=1, label=f"outlier bounds [{OUTLIER_LOW}, {OUTLIER_HIGH}]")
    ax.axvline(OUTLIER_HIGH, color="red", ls="--", lw=1)
    ax.set_xlabel("Passes per match")
    ax.set_ylabel("Matches")
    ax.set_title("La Liga 2023/24, passes per match (N=380)")
    ax.legend()
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(QC_DIR / f"passes_per_match_hist.{ext}", dpi=300)
    print(f"    Histogram saved to {QC_DIR}/passes_per_match_hist.[pdf|png]")

    # (2) substitution count
    n_subs = len(df_sub)
    print(f"\n[2] Substitution events: {n_subs}; reference 3528 {'OK, exact match' if n_subs == 3528 else 'MISMATCH, stop and investigate'}")

    # (3) missing rates
    print("\n[3] Missing rates (pass table):")
    hard_fields = ["timestamp", "period", "minute", "second", "team_id", "player_id", "x", "y", "end_x", "end_y"]
    for c in hard_fields:
        rate = df_pass[c].isna().mean()
        print(f"    {c:12s}: {rate:.4%}{'  WARN' if rate > 0 else ''}")
    complete_mask = df_pass["outcome"].isna()
    rec_missing_complete = df_pass.loc[complete_mask, "recipient_id"].isna().mean()
    print(f"    recipient_id missing among completed passes: {rec_missing_complete:.4%}")
    print(f"    pass_type labelled share: {df_pass['pass_type'].notna().mean():.2%} (missing = open play, not a data defect)")
    n_sp = df_pass["is_set_piece"].sum()
    n_openplay_typed = (df_pass["pass_type"].notna() & ~df_pass["is_set_piece"]).sum()
    print(f"    is_set_piece: {n_sp} set-piece passes ({n_sp/len(df_pass):.2%}); "
          f"labelled but open play (Recovery/Interception): {n_openplay_typed}")
    print(f"    outcome distribution: Complete(None)={complete_mask.mean():.2%}, "
          f"other={df_pass['outcome'].value_counts().to_dict()}")
    print("\n    Missing rates (substitution table):")
    for c in ["timestamp", "period", "minute", "team_id", "player_out_id", "player_in_id", "outcome", "position_out"]:
        rate = df_sub[c].isna().mean()
        print(f"    {c:15s}: {rate:.4%}{'  WARN' if rate > 0 else ''}")

    # (4) coordinate clamping
    df_pass, clamp_stats = clamp_coords(df_pass)
    n_coord = df_pass[["x", "y", "end_x", "end_y"]].notna().all(axis=1).sum()
    total_clamped = sum(v for k, v in clamp_stats.items() if k != "total_clamped_events")
    print(f"\n[4] Coordinate clamping: out-of-range counts per coordinate {clamp_stats}; "
          f"share of all coordinate values {total_clamped / (4 * len(df_pass)):.4%} (limit 0.5%)")

    # (5) timestamp order
    print(f"\n[5] Within-period timestamp order violations: {len(all_violations)}")
    if all_violations:
        print(pd.DataFrame(all_violations).head(20).to_string())

    # write
    df_pass.to_parquet(OUT_DIR / "events_pass.parquet", index=False)
    df_sub.to_parquet(OUT_DIR / "events_substitution.parquet", index=False)
    df_meta.to_parquet(OUT_DIR / "matches_meta.parquet", index=False)
    print(f"\nSaved: events_pass {df_pass.shape}, events_substitution {df_sub.shape}, "
          f"matches_meta {df_meta.shape} -> {OUT_DIR}")


if __name__ == "__main__":
    main()
