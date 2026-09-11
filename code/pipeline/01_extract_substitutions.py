"""
Pipeline step 1: extract substitution batches (same team and match, consecutive substitutions
<= 2 integer minutes apart) from the licensed StatsBomb La Liga 2023/24 events. Drops first-half,
injury and goalkeeper substitutions. Timing groups: strategic <= 60', regular 61-75', late > 75'.
Expected counts: 3528 events -> 3218 valid -> 2077 batches (445 / 829 / 803).
Inputs: $STATSBOMB_EVENTS_DIR/<match_id>_events.json (380 matches)
Outputs: data/substitution_batches_raw.csv, data/substitution_singles.csv
Run: python code/pipeline/01_extract_substitutions.py
"""

import json
import os
from pathlib import Path

import pandas as pd

DATA_DIR = Path(
    os.environ.get("STATSBOMB_EVENTS_DIR", "path/to/statsbomb_event_data/11_281_2023_24")
)
OUT_DIR = Path(__file__).resolve().parents[2] / "data"
BATCH_GAP_MIN = 2  # merge gap, integer minutes

# strategic <= 60', regular 61-75', late > 75'
def timing_group(minute: float) -> str:
    if minute <= 60:
        return "strategic"
    elif minute <= 75:
        return "regular"
    return "late"


def extract_subs_from_match(events_path: Path) -> list[dict]:
    """All substitution events of one match, unfiltered."""
    with open(events_path) as f:
        events = json.load(f)
    match_id = int(events_path.stem.split("_")[0])

    subs = []
    for e in events:
        if e.get("type", {}).get("name") != "Substitution":
            continue
        sub = e.get("substitution", {})
        subs.append(
            {
                "match_id": match_id,
                "event_uuid": e.get("id"),
                "period": e.get("period"),
                "minute": e.get("minute"),
                "second": e.get("second"),
                "team_id": e.get("team", {}).get("id"),
                "team_name": e.get("team", {}).get("name"),
                "player_out_id": e.get("player", {}).get("id"),
                "player_out_name": e.get("player", {}).get("name"),
                "player_in_id": sub.get("replacement", {}).get("id"),
                "player_in_name": sub.get("replacement", {}).get("name"),
                "outcome": sub.get("outcome", {}).get("name"),
                "position_out": e.get("position", {}).get("name"),
            }
        )
    return subs


def main():
    event_files = sorted(DATA_DIR.glob("*_events.json"))
    print(f"Found {len(event_files)} match event files")

    all_subs = []
    for p in event_files:
        all_subs.extend(extract_subs_from_match(p))
    df = pd.DataFrame(all_subs)
    print(f"Raw substitution events: {len(df)}")

    # exclusion flags
    df["excl_first_half"] = df["period"] == 1
    df["excl_injury"] = df["outcome"] == "Injury"
    df["excl_goalkeeper"] = df["position_out"] == "Goalkeeper"
    df["valid"] = ~(df["excl_first_half"] | df["excl_injury"] | df["excl_goalkeeper"])
    n_fh = df["excl_first_half"].sum()
    n_inj = (df["excl_injury"] & ~df["excl_first_half"]).sum()
    n_gk = (df["excl_goalkeeper"] & ~df["excl_first_half"] & ~df["excl_injury"]).sum()
    print(f"Excluded first-half substitutions: {n_fh}")
    print(f"Excluded second-half injury substitutions: {n_inj}")
    print(f"Excluded goalkeeper substitutions: {n_gk}")
    print(f"Valid single substitutions: {df['valid'].sum()}")

    # chain-merge within match, team, half when integer minute gap <= 2
    valid = df[df["valid"]].copy()
    valid["abs_min"] = valid["minute"] + valid["second"] / 60.0
    valid = valid.sort_values(["match_id", "team_id", "period", "abs_min"])

    batch_ids = []
    cur_batch = 0
    prev_key, prev_minute = None, None
    for _, row in valid.iterrows():
        key = (row["match_id"], row["team_id"], row["period"])
        if key != prev_key or row["minute"] - prev_minute > BATCH_GAP_MIN:
            cur_batch += 1
        batch_ids.append(cur_batch)
        prev_key, prev_minute = key, row["minute"]
    valid["batch_id"] = batch_ids

    # batch time = first substitution, n_subs = batch size
    batches = (
        valid.groupby("batch_id")
        .agg(
            match_id=("match_id", "first"),
            team_id=("team_id", "first"),
            team_name=("team_name", "first"),
            period=("period", "first"),
            batch_minute=("minute", "min"),
            batch_abs_min=("abs_min", "min"),
            n_subs=("event_uuid", "count"),
            players_out=("player_out_name", lambda s: "|".join(s)),
            players_in=("player_in_name", lambda s: "|".join(s)),
            player_out_ids=("player_out_id", lambda s: "|".join(map(str, s))),
            player_in_ids=("player_in_id", lambda s: "|".join(map(str, s))),
        )
        .reset_index()
    )
    batches["timing_group"] = batches["batch_minute"].apply(timing_group)

    print(f"\nSubstitution batches N = {len(batches)}")
    print(batches["timing_group"].value_counts().reindex(["strategic", "regular", "late"]))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["valid"]).to_csv(OUT_DIR / "substitution_singles.csv", index=False)
    batches.to_csv(OUT_DIR / "substitution_batches_raw.csv", index=False)
    print(f"\nWritten: {OUT_DIR / 'substitution_batches_raw.csv'}")
    print(f"Written: {OUT_DIR / 'substitution_singles.csv'}")

    # reference counts
    print("\nReference counts")
    print(f"Raw events 3528 vs {len(df)}")
    print(f"Valid singles 3218 vs {df['valid'].sum()}")
    print(f"Batches 2077 vs {len(batches)}")
    print("Groups 445/829/803 vs", dict(batches["timing_group"].value_counts()))


if __name__ == "__main__":
    main()
