"""
Pipeline step 8: delta = post - pre per metric (all windows) and DiD = delta_treated -
delta_control (W = 15 and 50-pass; NaN without a control), merged with timing_group, window
flags, low_conf_any (zone layer, either side) and control_type.
Inputs: data/metrics/{metrics_player,metrics_zone}.parquet, data/windows/batch_flags.parquet,
        data/did_controls.parquet, data/substitution_batches.csv
Outputs: data/metrics/deltas_player.parquet, data/metrics/deltas_zone.parquet
Run: python code/pipeline/06b_did_responses.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
MET = DATA / "metrics"

METRICS_P = ["density", "clustering", "betweenness", "lambda2", "efficiency"]
VERS = [f"vers_{l}_w{t}" for l in ["P", "Z"] for t in ["05", "10", "20"]]


def pivot_delta(df, keys, metrics):
    """Long table (one row per side) -> pre values and post - pre deltas."""
    pre = df[df.side == "pre"].set_index(keys)[metrics]
    post = df[df.side == "post"].set_index(keys)[metrics]
    common = pre.index.intersection(post.index)
    delta = (post.loc[common] - pre.loc[common]).add_prefix("d_")
    return pre.loc[common].add_prefix("pre_"), delta


def main():
    P = pd.read_parquet(MET / "metrics_player.parquet")
    Z = pd.read_parquet(MET / "metrics_zone.parquet")
    flags = pd.read_parquet(DATA / "windows" / "batch_flags.parquet")
    ctrl = pd.read_parquet(DATA / "did_controls.parquet")[["batch_id", "control_type"]]
    b = pd.read_csv(DATA / "substitution_batches.csv")[["batch_id", "timing_group", "match_id", "team_id", "minute_first", "n_subs"]]

    out = {}
    for name, df, keys, mets in [
            ("player", P, ["batch_id", "kind", "W_min"], METRICS_P),
            ("zone", Z, ["batch_id", "kind", "W_min", "tiling"], METRICS_P + VERS)]:
        rows = []
        for unit in ["treated", "control"]:
            d = df[df.unit == unit]
            pre, delta = pivot_delta(d, keys, mets)
            wide = pd.concat([pre, delta], axis=1).reset_index()
            wide["unit"] = unit
            rows.append(wide)
        treated = rows[0]
        control = rows[1].set_index(keys)[[f"d_{m}" for m in mets]].add_prefix("ctrl_")

        treated = treated.join(control, on=keys)
        for m in mets:
            treated[f"did_{m}"] = treated[f"d_{m}"] - treated[f"ctrl_d_{m}"]

        # merge flags
        treated = treated.merge(b, on="batch_id", how="left")
        treated = treated.merge(ctrl, on="batch_id", how="left")
        fl = flags.rename(columns={"W_min": "W_min"})[
            ["batch_id", "W_min", "post_contaminated", "pre_overlap_prev",
             "opp_sub_in_window", "post_truncated"]]
        treated = treated.merge(fl, on=["batch_id", "W_min"], how="left")  # pass50 rows (W_min=0) get NaN flags
        if name == "zone":
            lc = Z[Z.unit == "treated"].groupby(keys).low_conf.any().rename("low_conf_any")
            treated = treated.join(lc, on=keys)
        out[name] = treated
        treated.to_parquet(MET / f"deltas_{name}.parquet", index=False)

    # ---- self-check
    dp, dz = out["player"], out["zone"]
    print(f"deltas_player: {dp.shape}; deltas_zone: {dz.shape}")
    t15 = dp[(dp.kind == "time") & (dp.W_min == 15)]
    print(f"\nW=15 batches: {len(t15)} (expected 2075); non-missing DiD: {t15.did_density.notna().sum()} "
          f"(expected about: batches with a control and both windows valid)")
    print("\nDelta distribution (W=15, player):")
    print(t15[[f"d_{m}" for m in METRICS_P]].describe().loc[["mean", "std", "50%"]].round(4).to_string())
    print("\nDiD distribution (W=15, player):")
    print(t15[[f"did_{m}" for m in METRICS_P]].describe().loc[["mean", "std", "50%"]].round(4).to_string())
    # one batch by hand: delta and DiD chain
    r = t15[t15.did_density.notna()].iloc[7]
    praw = P[(P.unit == "treated") & (P.batch_id == r.batch_id) & (P.kind == "time") & (P.W_min == 15)]
    man_d = praw[praw.side == "post"].density.iloc[0] - praw[praw.side == "pre"].density.iloc[0]
    craw = P[(P.unit == "control") & (P.batch_id == r.batch_id) & (P.kind == "time") & (P.W_min == 15)]
    man_c = craw[craw.side == "post"].density.iloc[0] - craw[craw.side == "pre"].density.iloc[0]
    print(f"\nSpot check batch {r.batch_id}: manual delta {man_d:.6f} = table {r.d_density:.6f}: "
          f"{abs(man_d - r.d_density) < 1e-12}; manual DiD {man_d - man_c:.6f} = table "
          f"{r.did_density:.6f}: {abs(man_d - man_c - r.did_density) < 1e-12}")


if __name__ == "__main__":
    main()
