"""
Pipeline step 4: pre/post windows for every batch. Time windows W in {5,8,10,12,15,20} min,
pre = [t_first - W, t_first), post = (t_last, t_last + W]; 50-pass windows; per-W flags
post_contaminated, pre_overlap_prev, opp_sub_in_window, post_truncated. Contamination and
opponent flags use all 3528 substitutions; passes = completed, non set piece. Continuous
clock: period 2 time = T1_end + t_period_sec, half time not counted.
Inputs: data/events_pass.parquet, data/substitution_batches.csv, data/events_substitution.parquet
Outputs: data/windows/window_specs.parquet, data/windows/batch_flags.parquet
Run: python code/pipeline/04_windows.py
"""

import numpy as np
import pandas as pd

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
WDIR = DATA / "windows"

W_GRID = [5, 8, 10, 12, 15, 20]  # minutes
N_PASS_WINDOW = 50


def parse_ts_sec(ts: str) -> float:
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def main():
    WDIR.mkdir(parents=True, exist_ok=True)
    passes = pd.read_parquet(DATA / "events_pass.parquet")
    batches = pd.read_csv(DATA / "substitution_batches.csv")
    subs_all = pd.read_parquet(DATA / "events_substitution.parquet")
    subs_all["t_sec"] = subs_all["timestamp"].map(parse_ts_sec)

    # ---- continuous clock
    t1_end = (
        pd.concat([
            passes.loc[passes.period == 1, ["match_id", "t_period_sec"]],
            subs_all.loc[subs_all.period == 1, ["match_id", "t_sec"]].rename(columns={"t_sec": "t_period_sec"}),
        ]).groupby("match_id").t_period_sec.max()
    )
    t2_end = (
        pd.concat([
            passes.loc[passes.period == 2, ["match_id", "t_period_sec"]],
            subs_all.loc[subs_all.period == 2, ["match_id", "t_sec"]].rename(columns={"t_sec": "t_period_sec"}),
        ]).groupby("match_id").t_period_sec.max()
    )
    assert t1_end.index.size == 380 and t2_end.index.size == 380
    match_end_cum = t1_end + t2_end  # match end on the continuous clock

    passes = passes.assign(
        t_cum=np.where(passes.period == 1, passes.t_period_sec,
                       passes.match_id.map(t1_end) + passes.t_period_sec))
    # completed, non set piece
    net_pass = passes[passes.outcome.isna() & ~passes.is_set_piece]

    # all batches are second half
    assert (batches.period == 2).all()
    batches = batches.assign(
        t_first_cum=batches.match_id.map(t1_end) + batches.t_sec_first,
        t_last_cum=batches.match_id.map(t1_end) + batches.t_sec_last,
        match_end_cum=batches.match_id.map(match_end_cum),
        t1_end=batches.match_id.map(t1_end),
    )

    # all substitution times, excluded ones included
    subs_all = subs_all.assign(
        t_cum=np.where(subs_all.period == 1, subs_all.t_sec,
                       subs_all.match_id.map(t1_end) + subs_all.t_sec))
    all_sub_times = {k: np.sort(g.t_cum.values)
                     for k, g in subs_all.groupby(["match_id", "team_id"])}

    def any_sub_in(mid, tid, lo, hi, lo_open=True):
        """Any substitution of the team in (lo, hi] or [lo, hi]."""
        arr = all_sub_times.get((mid, tid), None)
        if arr is None:
            return False
        i = np.searchsorted(arr, lo, side="right" if lo_open else "left")
        return i < len(arr) and arr[i] <= hi
    # previous / next batch of the same team
    batches = batches.sort_values(["match_id", "team_id", "t_first_cum"]).reset_index(drop=True)
    grp = batches.groupby(["match_id", "team_id"])
    batches["prev_t_last_cum"] = grp.t_last_cum.shift(1)
    batches["next_t_first_cum"] = grp.t_first_cum.shift(-1)

    # team passes sorted by t_cum
    np_idx = {k: g.sort_values("t_cum") for k, g in net_pass.groupby(["match_id", "team_id"])}

    spec_rows, flag_rows = [], []
    for _, b in batches.iterrows():
        key = (b.match_id, b.team_id)
        tp = np_idx[key].t_cum.values  # team pass times
        for W in W_GRID:
            w = W * 60.0
            pre_s, pre_e = b.t_first_cum - w, b.t_first_cum  # [pre_s, pre_e)
            post_s = b.t_last_cum  # (post_s, post_e]
            post_e_nominal = b.t_last_cum + w
            post_e = min(post_e_nominal, b.match_end_cum)
            spans_ht = pre_s < b.t1_end  # pre-window spans half time
            n_pre = int(((tp >= pre_s) & (tp < pre_e)).sum())
            n_post = int(((tp > post_s) & (tp <= post_e)).sum())
            truncated = post_e_nominal > b.match_end_cum
            spec_rows += [
                dict(batch_id=b.batch_id, kind="time", W_min=W, side="pre",
                     t_start=pre_s, t_end=pre_e, n_passes=n_pre,
                     actual_len_min=W, truncated=False, spans_halftime=spans_ht),
                dict(batch_id=b.batch_id, kind="time", W_min=W, side="post",
                     t_start=post_s, t_end=post_e, n_passes=n_post,
                     actual_len_min=(post_e - post_s) / 60.0, truncated=truncated,
                     spans_halftime=False),
            ]
            # contamination / overlap / opponent flags per W
            contaminated = any_sub_in(b.match_id, b.team_id, post_s, post_e_nominal,
                                      lo_open=True)  # (t_last, t_last+W]: own batch events <= t_last stay out
            pre_overlap_prev = (b.prev_t_last_cum + w > pre_s) if pd.notna(b.prev_t_last_cum) else False
            opp_tid = [t for t in subs_all.loc[subs_all.match_id == b.match_id, "team_id"].unique()
                       if t != b.team_id]
            opp = any(any_sub_in(b.match_id, t, pre_s, post_e_nominal, lo_open=False)
                      for t in opp_tid)
            flag_rows.append(dict(batch_id=b.batch_id, W_min=W,
                                  post_contaminated=bool(contaminated),
                                  pre_overlap_prev=bool(pre_overlap_prev),
                                  opp_sub_in_window=bool(opp),
                                  post_truncated=truncated,
                                  post_len_min=(post_e - post_s) / 60.0))
        # 50-pass windows
        pre_idx = tp[tp < b.t_first_cum]
        post_idx = tp[tp > b.t_last_cum]
        pre50 = pre_idx[-N_PASS_WINDOW:]
        post50 = post_idx[:N_PASS_WINDOW]
        spec_rows += [
            dict(batch_id=b.batch_id, kind="pass50", W_min=0, side="pre",
                 t_start=pre50[0] if len(pre50) else np.nan,
                 t_end=b.t_first_cum, n_passes=len(pre50),
                 actual_len_min=(b.t_first_cum - pre50[0]) / 60.0 if len(pre50) else np.nan,
                 truncated=len(pre50) < N_PASS_WINDOW, spans_halftime=bool(len(pre50) and pre50[0] < b.t1_end)),
            dict(batch_id=b.batch_id, kind="pass50", W_min=0, side="post",
                 t_start=b.t_last_cum,
                 t_end=post50[-1] if len(post50) else np.nan, n_passes=len(post50),
                 actual_len_min=(post50[-1] - b.t_last_cum) / 60.0 if len(post50) else np.nan,
                 truncated=len(post50) < N_PASS_WINDOW, spans_halftime=False),
        ]

    specs = pd.DataFrame(spec_rows)
    flags = pd.DataFrame(flag_rows)
    specs.to_parquet(WDIR / "window_specs.parquet", index=False)
    flags.to_parquet(WDIR / "batch_flags.parquet", index=False)

    # ---- summary
    print("=" * 60)
    print("[1] Passes per time window (completed open-play passes of the team)")
    t = specs[specs.kind == "time"]
    print(t.groupby(["W_min", "side"]).n_passes.median().unstack().rename(columns=str).to_string())

    print("\n[2] Post-window truncation (by W and timing group)")
    fl = flags.merge(batches[["batch_id", "timing_group"]], on="batch_id")
    print(fl.groupby(["W_min", "timing_group"]).post_truncated.mean().unstack().round(3).to_string())

    print("\n[3] Contamination: share of post-windows with another same-team substitution")
    print(fl.groupby(["W_min", "timing_group"]).post_contaminated.mean().unstack().round(3).to_string())
    print("\n    Clean subsample N (no same-team substitution in the post-window):")
    clean = fl[~fl.post_contaminated].groupby(["W_min", "timing_group"]).size().unstack()
    print(clean.to_string())

    print("\n[4] Share of pre-windows overlapping the previous batch's post-window")
    print(fl.groupby(["W_min", "timing_group"]).pre_overlap_prev.mean().unstack().round(3).to_string())

    print("\n[5] Share of windows with an opponent substitution")
    print(fl.groupby(["W_min", "timing_group"]).opp_sub_in_window.mean().unstack().round(3).to_string())

    print("\n[6] 50-pass windows: batches with fewer than 50 passes")
    p50 = specs[specs.kind == "pass50"]
    print(p50.groupby("side").truncated.agg(["sum", "mean"]).round(3).to_string())

    print(f"\nSaved: window_specs {specs.shape}, batch_flags {flags.shape} -> {WDIR}")


if __name__ == "__main__":
    main()
