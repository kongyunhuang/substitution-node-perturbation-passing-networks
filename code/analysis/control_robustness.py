"""
Robustness of the threshold to the composition of the control pool;
authoritative headline SNRs (Section 4.1, Supplementary Table S2). Teammates-only
zone SNR on three control subsets: all, cross_match only, same_match only.
Inputs: data/events_pass.parquet, data/substitution_batches.csv,
        data/windows/{window_specs,batch_flags}.parquet, data/did_controls.parquet
Output: results/tables/control_robustness.csv
Run:    python code/analysis/control_robustness.py
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
TAB = ROOT / "results" / "tables"
NX, NY, PL, PW = 6, 4, 120, 80
SEED = 20260613
GROUPS = ["strategic", "regular", "late"]


def zbin_vec(x, y):
    zx = np.clip((x / (PL / NX)).astype(int), 0, NX - 1)
    zy = np.clip((y / (PW / NY)).astype(int), 0, NY - 1)
    return zx * NY + zy


def zvec(w, ex_ids=None):
    if ex_ids is not None:
        w = w[~w.player_id.isin(ex_ids) & ~w.recipient_id.isin(ex_ids)]
    if len(w) == 0:
        return np.zeros(24)
    Z = np.zeros((24, 24))
    np.add.at(Z, (zbin_vec(w.x.values, w.y.values), zbin_vec(w.end_x.values, w.end_y.values)), 1.0)
    np.fill_diagonal(Z, 0)
    t = Z / Z.sum() if Z.sum() > 0 else Z
    return t.sum(0) + t.sum(1)


def main():
    passes = pd.read_parquet(DATA / "events_pass.parquet")
    net = passes[passes.outcome.isna() & ~passes.is_set_piece].copy()
    t1 = passes[passes.period == 1].groupby("match_id").t_period_sec.max()
    net["tc"] = np.where(net.period == 1, net.t_period_sec, net.match_id.map(t1) + net.t_period_sec)
    ni = {k: g for k, g in net.groupby(["match_id", "team_id"])}
    b = pd.read_csv(DATA / "substitution_batches.csv").set_index("batch_id")
    sp = pd.read_parquet(DATA / "windows" / "window_specs.parquet")
    st = sp[(sp.kind == "time") & (sp.W_min == 15)].set_index(["batch_id", "side"])
    fl = pd.read_parquet(DATA / "windows" / "batch_flags.parquet")
    nontr = set(fl[(fl.W_min == 15) & ~fl.post_truncated].batch_id)
    ctrl = pd.read_parquet(DATA / "did_controls.parquet")
    ctrl = ctrl[ctrl.control_type != "none"].set_index("batch_id")

    def winp(mid, tid, lo, hi, lo_open):
        g = ni.get((mid, tid))
        if g is None:
            return None
        return g[(g.tc > lo) & (g.tc <= hi)] if lo_open else g[(g.tc >= lo) & (g.tc < hi)]

    # teammates-only DiD vector per batch, with control type
    recs = {}  # bid -> (group, ctype, dz_excl)
    for bid in nontr:
        if bid not in ctrl.index or (bid, "pre") not in st.index:
            continue
        r = b.loc[bid]
        pre, post = st.loc[(bid, "pre")], st.loc[(bid, "post")]
        wp = winp(r.match_id, r.team_id, pre.t_start, pre.t_end, False)
        wq = winp(r.match_id, r.team_id, post.t_start, post.t_end, True)
        c = ctrl.loc[bid]; tc = c.control_t_center
        wcp = winp(int(c.control_match_id), int(c.control_team_id), tc - 900, tc, False)
        wcq = winp(int(c.control_match_id), int(c.control_team_id), tc, tc + 900, True)
        if any(w is None or len(w) < 10 for w in [wp, wq, wcp, wcq]):
            continue
        swapped = ([int(x) for x in str(r.players_out_id).split("|")]
                   + [int(x) for x in str(r.players_in_id).split("|")])
        dz = (zvec(wq, swapped) - zvec(wp, swapped)) - (zvec(wcq) - zvec(wcp))
        recs[bid] = (r.timing_group, c.control_type, dz)

    rng = np.random.default_rng(SEED)

    def snr(arr, N):
        mags = [np.linalg.norm(arr[rng.integers(len(arr), size=N)].mean(0)) for _ in range(800)]
        nf = []
        for _ in range(400):
            idx = rng.integers(len(arr), size=N)
            sgn = rng.choice([1, -1], size=len(arr))
            nf.append(np.linalg.norm((arr[idx] * sgn[idx, None]).mean(0)))
        obs = np.mean(mags)
        return obs / np.mean(nf), 1 - (obs > np.array(nf)).mean()

    rows = []
    subsets = [("all", lambda ct: True),
               ("cross_only", lambda ct: ct == "cross_match"),
               ("same_only", lambda ct: ct == "same_match")]
    for sub, filt in subsets:
        by = {g: np.array([dz for (gg, ct, dz) in recs.values() if gg == g and filt(ct)])
              for g in GROUPS}
        ns = {g: len(by[g]) for g in GROUPS}
        valid = [n for n in ns.values() if n >= 10]
        if len(valid) < 3:
            print(f"[{sub}] insufficient sample {ns}, SNR skipped (n only)")
            for g in GROUPS:
                rows.append(dict(subset=sub, timing=g, n=ns[g], snr_excl=np.nan, p=np.nan))
            continue
        N = min(valid)
        print(f"\n=== control subset [{sub}]  equal N={N}  group n={ns} ===")
        for g in GROUPS:
            if ns[g] < 10:
                print(f"  {g:10s}: n={ns[g]} insufficient"); rows.append(dict(subset=sub, timing=g, n=ns[g], snr_excl=np.nan, p=np.nan)); continue
            s, p = snr(by[g], N)
            print(f"  {g:10s}: n={ns[g]:4d}  SNR={s:.2f}  p={p:.3f} {'*' if p < .05 else 'ns'}")
            rows.append(dict(subset=sub, timing=g, n=ns[g], snr_excl=round(s, 3), p=round(p, 4)))

    pd.DataFrame(rows).to_csv(TAB / "control_robustness.csv", index=False)
    print("\nIf cross_only still shows strategic near 1 and below regular/late, the threshold is not a control-composition artifact. Saved control_robustness.csv")


if __name__ == "__main__":
    main()
