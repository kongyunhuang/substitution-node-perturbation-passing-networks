"""
Zone-level reorganization maps per timing group (input to Figures 1 and 2).

Per timing group: group-mean teammates-only DiD vector over the 24 zones (positive = zone
gained flow share), per-zone bootstrap SE, and n. Prints a forward-minus-back sanity check.
Inputs: data/events_pass.parquet, data/substitution_batches.csv, data/did_controls.parquet,
        data/windows/{window_specs,batch_flags}.parquet
Outputs: results/tables/reorg_maps.npz, results/tables/reorg_maps_summary.csv
Run: python code/figures/fig1_fig2_zone_maps.py
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
TAB = ROOT / "results" / "tables"
NX, NY = 6, 4
PL, PW = 120, 80
SEED = 20260613


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

    excl = {"strategic": [], "regular": [], "late": []}
    for bid in nontr:
        if bid not in ctrl.index or (bid, "pre") not in st.index:
            continue
        r = b.loc[bid]
        pre, post = st.loc[(bid, "pre")], st.loc[(bid, "post")]
        wp = winp(r.match_id, r.team_id, pre.t_start, pre.t_end, False)
        wq = winp(r.match_id, r.team_id, post.t_start, post.t_end, True)
        c = ctrl.loc[bid]
        tc = c.control_t_center
        wcp = winp(int(c.control_match_id), int(c.control_team_id), tc - 900, tc, False)
        wcq = winp(int(c.control_match_id), int(c.control_team_id), tc, tc + 900, True)
        if any(w is None or len(w) < 10 for w in [wp, wq, wcp, wcq]):
            continue
        swapped = ([int(x) for x in str(r.players_out_id).split("|")]
                   + [int(x) for x in str(r.players_in_id).split("|")])
        ctrl_did = zvec(wcq) - zvec(wcp)
        excl[r.timing_group].append((zvec(wq, swapped) - zvec(wp, swapped)) - ctrl_did)

    rng = np.random.default_rng(SEED)
    out = {}
    rows = []
    for g, arr in excl.items():
        arr = np.array(arr)
        mean = arr.mean(0)
        boot = np.array([arr[rng.integers(len(arr), size=len(arr))].mean(0) for _ in range(2000)])
        se = boot.std(0)
        out[f"mean_{g}"] = mean
        out[f"se_{g}"] = se
        out[f"n_{g}"] = np.array([len(arr)])
        # sanity: forward (x bins 3-5) minus back (0-2) share gain should rise with timing
        zone_x = np.array([k // NY for k in range(24)])
        fwd = mean[zone_x >= 3].sum() - mean[zone_x <= 2].sum()
        rows.append(dict(group=g, n=len(arr), mean_norm=round(np.linalg.norm(mean), 4),
                         fwd_minus_back=round(fwd, 4),
                         n_sig_zones=int((np.abs(mean) > 2 * se).sum())))
        print(f"{g:10s}: n={len(arr):4d} |mean|={np.linalg.norm(mean):.4f} "
              f"fwd-back={fwd:+.4f} sig zones (|m|>2se)={int((np.abs(mean)>2*se).sum())}/24")
    np.savez(TAB / "reorg_maps.npz", **out)
    pd.DataFrame(rows).to_csv(TAB / "reorg_maps_summary.csv", index=False)
    print("saved reorg_maps.npz, reorg_maps_summary.csv")


if __name__ == "__main__":
    main()
