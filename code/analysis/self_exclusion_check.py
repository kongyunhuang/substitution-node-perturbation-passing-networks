"""
Check that the timing threshold survives excluding the substituted players' own
passes (teammates-only response). Zone DiD vs matched control, with and without
the subs' passes; per timing group, equal-N bootstrap magnitude vs sign-flip null.
Inputs: data/events_pass.parquet, data/substitution_batches.csv,
        data/windows/{window_specs,batch_flags}.parquet, data/did_controls.parquet
Output: results/tables/gradient_selfexcl.csv
Run:    python code/analysis/self_exclusion_check.py
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

    incl, excl = {"strategic": [], "regular": [], "late": []}, {"strategic": [], "regular": [], "late": []}
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
        ctrl_did = zvec(wcq) - zvec(wcp)
        # substituted players included
        incl[r.timing_group].append((zvec(wq) - zvec(wp)) - ctrl_did)
        # substituted players excluded from treated windows; control unchanged
        excl[r.timing_group].append((zvec(wq, swapped) - zvec(wp, swapped)) - ctrl_did)
    for d in (incl, excl):
        for g in d:
            d[g] = np.array(d[g])

    rng = np.random.default_rng(SEED)
    N = min(len(incl[g]) for g in incl)
    print(f"Equal sample size N={N}; group n: {{'s':{len(incl['strategic'])},'r':{len(incl['regular'])},'l':{len(incl['late'])}}}\n")

    def test(d, tag):
        print(f"=== {tag} ===")
        snrs = {}
        for g in ["strategic", "regular", "late"]:
            mags = [np.linalg.norm(d[g][rng.integers(len(d[g]), size=N)].mean(0)) for _ in range(800)]
            nf = []
            for _ in range(400):
                idx = rng.integers(len(d[g]), size=N)
                sgn = rng.choice([1, -1], size=len(d[g]))
                nf.append(np.linalg.norm((d[g][idx] * sgn[idx, None]).mean(0)))
            obs = np.mean(mags); p = 1 - (obs > np.array(nf)).mean()
            snrs[g] = obs / np.mean(nf)
            print(f"  {g:10s}: magnitude {obs:.4f}, SNR {snrs[g]:.2f}, p={p:.3f} {'*' if p<0.05 else 'ns'}")
        print(f"  late > strategic? SNR late {snrs['late']:.2f} vs strategic {snrs['strategic']:.2f} "
              f"-> {'gradient present' if snrs['late']>snrs['strategic'] else 'gradient absent'}")
        return snrs

    si = test(incl, "substituted players included")
    print()
    se = test(excl, "substituted players excluded (teammates only)")
    rows = [dict(timing=g, snr_incl=si[g], snr_excl=se[g]) for g in si]
    pd.DataFrame(rows).to_csv(TAB / "gradient_selfexcl.csv", index=False)
    print("\nIf late stays significant with late > strategic under exclusion, the gradient is team-level reorganization, not the substituted players' own passes.")


if __name__ == "__main__":
    main()
