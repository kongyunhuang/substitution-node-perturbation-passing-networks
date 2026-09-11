"""
Territorial displacement after substitution and its self-exclusion audit
(Section 4.2, Figure 3). Territory = mean start x of completed passes; post minus
pre DiD vs matched control, with and without the subs' passes; one-sample t-tests.
Inputs: data/events_pass.parquet, data/substitution_batches.csv,
        data/windows/{window_specs,batch_flags}.parquet, data/did_controls.parquet
Output: results/tables/dterr_audit.csv
Run:    python code/analysis/territorial_shift.py
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
TAB = ROOT / "results" / "tables"


def terr(w, ex_ids=None):
    """Mean start x of completed passes; ex_ids drops passes made or received by those players."""
    cw = w[w.outcome.isna()]
    if ex_ids is not None:
        cw = cw[~cw.player_id.isin(ex_ids) & ~cw.recipient_id.isin(ex_ids)]
    return cw.x.mean() if len(cw) >= 5 else np.nan


def main():
    passes = pd.read_parquet(DATA / "events_pass.parquet")
    pall = passes[~passes.is_set_piece].copy()
    t1 = passes[passes.period == 1].groupby("match_id").t_period_sec.max()
    pall["tc"] = np.where(pall.period == 1, pall.t_period_sec, pall.match_id.map(t1) + pall.t_period_sec)
    pi = {k: g for k, g in pall.groupby(["match_id", "team_id"])}
    b = pd.read_csv(DATA / "substitution_batches.csv").set_index("batch_id")
    sp = pd.read_parquet(DATA / "windows" / "window_specs.parquet")
    st = sp[(sp.kind == "time") & (sp.W_min == 15)].set_index(["batch_id", "side"])
    fl = pd.read_parquet(DATA / "windows" / "batch_flags.parquet")
    nontr = set(fl[(fl.W_min == 15) & ~fl.post_truncated].batch_id)
    ctrl = pd.read_parquet(DATA / "did_controls.parquet")
    ctrl = ctrl[ctrl.control_type != "none"].set_index("batch_id")

    def win(mid, tid, lo, hi, lo_open):
        g = pi.get((mid, tid))
        if g is None:
            return None
        return g[(g.tc > lo) & (g.tc <= hi)] if lo_open else g[(g.tc >= lo) & (g.tc < hi)]

    rows = []
    for bid in b.index:
        if bid not in nontr or bid not in ctrl.index or (bid, "pre") not in st.index:
            continue
        r = b.loc[bid]
        pre, post = st.loc[(bid, "pre")], st.loc[(bid, "post")]
        wp = win(r.match_id, r.team_id, pre.t_start, pre.t_end, False)
        wq = win(r.match_id, r.team_id, post.t_start, post.t_end, True)
        c = ctrl.loc[bid]; tc = c.control_t_center
        wcp = win(int(c.control_match_id), int(c.control_team_id), tc - 900, tc, False)
        wcq = win(int(c.control_match_id), int(c.control_team_id), tc, tc + 900, True)
        if any(w is None or len(w) < 10 for w in [wp, wq, wcp, wcq]):
            continue
        swapped = ([int(x) for x in str(r.players_out_id).split("|")]
                   + [int(x) for x in str(r.players_in_id).split("|")])
        # control displacement (no substitution, nothing to exclude)
        ctrl_dterr = terr(wcq) - terr(wcp)
        # included / excluded
        di = (terr(wq) - terr(wp)) - ctrl_dterr
        de = (terr(wq, swapped) - terr(wp, swapped)) - ctrl_dterr
        if np.isnan(di) or np.isnan(de):
            continue
        rows.append(dict(batch_id=bid, timing=r.timing_group, did_terr_incl=di, did_terr_excl=de))
    M = pd.DataFrame(rows)
    M.to_csv(TAB / "dterr_audit.csv", index=False)
    print(f"Sample: {len(M)} batches (with control)\n")

    print("=== d_terr DiD, substituted players included vs excluded (remaining teammates) ===")
    for col, lab in [("did_terr_incl", "included"), ("did_terr_excl", "excluded (teammates)")]:
        x = M[col].dropna()
        t, p = ttest_1samp(x, 0)
        print(f"  {lab:20s}: mean {x.mean():+.3f} (pitch length 120), t={t:+.2f}, p={p:.2e} "
              f"{'* significant shift' if p < 0.05 else 'ns'}")
    drop = (1 - M.did_terr_excl.mean() / M.did_terr_incl.mean()) * 100 if M.did_terr_incl.mean() else 0
    print(f"  Displacement drops {drop:.0f}% after exclusion")
    print("\n=== by timing group ===")
    print(M.groupby("timing")[["did_terr_incl", "did_terr_excl"]].mean().round(3).to_string())

    excl = M.did_terr_excl.dropna()
    _, pe = ttest_1samp(excl, 0)
    print("\nVerdict:", end=" ")
    if pe < 0.05 and excl.mean() > 0:
        print("remaining teammates shift territory (significant after exclusion); d_terr is a genuine consequence")
    elif pe < 0.05 and excl.mean() < 0:
        print("direction reverses after exclusion; needs further investigation")
    else:
        print("no displacement after exclusion; d_terr is a shared-activity artifact of the substituted players' own passes")


if __name__ == "__main__":
    main()
