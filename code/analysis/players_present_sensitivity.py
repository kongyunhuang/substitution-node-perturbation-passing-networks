"""
Sensitivity to restricting the analysis to players present throughout both
windows (Supplementary Section S1.3, Table S3): also drop players subbed by the
same team in any other batch inside [pre.t_start, post.t_end]. Headline SNRs
must match gradient_selfexcl.csv within 0.10 or the script aborts.
Inputs: data/events_pass.parquet, data/substitution_batches.csv, data/windows/*,
        data/did_controls.parquet, results/tables/gradient_selfexcl.csv
Output: results/tables/fullwindow_sensitivity.csv
Run:    python code/analysis/players_present_sensitivity.py
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
GROUPS = ["strategic", "regular", "late"]


def zbin_vec(x, y):
    zx = np.clip((x / (PL / NX)).astype(int), 0, NX - 1)
    zy = np.clip((y / (PW / NY)).astype(int), 0, NY - 1)
    return zx * NY + zy


def zvec(w, ex_ids=None):
    """24-dim zone flow-share vector, same as zvec in self_exclusion_check.py
    (24x24 inter-zone flow, self-loops dropped, normalized, in-share + out-share)."""
    if ex_ids is not None:
        w = w[~w.player_id.isin(ex_ids) & ~w.recipient_id.isin(ex_ids)]
    if len(w) == 0:
        return np.zeros(NX * NY)
    Z = np.zeros((NX * NY, NX * NY))
    np.add.at(Z, (zbin_vec(w.x.values, w.y.values),
                  zbin_vec(w.end_x.values, w.end_y.values)), 1.0)
    np.fill_diagonal(Z, 0)
    t = Z / Z.sum() if Z.sum() > 0 else Z
    return t.sum(0) + t.sum(1)


def ids(s):
    return [int(x) for x in str(s).split("|") if str(x).strip() not in ("", "nan")]


def main():
    passes = pd.read_parquet(DATA / "events_pass.parquet")
    net = passes[passes.outcome.isna() & ~passes.is_set_piece].copy()
    t1 = passes[passes.period == 1].groupby("match_id").t_period_sec.max()
    net["tc"] = np.where(net.period == 1, net.t_period_sec,
                         net.match_id.map(t1) + net.t_period_sec)
    ni = {k: g for k, g in net.groupby(["match_id", "team_id"])}
    b = pd.read_csv(DATA / "substitution_batches.csv")
    # batch clock on the window time base (second half offset by first-half length)
    b["tc"] = np.where(b.period == 1, b.t_sec_first, b.match_id.map(t1) + b.t_sec_first)
    by_team = {k: g for k, g in b.groupby(["match_id", "team_id"])}
    b = b.set_index("batch_id")
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

    head = {g: [] for g in GROUPS}          # headline: exclude this batch's subs only
    full = {g: [] for g in GROUPS}          # full-window: also exclude other batches' subs inside the windows
    n_extra, n_drop = {g: [] for g in GROUPS}, {g: 0 for g in GROUPS}
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
        own = ids(r.players_out_id) + ids(r.players_in_id)
        # other batches of same team/match with sub time inside the two windows
        sib = by_team.get((r.match_id, r.team_id))
        extra = []
        if sib is not None:
            m = ((sib.batch_id != bid) & (sib.tc >= pre.t_start) & (sib.tc <= post.t_end))
            for _, s_ in sib[m].iterrows():
                extra += ids(s_.players_out_id) + ids(s_.players_in_id)
        allex = own + extra
        ctrl_did = zvec(wcq) - zvec(wcp)
        head[r.timing_group].append((zvec(wq, own) - zvec(wp, own)) - ctrl_did)
        # drop batch if either window < 10 passes after exclusion
        wq2 = wq[~wq.player_id.isin(allex) & ~wq.recipient_id.isin(allex)]
        wp2 = wp[~wp.player_id.isin(allex) & ~wp.recipient_id.isin(allex)]
        if len(wq2) < 10 or len(wp2) < 10:
            n_drop[r.timing_group] += 1
            continue
        full[r.timing_group].append((zvec(wq, allex) - zvec(wp, allex)) - ctrl_did)
        n_extra[r.timing_group].append(len(set(extra)))

    rng = np.random.default_rng(SEED)
    # SNR depends on N; two equal sample sizes
    #   N_ref = min headline group size, reproduces gradient_selfexcl.csv
    #   N_cmp = min over both specs, used for the headline vs full-window comparison
    N_ref = min(len(head[g]) for g in GROUPS)
    N_cmp = min(N_ref, min(len(full[g]) for g in GROUPS))
    print(f"Equal sample sizes: N_ref={N_ref} (reference table) / N_cmp={N_cmp} (headline vs full-window)")
    for g in GROUPS:
        print(f"  {g:10s} headline n={len(head[g]):4d} | full-window n={len(full[g]):4d} "
              f"(dropped after exclusion {n_drop[g]}) | mean extra excluded {np.mean(n_extra[g]):.2f} players")

    def snr(arr, N):
        A = np.array(arr)
        mags = [np.linalg.norm(A[rng.integers(len(A), size=N)].mean(0)) for _ in range(800)]
        nf = []
        for _ in range(400):
            idx = rng.integers(len(A), size=N)
            sgn = rng.choice([1, -1], size=len(A))
            nf.append(np.linalg.norm((A[idx] * sgn[idx, None]).mean(0)))
        obs = float(np.mean(mags))
        return obs / float(np.mean(nf)), 1 - float((obs > np.array(nf)).mean())

    rows = []
    print(f"\n{'group':10s}{'head(N_ref)':>13s}{'head(N_cmp)':>13s}{'p':>7s}{'full(N_cmp)':>13s}{'p':>7s}")
    for g in GROUPS:
        sref, _ = snr(head[g], N_ref)
        sh, ph = snr(head[g], N_cmp)
        sf, pf = snr(full[g], N_cmp)
        rows.append(dict(timing=g, n_batches_headline=len(head[g]),
                         n_batches_fullwindow=len(full[g]), N_ref=N_ref, N_cmp=N_cmp,
                         snr_headline_Nref=sref, snr_headline=sh, p_headline=ph,
                         snr_fullwindow=sf, p_fullwindow=pf,
                         mean_extra_excluded=float(np.mean(n_extra[g])), n_dropped=n_drop[g]))
        print(f"{g:10s}{sref:>13.3f}{sh:>13.3f}{ph:>7.3f}{sf:>13.3f}{pf:>7.3f}")

    R = pd.DataFrame(rows)
    # consistency gate: headline spec must reproduce the reference table
    ref = pd.read_csv(TAB / "gradient_selfexcl.csv").set_index("timing").snr_excl
    d = (R.set_index("timing").snr_headline_Nref - ref).abs()
    assert d.max() < 0.10, f"headline spec does not reproduce gradient_selfexcl.csv (max diff {d.max():.3f}):\n{d.round(3)}"
    print(f"\n  consistency gate passed: headline SNRs match gradient_selfexcl.csv snr_excl (max diff {d.max():.3f})")
    keep = (R.snr_fullwindow.iloc[0] < 1.25) and (R.snr_fullwindow.iloc[1:] > 1.25).all()
    print(f"  threshold shape under the full-window spec: {'preserved' if keep else 'CHANGED'}")
    R.to_csv(TAB / "fullwindow_sensitivity.csv", index=False)
    print("saved fullwindow_sensitivity.csv")


if __name__ == "__main__":
    main()
