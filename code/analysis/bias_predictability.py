"""
Shared-activity bias in the primary dataset: naive versus corrected cross-layer
alignment as a function of the perturber's activity share (Section 4.4,
Figure 5b). naive = cos(dB, dz) with all passes; lpo = subs' passes dropped from dz.
Inputs: data/events_pass.parquet, data/substitution_batches.csv,
        data/windows/{window_specs,batch_flags}.parquet
Output: results/tables/bias_predictability.csv
Run:    python code/analysis/bias_predictability.py
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
TAB = ROOT / "results" / "tables"
NX, NY = 6, 4
PL, PW = 120, 80


def zbin(x, y):
    return int(np.clip(int(x / (PL / NX)), 0, NX - 1)) * NY + int(np.clip(int(y / (PW / NY)), 0, NY - 1))


def pdist(w, pid):
    v = np.zeros(24)
    for x, y in zip(w[w.player_id == pid].x, w[w.player_id == pid].y):
        if pd.notna(x):
            v[zbin(x, y)] += 1
    for x, y in zip(w[w.recipient_id == pid].end_x, w[w.recipient_id == pid].end_y):
        if pd.notna(x):
            v[zbin(x, y)] += 1
    s = v.sum()
    return (v / s if s > 0 else None), s


def zshare(w, ex=None):
    if ex is not None:
        w = w[~w.player_id.isin(ex) & ~w.recipient_id.isin(ex)]
    Z = np.zeros((24, 24))
    for x, y, exx, eyy in zip(w.x, w.y, w.end_x, w.end_y):
        if pd.notna(x) and pd.notna(exx):
            Z[zbin(x, y), zbin(exx, eyy)] += 1
    np.fill_diagonal(Z, 0)
    t = Z / Z.sum() if Z.sum() > 0 else Z
    return t.sum(0) + t.sum(1)


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else np.nan


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

    rows = []
    for bid in b.index:
        if bid not in nontr or (bid, "pre") not in st.index:
            continue
        r = b.loc[bid]
        g = ni.get((r.match_id, r.team_id))
        if g is None:
            continue
        pre, post = st.loc[(bid, "pre")], st.loc[(bid, "post")]
        wp = g[(g.tc >= pre.t_start) & (g.tc < pre.t_end)]
        wq = g[(g.tc > post.t_start) & (g.tc <= post.t_end)]
        if len(wp) < 10 or len(wq) < 10:
            continue
        out_ids = [int(x) for x in str(r.players_out_id).split("|")]
        in_ids = [int(x) for x in str(r.players_in_id).split("|")]
        din, sin = zip(*[pdist(wq, p) for p in in_ids])
        dout, sout = zip(*[pdist(wp, p) for p in out_ids])
        din = [d for d in din if d is not None]; dout = [d for d in dout if d is not None]
        if not din or not dout:
            continue
        dB = np.mean(din, 0) - np.mean(dout, 0)
        dz_incl = zshare(wq) - zshare(wp)
        dz_excl = zshare(wq, out_ids + in_ids) - zshare(wp, out_ids + in_ids)
        naive = cos(dB, dz_incl)
        lpo = cos(dB, dz_excl)
        # perturber_share: substituted players' touches / all window touches
        tot_q = len(wq) * 2  # two touches per pass (pass + reception)
        tot_p = len(wp) * 2
        share = (sum(sin) + sum(sout)) / (tot_q + tot_p)
        vol = min(len(wp), len(wq))
        rows.append(dict(batch_id=bid, match_id=r.match_id, naive=naive, lpo=lpo,
                         perturber_share=share, volume=vol, timing=r.timing_group))
    M = pd.DataFrame(rows).dropna(subset=["naive"])
    M.to_csv(TAB / "bias_predictability.csv", index=False)
    print(f"Sample: {len(M)} batches\n")

    print("=== naive alignment by quintile of perturber activity share ===")
    M["sbin"] = pd.qcut(M.perturber_share, 5, labels=["lowest","low","mid","high","highest"])
    print(M.groupby("sbin")[["perturber_share", "naive", "lpo", "volume"]].mean().round(4).to_string())

    def zc(s): return (s - s.mean()) / s.std()
    M["share_z"] = zc(M.perturber_share); M["logvol_z"] = zc(np.log(M.volume))
    print("\n=== predictability: naive ~ perturber_share + log(volume) ===")
    m = smf.mixedlm("naive ~ share_z + logvol_z", M, groups=M.match_id).fit(reml=False)
    for t in ["share_z", "logvol_z"]:
        print(f"  {t:10s}: beta={m.params[t]:+.4f} p={m.pvalues[t]:.2e} {'*' if m.pvalues[t]<0.05 else ''}")
    r2 = np.corrcoef(M.naive, m.fittedvalues)[0, 1] ** 2
    print(f"  share alone vs naive: r={np.corrcoef(M.perturber_share, M.naive)[0,1]:+.3f}")
    print(f"  (share + volume) fit of naive: R2={r2:.3f}")
    print("\n=== control: does the corrected (lpo) alignment also depend on share? (should be flat) ===")
    ml = smf.mixedlm("lpo ~ share_z + logvol_z", M, groups=M.match_id).fit(reml=False)
    print(f"  share_z -> lpo: beta={ml.params['share_z']:+.4f} p={ml.pvalues['share_z']:.2e}")
    print("\nVerdict:", end=" ")
    if m.pvalues["share_z"] < 0.05 and m.params["share_z"] > 0 and abs(ml.params["share_z"]) < abs(m.params["share_z"]):
        print("bias rises with perturber share and the dependence shrinks after correction (predictable and correctable)")
    else:
        print("predictability weaker than expected")


if __name__ == "__main__":
    main()
