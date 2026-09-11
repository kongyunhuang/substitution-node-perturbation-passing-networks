"""
Opponent-coupling features as candidate mechanisms of the timing threshold
(Section 4.3). Opponent reorg in the same windows, opponent pre-window tilt and
pass length, tested for a main effect on own reorg and mediation of timing.
Inputs: data/events_pass.parquet, data/substitution_batches.csv,
        data/windows/{window_specs,batch_flags}.parquet
Output: results/tables/opponent_mechanism.csv
Run:    python code/analysis/opponent_mechanisms.py
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
TAB = ROOT / "results" / "tables"
NX, NY = 6, 4
PL, PW = 120, 80


def zone_share(w):
    zx = np.clip((w.x / (PL / NX)).astype(int), 0, NX - 1)
    zy = np.clip((w.y / (PW / NY)).astype(int), 0, NY - 1)
    ex = np.clip((w.end_x / (PL / NX)).astype(int), 0, NX - 1)
    ey = np.clip((w.end_y / (PW / NY)).astype(int), 0, NY - 1)
    Z = np.zeros((24, 24))
    np.add.at(Z, ((zx * NY + zy).values, (ex * NY + ey).values), 1.0)
    np.fill_diagonal(Z, 0)
    t = Z / Z.sum() if Z.sum() > 0 else Z
    return t.sum(0) + t.sum(1)


def reorg_of(g, pre, post):
    wp = g[(g.tc >= pre.t_start) & (g.tc < pre.t_end)]
    wq = g[(g.tc > post.t_start) & (g.tc <= post.t_end)]
    if len(wp) < 10 or len(wq) < 10:
        return None
    return (np.linalg.norm(zone_share(wq) - zone_share(wp)), min(len(wp), len(wq)),
            float(wp.x.mean()), float(np.sqrt((wp.end_x - wp.x).values**2 + (wp.end_y - wp.y).values**2).mean()))


def zc(s):
    return (s - s.mean()) / s.std()


def resid_npass(y_z, npass):
    npz = zc(npass)
    X = pd.DataFrame({"a": npz, "b": npz**2, "c": zc(1 / np.sqrt(npass)), "d": zc(np.log(npass)), "const": 1.0})
    return y_z - sm.OLS(y_z, X).fit().predict(X)


def main():
    passes = pd.read_parquet(DATA / "events_pass.parquet")
    net = passes[passes.outcome.isna() & ~passes.is_set_piece].copy()
    t1 = passes[passes.period == 1].groupby("match_id").t_period_sec.max()
    net["tc"] = np.where(net.period == 1, net.t_period_sec, net.match_id.map(t1) + net.t_period_sec)
    ni = {k: g for k, g in net.groupby(["match_id", "team_id"])}
    teams_of = {}
    for (mid, tid) in ni:
        teams_of.setdefault(mid, []).append(tid)
    b = pd.read_csv(DATA / "substitution_batches.csv").set_index("batch_id")
    sp = pd.read_parquet(DATA / "windows" / "window_specs.parquet")
    stt = sp[(sp.kind == "time") & (sp.W_min == 15)].set_index(["batch_id", "side"])
    fl = pd.read_parquet(DATA / "windows" / "batch_flags.parquet")
    nontr = set(fl[(fl.W_min == 15) & ~fl.post_truncated].batch_id)

    rows = []
    for bid in b.index:
        if bid not in nontr or (bid, "pre") not in stt.index:
            continue
        r = b.loc[bid]
        g = ni.get((r.match_id, r.team_id))
        if g is None:
            continue
        opp_ids = [t for t in teams_of.get(r.match_id, []) if t != r.team_id]
        if not opp_ids:
            continue
        go = ni.get((r.match_id, opp_ids[0]))
        if go is None:
            continue
        pre, post = stt.loc[(bid, "pre")], stt.loc[(bid, "post")]
        ours = reorg_of(g, pre, post)
        opp = reorg_of(go, pre, post)
        if ours is None or opp is None:
            continue
        rows.append(dict(batch_id=bid, match_id=r.match_id,
                         reorg=ours[0], npass=ours[1],
                         opp_reorg=opp[0], opp_npass=opp[1], opp_tilt=opp[2], opp_plen=opp[3],
                         tnum={"strategic": 1, "regular": 2, "late": 3}[r.timing_group]))
    M = pd.DataFrame(rows)
    print(f"Sample: {len(M)} batches (both teams meet the window criteria)\n")

    # residualize each team's reorg on its own npass
    M["reorg_resid"] = resid_npass(zc(M.reorg), M.npass)
    M["opp_reorg_resid"] = resid_npass(zc(M.opp_reorg), M.opp_npass)

    base = smf.mixedlm("reorg_resid ~ tnum", M, groups=M.match_id).fit(reml=False)
    b_tnum0 = base.params["tnum"]
    print(f"=== timing baseline reorg_resid ~ tnum: beta={b_tnum0:+.3f} p={base.pvalues['tnum']:.2e} ===\n")

    FEATS = ["opp_reorg_resid", "opp_tilt", "opp_plen"]
    res = []
    for f in FEATS:
        d = M.dropna(subset=[f, "reorg_resid"]).copy()
        d["fz"] = zc(d[f])
        m = smf.mixedlm("reorg_resid ~ fz + tnum", d, groups=d.match_id).fit(reml=False)
        atten = (b_tnum0 - m.params["tnum"]) / b_tnum0 if b_tnum0 != 0 else np.nan
        res.append(dict(feature=f, n=len(d), beta=m.params["fz"], p=m.pvalues["fz"],
                        corr_timing=np.corrcoef(d.fz, d.tnum)[0, 1], tnum_atten=atten))
    R = pd.DataFrame(res)
    R["q"] = multipletests(R.p.fillna(1), method="fdr_bh")[1]
    R.to_csv(TAB / "opponent_mechanism.csv", index=False)

    print("=== opponent feature -> own reorg (timing controlled, FDR), correlation with timing, attenuation ===")
    print(f"  {'feature':16s} {'beta':>7s} {'q':>7s} {'corr_t':>8s} {'tnum_att':>9s}")
    for _, x in R.iterrows():
        print(f"  {x.feature:16s} {x.beta:+7.3f} {x.q:7.3f} {x.corr_timing:+8.3f} {x.tnum_atten*100:+8.1f}%")

    # dyadic coupling, the main question
    cpl = R[R.feature == "opp_reorg_resid"].iloc[0]
    rc = np.corrcoef(M.reorg_resid, M.opp_reorg_resid)[0, 1]
    print(f"\n=== dyadic coupling: own reorg vs opponent reorg in the same windows ===")
    print(f"  raw r={rc:+.3f};  timing-controlled beta={cpl.beta:+.3f} q={cpl.q:.3f} {'significant' if cpl.q<0.05 else 'not significant'}")

    print("\n" + "=" * 70 + "\nSUMMARY CHECKS\n" + "=" * 70)
    n_hit = int((R.q < 0.05).sum())
    med = R[(R.q < 0.05) & (R.tnum_atten > 0.20)]
    print(f"  opponent-feature FDR hits: {n_hit}/{len(R)};  mediating timing (>20%): {len(med)}")
    print("  caveat: both teams respond to the same situation, score and clock; a significant coupling means co-variation, not an opponent-driven effect (common cause).")
    if cpl.q < 0.05:
        print("\nVerdict: own reorg is coupled with opponent reorg in the same windows; cross-competition replication and a common-cause check decide.")
    else:
        print("\nVerdict: opponent behaviour (reorg, openness) explains neither own reorg nor the timing effect.")
        print("         With the other screens, no quantifiable mechanism explains the threshold.")
        print("         The explanation rests on literature-based hypotheses.")


if __name__ == "__main__":
    main()
