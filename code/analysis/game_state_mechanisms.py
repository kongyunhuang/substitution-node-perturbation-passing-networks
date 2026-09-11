"""
Game-state and openness features as candidate mechanisms of the timing threshold
(Section 4.3). Six pre-window geometry features tested for a main effect on
volume-residualized reorg and for mediation (attenuation) of the timing coefficient.
Inputs: data/events_pass.parquet, data/substitution_batches.csv,
        data/windows/{window_specs,batch_flags}.parquet
Output: results/tables/gamestate_mechanism.csv
Run:    python code/analysis/game_state_mechanisms.py
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


def openness(w):
    """Openness / stretch features of a window (completed-pass geometry)."""
    dx = (w.end_x - w.x).values
    dy = (w.end_y - w.y).values
    plen = float(np.mean(np.sqrt(dx**2 + dy**2)))
    return dict(plen=plen, vprog=float(dx.mean()), fwd=float((dx > 0).mean()),
                spr_x=float(w.x.std()), spr_y=float(w.y.std()), tilt=float(w.x.mean()))


def zc(s):
    return (s - s.mean()) / s.std()


def main():
    passes = pd.read_parquet(DATA / "events_pass.parquet")
    net = passes[passes.outcome.isna() & ~passes.is_set_piece].copy()
    t1 = passes[passes.period == 1].groupby("match_id").t_period_sec.max()
    net["tc"] = np.where(net.period == 1, net.t_period_sec, net.match_id.map(t1) + net.t_period_sec)
    ni = {k: g for k, g in net.groupby(["match_id", "team_id"])}
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
        pre, post = stt.loc[(bid, "pre")], stt.loc[(bid, "post")]
        wp = g[(g.tc >= pre.t_start) & (g.tc < pre.t_end)]
        wq = g[(g.tc > post.t_start) & (g.tc <= post.t_end)]
        if len(wp) < 10 or len(wq) < 10:
            continue
        reorg = np.linalg.norm(zone_share(wq) - zone_share(wp))
        rows.append(dict(batch_id=bid, match_id=r.match_id, reorg=reorg,
                         npass=min(len(wp), len(wq)),
                         tnum={"strategic": 1, "regular": 2, "late": 3}[r.timing_group],
                         **openness(wp)))
    M = pd.DataFrame(rows)
    print(f"Sample: {len(M)} batches\n")

    # residualize reorg on [npass, npass^2, 1/sqrt(npass), log npass]
    M["reorg_z"] = zc(M.reorg)
    npz = zc(M.npass)
    Xv = pd.DataFrame({"a": npz, "b": npz**2, "c": zc(1 / np.sqrt(M.npass)), "d": zc(np.log(M.npass)), "const": 1.0})
    fitv = sm.OLS(M.reorg_z, Xv).fit()
    M["reorg_resid"] = M.reorg_z - fitv.predict(Xv)
    print(f"[volume control] reorg ~ flex(npass) R2={fitv.rsquared:.3f}; using reorg_resid\n")

    FEATS = ["plen", "vprog", "fwd", "spr_x", "spr_y", "tilt"]
    # baseline timing effect, no features
    base = smf.mixedlm("reorg_resid ~ tnum", M, groups=M.match_id).fit(reml=False)
    b_tnum0, p_tnum0 = base.params["tnum"], base.pvalues["tnum"]
    print(f"=== timing baseline: reorg_resid ~ tnum -> beta_tnum={b_tnum0:+.3f} p={p_tnum0:.2e} (effect to be explained) ===\n")

    res = []
    for f in FEATS:
        d = M.dropna(subset=[f, "reorg_resid"]).copy()
        d["fz"] = zc(d[f])
        # main effect, timing controlled
        m = smf.mixedlm("reorg_resid ~ fz + tnum", d, groups=d.match_id).fit(reml=False)
        bm, pm = m.params["fz"], m.pvalues["fz"]
        # mediation: tnum attenuation after adding the feature
        b_tnum_adj = m.params["tnum"]
        atten = (b_tnum0 - b_tnum_adj) / b_tnum0 if b_tnum0 != 0 else np.nan
        # does the feature itself vary with timing
        corr_t = np.corrcoef(d.fz, d.tnum)[0, 1]
        res.append(dict(feature=f, n=len(d), beta_main=bm, p_main=pm,
                        corr_timing=corr_t, tnum_attenuation=atten, b_tnum_adj=b_tnum_adj))
    R = pd.DataFrame(res)
    R["q_main"] = multipletests(R.p_main.fillna(1), method="fdr_bh")[1]
    R = R.sort_values("p_main")
    R.to_csv(TAB / "gamestate_mechanism.csv", index=False)

    print("=== (M) openness feature -> reorg (FDR), correlation with timing, attenuation of tnum ===")
    print(f"  {'feat':6s} {'b_main':>8s} {'q_main':>8s} {'corr_t':>8s} {'tnum_att':>9s}  verdict")
    for _, x in R.iterrows():
        tag = "* significant" if x.q_main < 0.05 else "none"
        print(f"  {x.feature:6s} {x.beta_main:+8.3f} {x.q_main:8.3f} {x.corr_timing:+8.3f} {x.tnum_attenuation*100:+8.1f}%  {tag}")

    # candidate: q<0.05 and >20% tnum attenuation
    med = R[(R.q_main < 0.05) & (R.tnum_attenuation > 0.20)]
    print("\n=== any feature that predicts reorg and absorbs the timing effect (>20%)? ===")
    if len(med):
        for _, x in med.iterrows():
            print(f"  {x.feature}: q={x.q_main:.3f}, absorbs {x.tnum_attenuation*100:.0f}% of the timing effect -> candidate (needs cross-competition replication and an RTM check)")
    else:
        print("  none; no openness feature both predicts reorg and mediates the timing effect")

    # summary checks
    print("\n" + "=" * 70)
    print("SUMMARY CHECKS")
    print("=" * 70)
    n_main = int((R.q_main < 0.05).sum())
    print(f"  main-effect FDR hits: {n_main}/{len(R)}")
    print(f"  of which mediate timing (>20% attenuation): {len(med)}")
    print("  caveat: pre-window features predicting a pre->post change are exposed to regression to the mean; a main-effect hit is not a mechanism unless it (a) mediates timing, (b) replicates across competitions, (c) survives an RTM check")
    if len(med):
        print("\nVerdict: a quantifiable candidate mechanism exists; cross-competition replication and an RTM check decide.")
    else:
        print("\nVerdict: quantifiable openness features do not explain the timing threshold.")
        print("         The explanation rests on literature-based hypotheses (fatigue, late-game openness, tactics),")
        print("         to be stated as hypotheses rather than demonstrated mechanisms.")


if __name__ == "__main__":
    main()
