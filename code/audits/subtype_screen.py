"""Offensive / defensive / neutral substitution types as post-hoc mechanism candidates.

Line shift = sum(line_in) - sum(line_out), GK=0 < DF=1 < MF=2 < FW=3. Incoming
position = modal event position in the raw StatsBomb JSON (player_positions holds the
inherited slot). Six candidates screened with the mechanism_screen.py spec, BH within
family and jointly with the 36 family; fidelity check against mechanism_mine.csv.
Inputs:  data/{events_pass,events_goals,events_substitution,analysis_table,player_positions}.parquet, data/substitution_batches.csv, data/windows/*, results/tables/mechanism_mine.csv, $STATSBOMB_EVENTS_DIR/{<match_id>_events,00_info_player_stats_11_281}.json
Outputs: results/tables/subtype_screen.csv
Run: STATSBOMB_EVENTS_DIR=<dir> python code/audits/subtype_screen.py
"""

import json
import warnings
from collections import Counter
import os
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

np.random.seed(20260808)  # no random steps

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
TAB = ROOT / "results" / "tables"
RAW = Path(
    os.environ.get("STATSBOMB_EVENTS_DIR", "path/to/statsbomb_event_data/11_281_2023_24")
)
NX, NY = 6, 4
PL, PW = 120, 80

# position -> line: GK=0 < DF=1 < MF=2 < FW=3; wings as FW, wide midfield as MF
LINE = {
    "Goalkeeper": 0,
    "Right Back": 1, "Left Back": 1, "Right Wing Back": 1, "Left Wing Back": 1,
    "Center Back": 1, "Right Center Back": 1, "Left Center Back": 1,
    "Right Defensive Midfield": 2, "Center Defensive Midfield": 2, "Left Defensive Midfield": 2,
    "Right Center Midfield": 2, "Left Center Midfield": 2, "Center Midfield": 2,
    "Right Midfield": 2, "Left Midfield": 2,
    "Center Attacking Midfield": 2, "Right Attacking Midfield": 2, "Left Attacking Midfield": 2,
    "Right Wing": 3, "Left Wing": 3,
    "Center Forward": 3, "Right Center Forward": 3, "Left Center Forward": 3,
    "Secondary Striker": 3,
}
LINE_SENS = dict(LINE)  # sensitivity (i): wings as MF
LINE_SENS["Right Wing"] = 2
LINE_SENS["Left Wing"] = 2
# sensitivity (ii): primary_position uses British spelling
LINE_NOMINAL = {
    "Goalkeeper": 0,
    "Right Back": 1, "Left Back": 1, "Right Wing Back": 1, "Left Wing Back": 1,
    "Centre Back": 1, "Right Centre Back": 1, "Left Centre Back": 1,
    "Right Defensive Midfielder": 2, "Centre Defensive Midfielder": 2, "Left Defensive Midfielder": 2,
    "Right Centre Midfielder": 2, "Left Centre Midfielder": 2, "Centre Midfielder": 2,
    "Right Midfielder": 2, "Left Midfielder": 2,
    "Centre Attacking Midfielder": 2, "Right Attacking Midfielder": 2, "Left Attacking Midfielder": 2,
    "Right Wing": 3, "Left Wing": 3,
    "Centre Forward": 3, "Right Centre Forward": 3, "Left Centre Forward": 3,
    "Secondary Striker": 3,
}

# 10-role map, as in fig2_role_positions.py; formation composition vector
ROLE_MAP = {
    "Goalkeeper": "GK", "Right Back": "RB", "Right Wing Back": "RB",
    "Right Center Back": "CB", "Left Center Back": "CB", "Center Back": "CB",
    "Left Back": "LB", "Left Wing Back": "LB",
    "Right Defensive Midfield": "DM", "Left Defensive Midfield": "DM", "Center Defensive Midfield": "DM",
    "Right Center Midfield": "CM", "Left Center Midfield": "CM",
    "Center Attacking Midfield": "AM", "Right Attacking Midfield": "AM", "Left Attacking Midfield": "AM",
    "Right Wing": "RW", "Right Midfield": "RW", "Left Wing": "LW", "Left Midfield": "LW",
    "Center Forward": "ST", "Right Center Forward": "ST", "Left Center Forward": "ST"}
ROLES = ["GK", "RB", "CB", "LB", "DM", "CM", "AM", "RW", "LW", "ST"]


def zone_share(w):
    """As in mechanism_screen.py."""
    zx = np.clip((w.x / (PL / NX)).astype(int), 0, NX - 1)
    zy = np.clip((w.y / (PW / NY)).astype(int), 0, NY - 1)
    ex = np.clip((w.end_x / (PL / NX)).astype(int), 0, NX - 1)
    ey = np.clip((w.end_y / (PW / NY)).astype(int), 0, NY - 1)
    Z = np.zeros((24, 24))
    np.add.at(Z, ((zx * NY + zy).values, (ex * NY + ey).values), 1.0)
    np.fill_diagonal(Z, 0)
    t = Z / Z.sum() if Z.sum() > 0 else Z
    return t.sum(0) + t.sum(1)


def parse_ts(ts):
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def extract_raw(match_ids, need_players):
    """Scan raw event JSON: (1) modal event position per (match, player) for incoming players;
    (2) per (match, team) tactical-state timeline [(period, t_period_sec, type, 10-role vector)].
    need_players: {match_id: set(player_id)}, incoming players only."""
    played, tactics = {}, {}
    role_i = {r: i for i, r in enumerate(ROLES)}
    for mid in match_ids:
        f = RAW / f"{mid}_events.json"
        ev = json.load(open(f))
        want = need_players.get(mid, set())
        cnt = {}
        for e in ev:
            ty = e["type"]["name"]
            if ty in ("Starting XI", "Tactical Shift"):
                vec = np.zeros(len(ROLES))
                for x in e["tactics"]["lineup"]:
                    r = ROLE_MAP.get(x["position"]["name"])
                    if r:
                        vec[role_i[r]] += 1
                tactics.setdefault((mid, e["team"]["id"]), []).append(
                    (e["period"], parse_ts(e["timestamp"]), ty, vec))
                continue
            pl = e.get("player")
            if pl and pl["id"] in want and "position" in e:
                cnt.setdefault(pl["id"], Counter())[e["position"]["name"]] += 1
        for pid, c in cnt.items():
            played[(mid, pid)] = c.most_common(1)[0][0]
    return played, tactics


def main():
    # data, as in mechanism_screen.py
    passes = pd.read_parquet(DATA / "events_pass.parquet")
    net = passes[passes.outcome.isna() & ~passes.is_set_piece].copy()
    t1 = passes[passes.period == 1].groupby("match_id").t_period_sec.max()
    net["tc"] = np.where(net.period == 1, net.t_period_sec, net.match_id.map(t1) + net.t_period_sec)
    ni = {k: g for k, g in net.groupby(["match_id", "team_id"])}
    goals = pd.read_parquet(DATA / "events_goals.parquet")
    gd = {m: g for m, g in goals.groupby("match_id")}
    b = pd.read_csv(DATA / "substitution_batches.csv").set_index("batch_id")
    sp = pd.read_parquet(DATA / "windows" / "window_specs.parquet")
    stt = sp[(sp.kind == "time") & (sp.W_min == 15)].set_index(["batch_id", "side"])
    fl = pd.read_parquet(DATA / "windows" / "batch_flags.parquet")
    nontr = set(fl[(fl.W_min == 15) & ~fl.post_truncated].batch_id)
    at_idx = set(pd.read_parquet(DATA / "analysis_table.parquet", columns=["batch_id"]).batch_id)

    # position sources
    es = pd.read_parquet(DATA / "events_substitution.parquet")
    pos_out = {(m, p): q for m, p, q in zip(es.match_id, es.player_out_id, es.position_out)}
    pp = pd.read_parquet(DATA / "player_positions.parquet")
    pos_slot = {(m, p): q for m, p, q in zip(pp.match_id, pp.player_id, pp.position)}  # inherited slot, fallback only
    nominal = {r["player_id"]: r.get("primary_position")
               for r in json.load(open(RAW / "00_info_player_stats_11_281.json"))}

    need = {}
    for _, r in b.iterrows():
        need.setdefault(r.match_id, set()).update(int(x) for x in str(r.players_in_id).split("|"))
    print(f"scanning raw event JSON for {b.match_id.nunique()} matches (incoming positions + Tactical Shift)...")
    played, tactics = extract_raw(sorted(b.match_id.unique()), need)
    # tactical timeline to cumulative seconds, sorted
    tact_cum = {}
    for k, lst in tactics.items():
        mid = k[0]
        tl = [(t + (t1.get(mid, 2700.0) if per == 2 else 0.0), ty, vec) for per, t, ty, vec in lst]
        tact_cum[k] = sorted(tl, key=lambda x: x[0])

    # per batch, same sample filter as mechanism_screen.py
    rows = []
    n_fallback = n_gk = 0
    for bid in b.index:
        if bid not in nontr or (bid, "pre") not in stt.index or bid not in at_idx:
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
        gg = gd.get(r.match_id)
        margin = 0
        if gg is not None:
            bef = gg[gg.t_cum < pre.t_end]
            margin = (bef.team_id == r.team_id).sum() - (bef.team_id != r.team_id).sum()

        # (a)(b) line shift
        out_ids = [int(x) for x in str(r.players_out_id).split("|")]
        in_ids = [int(x) for x in str(r.players_in_id).split("|")]
        p_out = [pos_out[(r.match_id, p)] for p in out_ids]
        p_in = []
        for p in in_ids:
            q = played.get((r.match_id, p))
            if q is None:  # no events after coming on, use inherited slot
                q = pos_slot.get((r.match_id, p))
                n_fallback += 1
            p_in.append(q)
        n_gk += any(q == "Goalkeeper" for q in p_out + p_in)
        lift = sum(LINE[q] for q in p_in) - sum(LINE[q] for q in p_out)
        lift_s = sum(LINE_SENS[q] for q in p_in) - sum(LINE_SENS[q] for q in p_out)
        nom = [nominal.get(p) for p in in_ids]
        lift_nom = (sum(LINE_NOMINAL[q] for q in nom) - sum(LINE[q] for q in p_out)
                    if all(q in LINE_NOMINAL for q in nom) else np.nan)

        # (c) formation-change proxy: team Tactical Shift in (t_sub-60, t_sub+300]
        t_sub = r.t_sec_first + (t1.get(r.match_id, 2700.0) if r.period == 2 else 0.0)
        tl = tact_cum.get((r.match_id, r.team_id), [])
        before = [v for t, ty, v in tl if t <= t_sub - 60]
        inwin = [v for t, ty, v in tl if t_sub - 60 < t <= t_sub + 300 and ty == "Tactical Shift"]
        tshift = int(len(inwin) > 0)
        fchg = float(np.abs(inwin[-1] - before[-1]).sum()) if (inwin and before) else 0.0

        rows.append(dict(
            batch_id=bid, match_id=r.match_id, reorg=reorg, npass=min(len(wp), len(wq)),
            tnum={"strategic": 1, "regular": 2, "late": 3}[r.timing_group],
            timing_group=r.timing_group, n_subs=r.n_subs, minute=r.minute_first,
            margin=margin,
            lineshift_net=lift, lineshift_mean=lift / r.n_subs,
            sub_offensive=int(lift > 0), sub_defensive=int(lift < 0),
            tact_shift=tshift, formchg_L1=fchg,
            lineshift_net_sensWM=lift_s, lineshift_net_nominal=lift_nom))
    M = pd.DataFrame(rows)
    M["trailing"] = (M.margin < 0).astype(int)  # fidelity check
    print(f"sample {len(M)} batches (reference n=1490); incoming-position fallback {n_fallback}; "
          f"batches with a GK {n_gk}; nominal missing {M.lineshift_net_nominal.isna().sum()}\n")

    # distributions
    M["subtype"] = np.select([M.lineshift_net > 0, M.lineshift_net < 0],
                             ["offensive", "defensive"], "neutral")
    vc = M.subtype.value_counts()
    print("=== substitution-type distribution (main coding: actual position, Wing=FW) ===")
    print((pd.concat([vc, (vc / len(M) * 100).round(1)], axis=1, keys=["n", "%"])).to_string(), "\n")
    ct = (pd.crosstab(M.timing_group, M.subtype, normalize="index") * 100).round(1)
    ct = ct.reindex(index=["strategic", "regular", "late"],
                    columns=["offensive", "neutral", "defensive"], fill_value=0.0)
    print("share by timing group, % (row-normalised):")
    print(ct.to_string(), "\n")
    print("lineshift_net distribution:",
          {k: round(v, 2) for k, v in M.lineshift_net.describe()[["min", "25%", "50%", "75%", "max"]].items()})
    print(f"tact_shift=1 share: {M.tact_shift.mean():.3f}; formchg_L1>0 share: {(M.formchg_L1 > 0).mean():.3f}; "
          f"formchg_L1 mean (within >0): {M.loc[M.formchg_L1 > 0, 'formchg_L1'].mean():.2f}")
    sens_sub = np.select([M.lineshift_net_sensWM > 0, M.lineshift_net_sensWM < 0],
                         ["offensive", "defensive"], "neutral")
    nom_sub = np.select([M.lineshift_net_nominal > 0, M.lineshift_net_nominal < 0],
                        ["offensive", "defensive"], "neutral")
    print("sensitivity (i) Wing=MF types:", pd.Series(sens_sub).value_counts().to_dict(),
          "| agreement with main coding:", round((M.subtype == sens_sub).mean(), 3))
    print("sensitivity (ii) nominal types:", pd.Series(nom_sub).value_counts().to_dict(),
          "| agreement with main coding:", round((M.subtype == nom_sub).mean(), 3), "\n")

    # volume residualisation, as in mechanism_screen.py
    def zc(s):
        return (s - s.mean()) / s.std()
    M["reorg_z"] = zc(M.reorg)
    npz = zc(M.npass)
    Xv = pd.DataFrame({"a": npz, "b": npz ** 2, "c": zc(1 / np.sqrt(M.npass)), "d": zc(np.log(M.npass))})
    Xv["const"] = 1.0
    fitv = sm.OLS(M.reorg_z, Xv).fit()
    M["reorg_resid"] = M.reorg_z - fitv.predict(Xv)
    print(f"[volume control] reorg~flex(npass) R2={fitv.rsquared:.3f}\n")

    # screening spec, three regressions as in mechanism_screen.py
    def screen(p, dat):
        d = dat.dropna(subset=[p, "reorg_resid"]).copy()
        if len(d) < 50 or d[p].std() == 0:
            return None
        d["pz"] = zc(d[p])
        try:
            m = smf.mixedlm("reorg_resid ~ pz + tnum", d, groups=d.match_id).fit(reml=False)
            bm, pm = m.params.get("pz", np.nan), m.pvalues.get("pz", np.nan)
        except Exception:
            bm, pm = np.nan, np.nan
        try:
            mn = smf.mixedlm("reorg_z ~ pz + npass_z + tnum", d.assign(npass_z=zc(d.npass)),
                             groups=d.match_id).fit(reml=False)
            pm_naive = mn.pvalues.get("pz", np.nan)
        except Exception:
            pm_naive = np.nan
        try:
            mg = smf.mixedlm("reorg_resid ~ pz * tnum", d, groups=d.match_id).fit(reml=False)
            bg, pg = mg.params.get("pz:tnum", np.nan), mg.pvalues.get("pz:tnum", np.nan)
        except Exception:
            bg, pg = np.nan, np.nan
        return dict(predictor=p, n=len(d), beta_main=bm, p_main=pm, p_main_naive=pm_naive,
                    beta_inter=bg, p_inter=pg)

    # fidelity check: three original candidates vs mechanism_mine.csv
    old = pd.read_csv(TAB / "mechanism_mine.csv").set_index("predictor")
    print("=== fidelity check (recomputed vs mechanism_mine.csv) ===")
    ok = True
    for p in ["minute", "n_subs", "trailing"]:
        x = screen(p, M)
        db = abs(x["beta_main"] - old.loc[p, "beta_main"])
        dp = abs(x["p_main"] - old.loc[p, "p_main"])
        good = (db < 1e-6) and (dp < 1e-6)
        ok &= good
        print(f"  {p:10s} dbeta={db:.2e} dp={dp:.2e} {'OK' if good else 'MISMATCH'}")
    print(f"  -> spec replication {'passed' if ok else 'FAILED (do not trust results below)'}\n")

    # new candidates; posthoc family enters BH, sensitivity is robustness only
    NEW = ["sub_offensive", "sub_defensive", "lineshift_net", "lineshift_mean",
           "tact_shift", "formchg_L1"]
    res = [dict(screen(p, M), family="posthoc_tactical") for p in NEW]
    for p in ["lineshift_net_sensWM", "lineshift_net_nominal"]:
        x = screen(p, M)
        if x:
            res.append(dict(x, family="sensitivity"))
    R = pd.DataFrame(res)

    # BH: (i) within posthoc family; (ii) jointly with the 36 family
    prim = R.family == "posthoc_tactical"
    R.loc[prim, "q_within"] = multipletests(R.loc[prim, "p_main"].fillna(1), method="fdr_bh")[1]
    R.loc[prim, "q_inter_within"] = multipletests(R.loc[prim, "p_inter"].fillna(1), method="fdr_bh")[1]
    pj = np.concatenate([old.p_main.values, R.loc[prim, "p_main"].fillna(1).values])
    qj = multipletests(pj, method="fdr_bh")[1]
    R.loc[prim, "q_joint36"] = qj[len(old):]
    pji = np.concatenate([old.p_inter.values, R.loc[prim, "p_inter"].fillna(1).values])
    qji = multipletests(pji, method="fdr_bh")[1]
    R.loc[prim, "q_inter_joint36"] = qji[len(old):]
    R.loc[prim, "survive_joint"] = R.loc[prim, "q_joint36"] < 0.05
    R.loc[prim, "survive_inter_joint"] = R.loc[prim, "q_inter_joint36"] < 0.05
    # does the original 36-family survivor list change
    old_surv = set(old.index[old.q_main < 0.05])
    new_surv36 = set(old.index[qj[: len(old)] < 0.05])
    thr = old[old.q_main < 0.05].p_main.max() if (old.q_main < 0.05).any() else np.nan
    print(f"36-family survivors (q_main<0.05): {sorted(old_surv)} (effective BH threshold p<={thr:.2e})")
    print("after adding the 6 new candidates the 36-family survivor list "
          + ("is unchanged" if old_surv == new_surv36 else f"becomes {sorted(new_surv36)}") + "\n")

    print("=== new candidate screen (post hoc, reported separately from the 36 family) ===")
    print(f"  {'predictor':22s} {'n':>5s} {'b_main':>8s} {'p_main':>8s} {'q_joint':>8s} "
          f"{'b_int':>8s} {'p_int':>8s} {'q_int_j':>8s}  verdict")
    for _, x in R.iterrows():
        tag = ("survives" if x.get("survive_joint") else "null") if x.family == "posthoc_tactical" else "(sensitivity)"
        qj_ = f"{x.q_joint36:8.3f}" if pd.notna(x.get("q_joint36")) else "     ---"
        qi_ = f"{x.q_inter_joint36:8.3f}" if pd.notna(x.get("q_inter_joint36")) else "     ---"
        print(f"  {x.predictor:22s} {x.n:5d} {x.beta_main:+8.3f} {x.p_main:8.3f} {qj_} "
              f"{x.beta_inter:+8.3f} {x.p_inter:8.3f} {qi_}  {tag}")

    TAB.mkdir(parents=True, exist_ok=True)
    R.to_csv(TAB / "subtype_screen.csv", index=False)
    print(f"\nsaved {TAB / 'subtype_screen.csv'}")


if __name__ == "__main__":
    main()
