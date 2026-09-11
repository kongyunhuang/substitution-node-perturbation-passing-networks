"""Early-group split (halftime vs in-play), cut-point sensitivity, halftime placebo.

A: early group split at halftime (period 2, minute 45, t_sec <= 60) vs 46-60' in play.
B: early/mid cut moved to 55/58/60/62/65. C: same split in 10 open-data competitions.
D: raw pre/post dz across halftime in substitution-free windows vs 60' in play.
Inputs:  data/{events_pass,events_substitution}.parquet, data/substitution_batches.csv, data/did_controls.parquet, data/windows/*, data/opendata/*.parquet
Outputs: results/tables/early_split.csv
Run: python code/audits/early_group_split.py
"""

import importlib.util
import warnings
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]  # <repo>/code

import numpy as np
import pandas as pd
from scipy.stats import binomtest

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OD_DIR = DATA / "opendata"
TAB = ROOT / "results" / "tables"
NX, NY, PL, PW = 6, 4, 120, 80
SEED = 20260613
W = 900.0
N_HEADLINE = 73  # headline equal-sample size (= late-group n)
CUTS = [55, 58, 60, 62, 65]  # early/mid cut variants (60 = reference); mid/late stays at 75'

# core functions as in control_robustness.py


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


def snr(arr, N, n_boot=800, n_flip=400):
    """Headline estimator; fresh rng(SEED) per call, so MC-level (~0.02) differences
    from the single-stream headline run."""
    rng = np.random.default_rng(SEED)
    mags = [np.linalg.norm(arr[rng.integers(len(arr), size=N)].mean(0)) for _ in range(n_boot)]
    nf = []
    for _ in range(n_flip):
        idx = rng.integers(len(arr), size=N)
        sgn = rng.choice([1, -1], size=len(arr))
        nf.append(np.linalg.norm((arr[idx] * sgn[idx, None]).mean(0)))
    obs = np.mean(mags)
    return obs / np.mean(nf), 1 - (obs > np.array(nf)).mean(), obs, float(np.mean(nf))


ROWS = []


def emit(analysis, scope, layer, n, N, s=np.nan, p=np.nan, obs=np.nan, null=np.nan, note=""):
    ROWS.append(dict(analysis=analysis, scope=scope, layer=layer, n=n, N_resample=N,
                     snr_excl=(round(s, 3) if np.isfinite(s) else np.nan),
                     p=(round(p, 4) if np.isfinite(p) else np.nan),
                     obs_mag=(round(obs, 5) if np.isfinite(obs) else np.nan),
                     null_mag=(round(null, 5) if np.isfinite(null) else np.nan), note=note))


def run_snr(analysis, scope, layer, arr, N_req, note=""):
    n = len(arr)
    if n < 10:
        emit(analysis, scope, layer, n, np.nan, note=note + " n<10 unreliable, skipped")
        return np.nan
    N = min(N_req, n)
    tag = note if N == N_req else (note + f" layer n={n}<{N_req}, using N={N}").strip()
    s, p, obs, null = snr(arr, N)
    emit(analysis, scope, layer, n, N, s, p, obs, null, tag)
    print(f"  [{analysis}|{scope}] {layer:22s} n={n:4d} N={N:3d} SNR={s:.3f} p={p:.4f}"
          f" {'*' if p < .05 else 'ns'} {tag}")
    return s


# A/B: main sample, batch DiD vectors with minute / halftime labels


def build_main_recs():
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

    recs = []  # dict(minute_first, is_ht, dz)
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
        dz = (zvec(wq, swapped) - zvec(wp, swapped)) - (zvec(wcq) - zvec(wcp))
        dz_raw = zvec(wq, swapped) - zvec(wp, swapped)  # no DiD, placebo reference
        is_ht = (r.period == 2) and (r.minute_first == 45) and (r.t_sec_first <= 60)
        recs.append(dict(minute=r.minute_first, is_ht=is_ht, dz=dz, dz_raw=dz_raw))
    return recs, ni, t1, net


def part_a(recs):
    print("\n=== A. main-sample split (common N=73, teammates-only DiD) ===")
    dz = lambda sel: np.array([r["dz"] for r in recs if sel(r)])
    ht = dz(lambda r: r["minute"] <= 60 and r["is_ht"])
    ip = dz(lambda r: r["minute"] <= 60 and not r["is_ht"])
    n_early = len(ht) + len(ip)
    print(f"  early sample n={n_early}, halftime layer n={len(ht)}"
          f" ({len(ht)/n_early*100:.1f}%), 46-60' in-play layer n={len(ip)}")
    run_snr("main_split", "laliga", "early_halftime", ht, N_HEADLINE,
            note="rule: period=2 & minute_first=45 & t_sec_first<=60")
    run_snr("main_split", "laliga", "early_inplay_46_60", ip, N_HEADLINE)
    run_snr("main_split", "laliga", "early_all(sanity)", dz(lambda r: r["minute"] <= 60), N_HEADLINE,
            note="compare control_robustness all/strategic=1.052")
    run_snr("main_split", "laliga", "mid_61_75(ref)", dz(lambda r: 60 < r["minute"] <= 75), N_HEADLINE,
            note="reference row; control_robustness=1.561")
    run_snr("main_split", "laliga", "late_gt75(ref)", dz(lambda r: r["minute"] > 75), N_HEADLINE,
            note="reference row; control_robustness=1.594")


def part_b(recs):
    print("\n=== B. cut-point sensitivity (early/mid = 55/58/60/62/65; 75' fixed) ===")
    for cut in CUTS:
        groups = {f"early_le{cut}": np.array([r["dz"] for r in recs if r["minute"] <= cut]),
                  f"mid_{cut+1}_75": np.array([r["dz"] for r in recs if cut < r["minute"] <= 75]),
                  "late_gt75": np.array([r["dz"] for r in recs if r["minute"] > 75])}
        ns = {k: len(v) for k, v in groups.items()}
        n_min = min(ns.values())
        print(f"  -- cut={cut}'  group n={ns}  Nmin={n_min} --")
        for k, arr in groups.items():
            run_snr("cutpoint", f"cut{cut}_Nmin", k, arr, n_min)
            if n_min != N_HEADLINE:
                run_snr("cutpoint", f"cut{cut}_N73", k, arr, N_HEADLINE)
            else:
                emit("cutpoint", f"cut{cut}_N73", k, ns[k], n_min, note="same as Nmin variant (Nmin=73)")


# C: cross-competition replication, cross_competition_threshold.did_lists plus layer labels

spec33 = importlib.util.spec_from_file_location("od33", CODE_ROOT / "replication" / "open_data_two_layer.py")
m33 = importlib.util.module_from_spec(spec33)
spec33.loader.exec_module(m33)
od = m33.od  # fetch_open_data module

COMPS = {"WC22": (43, 106), "Euro24": (55, 282), "Euro20": (55, 43), "ISL": (1238, 108),
         "WSL": (37, 90), "Bund23": (9, 281), "Bund15": (9, 27), "LaLiga15": (11, 27),
         "PL15": (2, 27), "SerieA15": (12, 27)}
LAYERS = ["halftime", "inplay_46_60", "regular", "late"]


def od_layered_did(cid, sid):
    """cross_competition_threshold.did_lists with the early group split in two;
    same rng stream, so same control assignment. Halftime = first sub at minute 45
    (open-data clock tc = minute*60 + second, period 2 starts at 45'). Known
    approximation: first-half stoppage time overlaps the start of period 2 on this
    clock, so the halftime post window may include some stoppage-time passes."""
    df = od.fetch(cid, sid)
    net, _ = od.prep(df)
    ni = {k: g.sort_values("tc") for k, g in net.groupby(["match_id", "team"])}
    keys = list(ni.keys())
    sub_times = {k: np.sort(g.minute.values * 60.0)
                 for k, g in df[df.type == "Substitution"].groupby(["match_id", "team"])}
    match_end = {m: g.minute.max() * 60 + g.second.max() for m, g in df.groupby("match_id")}
    B = m33.batches_named(df)
    rng = np.random.default_rng(od.SEED)
    excl = {g: [] for g in LAYERS}
    for b in B:
        key, t0 = (b["match_id"], b["team"]), b["t0"]
        if t0 + W > match_end.get(b["match_id"], 0):
            continue
        g = ni.get(key)
        if g is None:
            continue
        wp = g[(g.tc >= t0 - W) & (g.tc < t0)]
        wq = g[(g.tc > t0) & (g.tc <= t0 + W)]
        swapped = list(b["out"]) + list(b["in"])
        zip_, ziq = od.zvec(wp), od.zvec(wq)
        zep = m33.od.zvec(wp[~wp.player.isin(swapped) & ~wp.pass_recipient.isin(swapped)])
        zeq = m33.od.zvec(wq[~wq.player.isin(swapped) & ~wq.pass_recipient.isin(swapped)])
        if any(v is None for v in (zip_, ziq, zep, zeq)):
            continue
        cands = [k for k in keys if k != key]
        rng.shuffle(cands)
        ctrl = None
        for ck in cands[:200]:
            for off in [0, 60, -60, 120, -120]:
                tcc = t0 + off
                if tcc - W < 0 or tcc + W > match_end.get(ck[0], 0):
                    continue
                if not od.clean(sub_times, ck, tcc - W, tcc + W):
                    continue
                gc = ni.get(ck)
                cp = gc[(gc.tc >= tcc - W) & (gc.tc < tcc)]
                cq = gc[(gc.tc > tcc) & (gc.tc <= tcc + W)]
                if od.zvec(cp) is not None and od.zvec(cq) is not None:
                    ctrl = od.zvec(cq) - od.zvec(cp)
                    break
            if ctrl is not None:
                break
        if ctrl is None:
            continue
        if b["minute"] <= 60:
            tg = "halftime" if b["minute"] == 45 else "inplay_46_60"
        else:
            tg = "regular" if b["minute"] <= 75 else "late"
        excl[tg].append((zeq - zep) - ctrl)
    return {g: np.array(excl[g]) for g in LAYERS}


def part_c():
    print("\n=== C. cross-competition replication (10 open-data competitions, early split in two, teammates-only) ===")
    counts = {k: [] for k in ["late>ht", "late>ip", "mid>ht", "mid>ip"]}
    for name, (cid, sid) in COMPS.items():
        if not (OD_DIR / f"{cid}_{sid}.parquet").exists():
            print(f"  {name}: cache missing, skipped")
            continue
        ex = od_layered_did(cid, sid)
        ns = {g: len(ex[g]) for g in LAYERS}
        usable = [n for n in ns.values() if n >= 10]
        N = min(usable) if usable else 0
        print(f"  {name}: layer n={ns}  equal-sample N={N}")
        s = {}
        for g in LAYERS:
            note = "" if ns[g] >= 30 else "thin n, unreliable"
            s[g] = run_snr("opendata_split", name, g, ex[g], N, note=note)
        fin = lambda *gs: all(np.isfinite(s.get(g, np.nan)) for g in gs)
        if fin("late", "halftime"):
            counts["late>ht"].append(s["late"] > s["halftime"])
        if fin("regular", "halftime"):
            counts["mid>ht"].append(s["regular"] > s["halftime"])
        if fin("late", "inplay_46_60"):
            counts["late>ip"].append(s["late"] > s["inplay_46_60"])
        if fin("regular", "inplay_46_60"):
            counts["mid>ip"].append(s["regular"] > s["inplay_46_60"])
    print("  -- direction counts (competitions with layer n>=10) --")
    for k, v in counts.items():
        if v:
            bt = binomtest(sum(v), len(v))
            print(f"    {k}: {sum(v)}/{len(v)} (binomial p={bt.pvalue:.4f})")
            emit("opendata_split", "pooled", f"direction_{k}", len(v), np.nan,
                 note=f"{sum(v)}/{len(v)}, binom p={bt.pvalue:.4f}")


# D: halftime placebo, pre/post dz in substitution-free windows


def part_d(ni, t1, net):
    print("\n=== D. halftime placebo (raw dz in substitution-free windows; all 3528 sub events) ===")
    subs = pd.read_parquet(DATA / "events_substitution.parquet")
    tp = np.where(subs.period == 1, subs.minute * 60 + subs.second,
                  (subs.minute - 45) * 60 + subs.second)
    subs = subs.assign(tc=np.where(subs.period == 1, tp, subs.match_id.map(t1) + tp))
    sub_tc = {k: np.sort(g.tc.values) for k, g in subs.groupby(["match_id", "team_id"])}
    t2 = net[net.period == 2].groupby("match_id").t_period_sec.max()

    def no_sub(mid, tid, lo, hi):
        arr = sub_tc.get((mid, tid))
        if arr is None:
            return True
        i = np.searchsorted(arr, lo)
        return i >= len(arr) or arr[i] > hi

    def collect(center_of):
        out = []
        for (mid, tid), g in ni.items():
            c = center_of(mid)
            if c is None:
                continue
            if not no_sub(mid, tid, c - W, c + W):
                continue
            wp = g[(g.tc >= c - W) & (g.tc < c)]
            wq = g[(g.tc > c) & (g.tc <= c + W)]
            if len(wp) < 10 or len(wq) < 10:
                continue
            out.append(zvec(wq) - zvec(wp))
        return np.array(out)

    ht_c = lambda mid: t1.get(mid)  # centre = halftime whistle; pre = last 15' H1, post = first 15' H2
    ip_c = lambda mid: (t1.get(mid) + 900.0
                        if mid in t2.index and t2[mid] >= 1800.0 else None)  # 60' in play
    plc_ht = collect(ht_c)
    plc_ip = collect(ip_c)
    run_snr("placebo", "laliga", "placebo_ht_nosub", plc_ht, N_HEADLINE,
            note="pre=last 15' H1, post=first 15' H2, no team sub within +/-15min")
    run_snr("placebo", "laliga", "placebo_inplay60_nosub", plc_ip, N_HEADLINE,
            note="centre=60' (inside H2), same no-sub rule")
    return plc_ht, plc_ip


def main():
    recs, ni, t1, net = build_main_recs()
    part_a(recs)
    # raw dz of the treated halftime layer (teammates-only, no DiD), same scale as placebo
    ht_raw = np.array([r["dz_raw"] for r in recs if r["minute"] <= 60 and r["is_ht"]])
    part_b(recs)
    part_c()
    part_d(ni, t1, net)
    run_snr("placebo", "laliga", "treated_ht_raw_excl", ht_raw, N_HEADLINE,
            note="treated halftime layer, teammates-only, no DiD; same scale as placebo_ht")
    df = pd.DataFrame(ROWS)
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / "early_split.csv", index=False)
    print(f"\nsaved {TAB / 'early_split.csv'} ({len(df)} rows)")
    print("read: A both layers SNR~1, p ns; B early~1 < mid/late at every cut;")
    print("      C cross-competition direction counts; D placebo_ht vs placebo_inplay60 vs treated_ht_raw")


if __name__ == "__main__":
    main()
