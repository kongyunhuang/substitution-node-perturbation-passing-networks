"""
Continuous signal-to-noise curve along substitution minute in sliding windows (Figure 2d).

Sliding-window SNR of the teammates-only DiD zone response versus substitution minute, at a
fixed bootstrap sample size (N_FIX) so windows are comparable. La Liga 2023/24 plus each
open-data competition separately (pooling competitions cancels their directions).
Inputs: data/{events_pass,did_controls}.parquet, data/substitution_batches.csv, data/windows/*.parquet,
        data/opendata/*.parquet (via code/replication/open_data_two_layer.py)
Outputs: results/tables/timing_curve.csv (source, center_min, n, snr, lo, hi)
Run: python code/figures/fig2_timing_curve.py
"""

import importlib.util
import warnings
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]  # <repo>/code

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OD = ROOT / "data" / "opendata"
TAB = ROOT / "results" / "tables"
NX, NY = 6, 4
PL, PW = 120, 80
W = 900.0
SEED = 20260725

WIDTH, STEP = 10.0, 2.5      # window width / step (min)
N_FIX, MIN_N = 73, 95        # fixed bootstrap size; min batches per window
N_BOOT, N_NULL = 600, 400


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


def laliga_vectors():
    """La Liga: (substitution minute, teammates-only DiD vector), same pipeline as self_exclusion_check.py."""
    passes = pd.read_parquet(DATA / "events_pass.parquet")
    net = passes[passes.outcome.isna() & ~passes.is_set_piece].copy()
    t1 = passes[passes.period == 1].groupby("match_id").t_period_sec.max()
    net["tc"] = np.where(net.period == 1, net.t_period_sec,
                         net.match_id.map(t1) + net.t_period_sec)
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

    mins, vecs, groups = [], [], []
    for bid in nontr:
        if bid not in ctrl.index or (bid, "pre") not in st.index:
            continue
        r = b.loc[bid]
        pre, post = st.loc[(bid, "pre")], st.loc[(bid, "post")]
        wp = winp(r.match_id, r.team_id, pre.t_start, pre.t_end, False)
        wq = winp(r.match_id, r.team_id, post.t_start, post.t_end, True)
        c = ctrl.loc[bid]
        tc = c.control_t_center
        wcp = winp(int(c.control_match_id), int(c.control_team_id), tc - W, tc, False)
        wcq = winp(int(c.control_match_id), int(c.control_team_id), tc, tc + W, True)
        if any(w is None or len(w) < 10 for w in [wp, wq, wcp, wcq]):
            continue
        swapped = ([int(x) for x in str(r.players_out_id).split("|")]
                   + [int(x) for x in str(r.players_in_id).split("|")])
        d = (zvec(wq, swapped) - zvec(wp, swapped)) - (zvec(wcq) - zvec(wcp))
        mins.append(float(r.minute_first))
        vecs.append(d)
        groups.append(r.timing_group)
    return np.array(mins), np.array(vecs), np.array(groups)


def opendata_vectors():
    """Open-data competitions: one (minutes, teammates-only DiD vectors) pair per competition."""
    spec = importlib.util.spec_from_file_location("od33", CODE_ROOT / "replication" / "open_data_two_layer.py")
    m33 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m33)
    od = m33.od
    COMPS = {"WC22": (43, 106), "Euro24": (55, 282), "Euro20": (55, 43), "ISL": (1238, 108),
             "WSL": (37, 90), "Bund23": (9, 281), "Bund15": (9, 27), "LaLiga15": (11, 27),
             "PL15": (2, 27), "SerieA15": (12, 27)}

    def zvec_excl(w, names):
        return od.zvec(w[~w.player.isin(names) & ~w.pass_recipient.isin(names)])

    per = {}
    for name, (cid, sid) in COMPS.items():
        if not (OD / f"{cid}_{sid}.parquet").exists():
            continue
        df = od.fetch(cid, sid)
        net, _ = od.prep(df)
        ni = {k: g.sort_values("tc") for k, g in net.groupby(["match_id", "team"])}
        keys = list(ni.keys())
        sub_times = {k: np.sort(g.minute.values * 60.0)
                     for k, g in df[df.type == "Substitution"].groupby(["match_id", "team"])}
        match_end = {m: g.minute.max() * 60 + g.second.max() for m, g in df.groupby("match_id")}
        B = m33.batches_named(df)
        rng = np.random.default_rng(od.SEED)
        mins, vecs = [], []
        for bb in B:
            key, t0 = (bb["match_id"], bb["team"]), bb["t0"]
            if t0 + W > match_end.get(bb["match_id"], 0):
                continue
            g = ni.get(key)
            if g is None:
                continue
            wp = g[(g.tc >= t0 - W) & (g.tc < t0)]
            wq = g[(g.tc > t0) & (g.tc <= t0 + W)]
            swapped = list(bb["out"]) + list(bb["in"])
            zep, zeq = zvec_excl(wp, swapped), zvec_excl(wq, swapped)
            if zep is None or zeq is None:
                continue
            cands = [k for k in keys if k != key]
            rng.shuffle(cands)
            ctrl = None
            for ck in cands[:200]:
                for off in [0, 60, -60, 120, -120]:
                    tc = t0 + off
                    if tc - W < 0 or tc + W > match_end.get(ck[0], 0):
                        continue
                    if not od.clean(sub_times, ck, tc - W, tc + W):
                        continue
                    gc = ni.get(ck)
                    cp = gc[(gc.tc >= tc - W) & (gc.tc < tc)]
                    cq = gc[(gc.tc > tc) & (gc.tc <= tc + W)]
                    if od.zvec(cp) is not None and od.zvec(cq) is not None:
                        ctrl = od.zvec(cq) - od.zvec(cp)
                        break
                if ctrl is not None:
                    break
            if ctrl is None:
                continue
            mins.append(float(bb["minute"]))
            vecs.append((zeq - zep) - ctrl)
        per[name] = (np.array(mins), np.array(vecs))
        print(f"    {name:10s} usable batches {len(mins)}")
    return per


def snr_ci(arr, N, rng):
    """Bootstrap mean magnitude over sign-flip null magnitude, plus bootstrap percentile band."""
    mags = np.array([np.linalg.norm(arr[rng.integers(len(arr), size=N)].mean(0))
                     for _ in range(N_BOOT)])
    nf = []
    for _ in range(N_NULL):
        idx = rng.integers(len(arr), size=N)
        sgn = rng.choice([1, -1], size=len(arr))
        nf.append(np.linalg.norm((arr[idx] * sgn[idx, None]).mean(0)))
    base = np.mean(nf)
    return mags.mean() / base, np.percentile(mags, 2.5) / base, np.percentile(mags, 97.5) / base


def curve(mins, vecs, rng, label):
    lo_edge, hi_edge = 45.0, 95.0
    rows = []
    c = lo_edge + WIDTH / 2
    while c <= hi_edge - WIDTH / 2:
        m = (mins >= c - WIDTH / 2) & (mins < c + WIDTH / 2)
        n = int(m.sum())
        if n >= MIN_N:
            s, lo, hi = snr_ci(vecs[m], N_FIX, rng)
            rows.append(dict(source=label, center_min=c, n=n, snr=s, lo=lo, hi=hi))
        c += STEP
    return rows


def main():
    rng = np.random.default_rng(SEED)
    print("[1/2] La Liga 2023/24 ...")
    lm, lv, lg = laliga_vectors()
    print(f"    usable batches {len(lm)}  (by group: "
          f"{ {g: int((lg == g).sum()) for g in ['strategic', 'regular', 'late']} })")
    rows = curve(lm, lv, rng, "La Liga 2023/24")
    print("[2/2] open-data competitions, one curve each ...")
    per = opendata_vectors()
    for name, (om, ov) in per.items():
        r = curve(om, ov, rng, name)
        if len(r) >= 4:                     # need >= 4 windows for a curve
            rows += r
        else:
            print(f"    (skipped {name}: only {len(r)} windows with n>={MIN_N})")
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "timing_curve.csv", index=False)
    print("\n=== timing_curve.csv ===")
    for src, g in df.groupby("source", sort=False):
        print(f"  {src}")
        for _, r in g.iterrows():
            print(f"    center {r.center_min:>5.1f}'  n={int(r.n):>4}  "
                  f"SNR={r.snr:.2f}  [{r.lo:.2f}, {r.hi:.2f}]")


if __name__ == "__main__":
    main()
