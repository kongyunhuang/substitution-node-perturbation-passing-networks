"""Short-window collapse: predicted vs observed, WSL sampling, threshold change-point fits.

P1: 1/m noise-term prediction vs observed SNR at W=10/12/15 (direction agrees, magnitude does not).
P2: teammates-only passes per window m for WSL vs La Liga; noise-floor lift, equivalent W.
P3: constant/step/hinge/step*boxcar/logistic fits to the PL15 timing curve, AICc, batch bootstrap.
Inputs:  results/tables/{cross_gradient_clean,mvp_multiscale,timing_curve,convergence_rho_pre}.csv, data/opendata/{37_90,11_27,2_27}.parquet, data/{events_pass.parquet,substitution_batches.csv,windows/*}
Outputs: results/tables/wsl_collapse_changepoint.csv
Run: python code/audits/window_collapse_changepoint.py [--nboot 200] [--skip-boot] [--cache DIR]
"""

import argparse
import importlib.util
import warnings
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]  # <repo>/code

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OD_DIR = DATA / "opendata"
TAB = ROOT / "results" / "tables"

SEED_BOOT = 20260808
NBOOT_DEFAULT = 200
N_FIX = 73                # fixed bootstrap size, as in the timing curve (= La Liga min group n)
MIN_N_WIN = 95            # min batches per sliding window
WIDTH, STEP = 10.0, 2.5   # sliding window width / step, minutes
NB_IN, NN_IN = 200, 150   # inner SNR resamples in the bootstrap (timing curve used 600/400; speed only)
TIMLAB = {"strategic": "early", "regular": "mid", "late": "late"}

ROWS = []  # (block, item, group, metric, value)


def emit(block, item, group, metric, value):
    ROWS.append(dict(block=block, item=item, group=group, metric=metric,
                     value=float(value) if np.isfinite(float(value)) else np.nan))


def load_od_modules():
    """Load od and batches_named without running main()."""
    spec = importlib.util.spec_from_file_location("od33", CODE_ROOT / "replication" / "open_data_two_layer.py")
    m33 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m33)
    return m33.od, m33


# P1
def invert_r(snr, N):
    """SNR^2 = (N r + 1)/(r + 1) -> r = (SNR^2 - 1)/(N - SNR^2); clipped at 0."""
    s2 = snr ** 2
    return max((s2 - 1) / (N - s2), 0.0)


def predict_snr(r, N):
    return np.sqrt((N * r + 1) / (r + 1))


def p1_pred_vs_obs():
    print("\n" + "=" * 78)
    print("P1  predicted vs observed: 1/m noise term vs observed SNR")
    print("=" * 78)
    base = pd.read_csv(TAB / "cross_gradient_clean.csv")
    multi = pd.read_csv(TAB / "mvp_multiscale.csv")
    rho = pd.read_csv(TAB / "convergence_rho_pre.csv", index_col=0)
    for df in (base, multi):
        df["N"] = df.groupby([c for c in ("config", "comp") if c in df.columns])["n"].transform("min")

    per = []
    for cfg, W in [("W12_6x4", 12.0), ("W10_6x4", 10.0)]:
        for _, m in multi[multi.config == cfg].iterrows():
            b = base[(base.comp == m.comp) & (base.timing == m.timing)]
            if b.empty:
                continue
            b = b.iloc[0]
            r15 = invert_r(b.snr_excl, b.N)
            pred = predict_snr(r15 * (W / 15.0), m.N)
            rw_obs = invert_r(m.snr_excl, m.N)
            per.append(dict(config=cfg, W=W, comp=m.comp, timing=m.timing,
                            snr15=b.snr_excl, r15=r15, n15=b.n,
                            snr_obs=m.snr_excl, snr_pred=pred, n_w=m.n,
                            ret_obs=rw_obs / r15 if r15 > 0.005 else np.nan,
                            ret_pred=W / 15.0))
    per = pd.DataFrame(per)

    w15n = base.set_index(["comp", "timing"]).n
    hdr = f"{'W':>4} {'group':>6} | {'SNR obs':>8} {'SNR pred':>8} {'obs<pred':>9} | {'ret obs':>10} {'ret pred':>10} {'gap x':>8}"
    for cfg, W in [("W15_6x4", 15.0), ("W12_6x4", 12.0), ("W10_6x4", 10.0)]:
        print(f"\n--- {cfg} (rho_dens P/Z = {rho.loc[int(W), 'density_P']:.3f}/{rho.loc[int(W), 'density_Z']:.3f}) ---")
        print(hdr)
        for tg in ["strategic", "regular", "late"]:
            lab = TIMLAB[tg]
            if cfg == "W15_6x4":  # baseline: pred = obs, retention = 1
                d = base[base.timing == tg]
                obs = pred_m = d.snr_excl.mean()
                nlt, ret_o, ret_p = np.nan, 1.0, 1.0
                nn = len(d)
                infl = 1.0
            else:
                d = per[(per.config == cfg) & (per.timing == tg)]
                obs, pred_m = d.snr_obs.mean(), d.snr_pred.mean()
                nlt = int((d.snr_obs < d.snr_pred).sum())
                ret_o, ret_p = d.ret_obs.median(), W / 15.0
                nn = len(d)
                infl = (multi[(multi.config == cfg)].set_index(["comp", "timing"]).n / w15n) \
                    .xs(tg, level="timing").mean()
            gap = ((1 - ret_o) / (1 - ret_p)) if (cfg != "W15_6x4" and np.isfinite(ret_o)) else np.nan
            print(f"{cfg[1:3]:>4} {lab:>6} | {obs:8.3f} {pred_m:8.3f} "
                  f"{('%d/%d' % (nlt, nn)) if np.isfinite(nlt) else '   -':>9} | "
                  f"{ret_o if np.isfinite(ret_o) else np.nan:10.2f} {ret_p:10.2f} "
                  f"{gap if np.isfinite(gap) else np.nan:8.1f}")
            nret = int(d.ret_obs.notna().sum()) if cfg != "W15_6x4" else nn
            for met, val in [("snr_obs_mean", obs), ("snr_pred_mean", pred_m),
                             ("n_obs_lt_pred", nlt), ("n_comps", nn),
                             ("retention_obs_median", ret_o), ("retention_pred", ret_p),
                             ("n_retention_comps", nret),
                             ("power_loss_gap_factor", gap),
                             ("n_inflation_vs_W15_mean", infl)]:
                emit("P1_pred_vs_obs", cfg, lab, met, val if val is not None else np.nan)
        emit("P1_pred_vs_obs", cfg, "all", "rho_density_P", rho.loc[int(W), "density_P"])
        emit("P1_pred_vs_obs", cfg, "all", "rho_density_Z", rho.loc[int(W), "density_Z"])

    # sign survival: noise term predicts unchanged group order
    print("\nsign survival (mid>early / late>early, 10 competitions):")
    for cfg in ["W12_6x4", "W10_6x4"]:
        d = per[per.config == cfg].pivot(index="comp", columns="timing",
                                         values=["snr_obs", "snr_pred"])
        for kind, tag in [("snr_pred", "pred"), ("snr_obs", "obs")]:
            mid = int((d[(kind, "regular")] > d[(kind, "strategic")]).sum())
            late = int((d[(kind, "late")] > d[(kind, "strategic")]).sum())
            n = len(d)
            print(f"  {cfg} {tag:4s}: mid>early {mid}/{n}, late>early {late}/{n}")
            emit("P1_pred_vs_obs", cfg, "all", f"sign_mid_gt_early_{tag}", mid)
            emit("P1_pred_vs_obs", cfg, "all", f"sign_late_gt_early_{tag}", late)
    print("\n[P1] direction/order: agree (both degrade below the convergence plateau, predicted group order mostly survives)")
    print("[P1] magnitude: disagree; mid group at W10 retains 0.14 vs predicted 0.67, gap 3-5x;")
    print("     late group also shows batch-count inflation (2.59x at W10), not a noise mechanism")
    return per


# P2
def m_counts_opendata(od, m33, cid, sid, Ws):
    """Open-data competition: teammates-only completed passes m per pre/post window, non-truncated batches."""
    df = od.fetch(cid, sid)
    net, _ = od.prep(df)
    ni = {k: g.sort_values("tc") for k, g in net.groupby(["match_id", "team"])}
    match_end = {m: g.minute.max() * 60 + g.second.max() for m, g in df.groupby("match_id")}
    B = m33.batches_named(df)
    out = {W: [] for W in Ws}
    nmatch = df.match_id.nunique()
    for W in Ws:
        for b in B:
            key, t0 = (b["match_id"], b["team"]), b["t0"]
            if t0 + W > match_end.get(b["match_id"], 0):
                continue
            g = ni.get(key)
            if g is None:
                continue
            sw = set(b["out"]) | set(b["in"])
            wp = g[(g.tc >= t0 - W) & (g.tc < t0)]
            wq = g[(g.tc > t0) & (g.tc <= t0 + W)]
            mp = int((~wp.player.isin(sw) & ~wp.pass_recipient.isin(sw)).sum())
            mq = int((~wq.player.isin(sw) & ~wq.pass_recipient.isin(sw)).sum())
            out[W].append((mp, mq))
    return out, nmatch


def m_counts_laliga_main(Ws):
    """La Liga 2023/24 main sample: window_specs (kind=time), non-truncated, teammates-only m."""
    passes = pd.read_parquet(DATA / "events_pass.parquet")
    net = passes[passes.outcome.isna() & ~passes.is_set_piece].copy()
    t1 = passes[passes.period == 1].groupby("match_id").t_period_sec.max()
    net["tc"] = np.where(net.period == 1, net.t_period_sec,
                         net.match_id.map(t1) + net.t_period_sec)
    ni = {k: g for k, g in net.groupby(["match_id", "team_id"])}
    b = pd.read_csv(DATA / "substitution_batches.csv")
    b = b[b.period == 2].set_index("batch_id")
    sp = pd.read_parquet(DATA / "windows" / "window_specs.parquet")
    fl = pd.read_parquet(DATA / "windows" / "batch_flags.parquet")
    out = {W: [] for W in Ws}
    for W in Ws:
        Wm = int(W / 60)
        st = sp[(sp.kind == "time") & (sp.W_min == Wm)].set_index(["batch_id", "side"])
        nontr = set(fl[(fl.W_min == Wm) & ~fl.post_truncated].batch_id)
        for bid in nontr:
            if bid not in b.index or (bid, "pre") not in st.index:
                continue
            r = b.loc[bid]
            pre, post = st.loc[(bid, "pre")], st.loc[(bid, "post")]
            g = ni.get((r.match_id, r.team_id))
            if g is None:
                continue
            sw = set(int(x) for x in str(r.players_out_id).split("|")) \
                | set(int(x) for x in str(r.players_in_id).split("|"))
            wp = g[(g.tc >= pre.t_start) & (g.tc < pre.t_end)]
            wq = g[(g.tc > post.t_start) & (g.tc <= post.t_end)]
            mp = int((~wp.player_id.isin(sw) & ~wp.recipient_id.isin(sw)).sum())
            mq = int((~wq.player_id.isin(sw) & ~wq.recipient_id.isin(sw)).sum())
            out[W].append((mp, mq))
    return out, passes.match_id.nunique()


def p2_wsl_sampling(od, m33):
    print("\n" + "=" * 78)
    print("P2  WSL sampling: teammates-only completed passes per window m (V_perp ~ 1/m)")
    print("=" * 78)
    Ws = [600.0, 720.0, 900.0]
    data = {}
    data["WSL"], nm_wsl = m_counts_opendata(od, m33, 37, 90, Ws)
    data["LaLiga15_od"], nm_ll15 = m_counts_opendata(od, m33, 11, 27, Ws)
    data["LaLiga2324_main"], nm_main = m_counts_laliga_main(Ws)
    print(f"(matches: WSL {nm_wsl}, LaLiga15 {nm_ll15}, LaLiga2324 {nm_main}; "
          f"m = pre and post windows pooled, substituted players excluded, i.e. the estimator input)")
    med = {}
    print(f"\n{'dataset':>16} {'W(min)':>7} {'batches':>6} {'m med':>7} {'P25':>6} {'P75':>6} {'m/min':>7}")
    for name, dd in data.items():
        for W in Ws:
            arr = np.array([v for pair in dd[W] for v in pair], dtype=float)  # pre + post pooled
            nb = len(dd[W])
            md, p25, p75 = np.median(arr), np.percentile(arr, 25), np.percentile(arr, 75)
            med[(name, W)] = md
            print(f"{name:>16} {W/60:7.0f} {nb:6d} {md:7.0f} {p25:6.0f} {p75:6.0f} {md/(W/60):7.1f}")
            for met, val in [("n_batches", nb), ("m_median", md), ("m_p25", p25),
                             ("m_p75", p75), ("m_per_min_median", md / (W / 60))]:
                emit("P2_wsl_m", name, f"W{int(W/60)}", met, val)

    print("\nV_perp ~ 1/m -> WSL noise-floor lift vs reference at the same W, and equivalent window length:")
    for W in [720.0, 900.0]:
        for ref in ["LaLiga2324_main", "LaLiga15_od"]:
            lift = med[(ref, W)] / med[("WSL", W)]
            emit("P2_wsl_m", "WSL", f"W{int(W/60)}", f"noise_lift_vs_{ref}", lift)
            print(f"  W{int(W/60):>2} vs {ref:>16}: x{lift:.2f}", end="")
            # equivalent W: reference minutes giving the m that WSL collects in W minutes
            weq = med[("WSL", W)] / (med[(ref, W)] / (W / 60))
            emit("P2_wsl_m", "WSL", f"W{int(W/60)}", f"W_equiv_min_vs_{ref}", weq)
            print(f"  (equivalent {weq:.1f} min window)")
    lift_int = med[("WSL", 900.0)] / med[("WSL", 720.0)]
    emit("P2_wsl_m", "WSL", "W12", "noise_lift_W12_vs_W15_internal", lift_int)
    print(f"  WSL internal W15 -> W12: noise x{lift_int:.2f} (window shortening alone)")

    # WSL baseline signal margin: rank of r15 among 10 competitions, equal-sample N by W
    base = pd.read_csv(TAB / "cross_gradient_clean.csv")
    base["N"] = base.groupby("comp")["n"].transform("min")
    multi = pd.read_csv(TAB / "mvp_multiscale.csv")
    multi["N"] = multi.groupby(["config", "comp"])["n"].transform("min")
    print("\nWSL baseline signal margin (W15; smaller r15 = less signal power, SNR more sensitive to noise lift):")
    for tg in ["regular", "late"]:
        d = base[base.timing == tg].copy()
        d["r15"] = [invert_r(s, N) for s, N in zip(d.snr_excl, d.N)]
        d = d.sort_values("r15")
        rank = int(np.where(d.comp.values == "WSL")[0][0]) + 1
        r_wsl = float(d[d.comp == "WSL"].r15.iloc[0])
        print(f"  {TIMLAB[tg]:>5}: WSL r15 = {r_wsl:.4f}, ascending rank {rank}/{len(d)} "
              f"(median over competitions {d.r15.median():.4f})")
        emit("P2_wsl_m", "WSL", TIMLAB[tg], "r15_baseline", r_wsl)
        emit("P2_wsl_m", "WSL", TIMLAB[tg], "r15_rank_ascending_of10", rank)
        emit("P2_wsl_m", "WSL", TIMLAB[tg], "r15_median_all_comps", float(d.r15.median()))
    n15 = int(base[base.comp == "WSL"].N.iloc[0])
    n12 = int(multi[(multi.comp == "WSL") & (multi.config == "W12_6x4")].N.iloc[0])
    n10 = int(multi[(multi.comp == "WSL") & (multi.config == "W10_6x4")].N.iloc[0])
    print(f"  WSL equal-sample bootstrap N: W15={n15}, W12={n12}, W10={n10} "
          f"(larger N at short W: truncated batches re-enter, composition changes)")
    for w, nv in [("W15", n15), ("W12", n12), ("W10", n10)]:
        emit("P2_wsl_m", "WSL", w, "N_equal_sample", nv)
    return med


# P3
def _ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return beta, float(resid @ resid)


def fit_step(t, y):
    best = None
    cands = [(t[i] + t[i + 1]) / 2 for i in range(1, len(t) - 2)]  # >= 2 points each side
    for c in cands:
        X = np.column_stack([np.ones_like(t), (t >= c).astype(float)])
        beta, rss = _ols(X, y)
        if best is None or rss < best[1]:
            best = (c, rss, beta)
    c, rss, beta = best
    return dict(rss=rss, k=3, c=c, p=dict(a=beta[0], b=beta[1]))


def fit_hinge(t, y):
    best = None
    for c in np.arange(t.min() - 5, t.max() - 2.5 + 1e-9, 0.25):
        X = np.column_stack([np.ones_like(t), np.maximum(0.0, t - c)])
        beta, rss = _ols(X, y)
        if best is None or rss < best[1]:
            best = (c, rss, beta)
    c, rss, beta = best
    return dict(rss=rss, k=3, c=c, p=dict(a=beta[0], slope=beta[1]))


def fit_stepconv(t, y, width=WIDTH):
    """True step at c seen through a sliding window of the given width: linear crossover."""
    best = None
    for c in np.arange(t.min() - 2.5, t.max() + 2.5 + 1e-9, 0.25):
        ramp = np.clip((t - c) / width + 0.5, 0.0, 1.0)
        X = np.column_stack([np.ones_like(t), ramp])
        beta, rss = _ols(X, y)
        if best is None or rss < best[1]:
            best = (c, rss, beta)
    c, rss, beta = best
    return dict(rss=rss, k=3, c=c, p=dict(a=beta[0], b=beta[1]))


def fit_logistic(t, y):
    def model(p):
        a, b, c, w = p
        return a + b / (1 + np.exp(-(t - c) / w))

    best = None
    for c0 in [60.0, 65.0, 70.0]:
        for w0 in [1.0, 3.0, 8.0]:
            try:
                r = least_squares(lambda p: model(p) - y,
                                  x0=[y.min(), y.max() - y.min(), c0, w0],
                                  bounds=([0, -2, 45, 0.2], [3, 2, 90, 25]))
                rss = float(r.fun @ r.fun)
                if best is None or rss < best[0]:
                    best = (rss, r.x)
            except Exception:
                continue
    rss, x = best
    return dict(rss=rss, k=4, c=x[2], p=dict(a=x[0], b=x[1], w=x[3], rise_10_90=4.394 * x[3]))


def fit_const(t, y):
    beta, rss = _ols(np.ones((len(t), 1)), y)
    return dict(rss=rss, k=1, c=np.nan, p=dict(a=beta[0]))


MODELS = [("constant", fit_const), ("step", fit_step), ("linear_ramp", fit_hinge),
          ("step_conv_boxcar10", fit_stepconv), ("logistic", fit_logistic)]


def ic(rss, n, k):
    """k shape parameters, +1 noise variance; AICc small-sample correction."""
    K = k + 1
    aic = n * np.log(rss / n) + 2 * K
    aicc = aic + 2 * K * (K + 1) / (n - K - 1) if n - K - 1 > 0 else np.inf
    bic = n * np.log(rss / n) + K * np.log(n)
    return aic, aicc, bic


def fit_all(t, y, label, emit_rows=True):
    res = {}
    for name, fn in MODELS:
        f = fn(t, y)
        f["aic"], f["aicc"], f["bic"] = ic(f["rss"], len(t), f["k"])
        res[name] = f
    best_aicc = min(v["aicc"] for v in res.values())
    if emit_rows:
        print(f"\n--- change-point model comparison: {label} (n={len(t)} points, unweighted LS) ---")
        print(f"{'model':>20} {'k':>3} {'RSS':>8} {'AIC':>8} {'AICc':>8} {'BIC':>8} {'dAICc':>7} {'c (loc)':>8} params")
        for name, f in res.items():
            d = f["aicc"] - best_aicc
            pstr = ", ".join(f"{k}={v:.3f}" for k, v in f["p"].items())
            print(f"{name:>20} {f['k']:>3} {f['rss']:8.4f} {f['aic']:8.2f} {f['aicc']:8.2f} "
                  f"{f['bic']:8.2f} {d:7.2f} {f['c'] if np.isfinite(f['c']) else np.nan:8.2f}  {pstr}")
            for met, val in [("k", f["k"]), ("rss", f["rss"]), ("aic", f["aic"]),
                             ("aicc", f["aicc"]), ("bic", f["bic"]),
                             ("delta_aicc", d), ("c_hat", f["c"])]:
                emit("P3_changepoint", name, label, met, val)
            if name == "logistic":
                emit("P3_changepoint", name, label, "width_w", f["p"]["w"])
                emit("P3_changepoint", name, label, "rise_10_90_min", f["p"]["rise_10_90"])
    return res


def pl15_vectors(od, m33, cache_dir=None):
    """Rebuild PL15 (2_27) batch-level (minute, teammates-only DiD vector) as in the timing
    curve; rng seeded per competition with od.SEED, so it reproduces exactly."""
    if cache_dir:
        cf = Path(cache_dir) / "pl15_vectors.npz"
        if cf.exists():
            d = np.load(cf)
            return d["mins"], d["vecs"]
    cid, sid = 2, 27
    df = od.fetch(cid, sid)
    net, _ = od.prep(df)
    ni = {k: g.sort_values("tc") for k, g in net.groupby(["match_id", "team"])}
    keys = list(ni.keys())
    sub_times = {k: np.sort(g.minute.values * 60.0)
                 for k, g in df[df.type == "Substitution"].groupby(["match_id", "team"])}
    match_end = {m: g.minute.max() * 60 + g.second.max() for m, g in df.groupby("match_id")}
    B = m33.batches_named(df)
    rng = np.random.default_rng(od.SEED)
    W = 900.0

    def zvec_excl(w, names):
        return od.zvec(w[~w.player.isin(names) & ~w.pass_recipient.isin(names)])

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
    mins, vecs = np.array(mins), np.array(vecs)
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        np.savez(Path(cache_dir) / "pl15_vectors.npz", mins=mins, vecs=vecs)
    return mins, vecs


def snr_fast(arr, N, rng, nb=NB_IN, nn=NN_IN):
    idx = rng.integers(len(arr), size=(nb, N))
    mags = np.linalg.norm(arr[idx].mean(axis=1), axis=1)
    idx2 = rng.integers(len(arr), size=(nn, N))
    sgn = rng.choice([-1.0, 1.0], size=(nn, len(arr)))
    s = np.take_along_axis(sgn, idx2, axis=1)
    nf = np.linalg.norm((arr[idx2] * s[..., None]).mean(axis=1), axis=1)
    return mags.mean() / nf.mean()


def build_curve(mins, vecs, centers, rng, min_n=MIN_N_WIN):
    t_out, y_out = [], []
    for c in centers:
        m = (mins >= c - WIDTH / 2) & (mins < c + WIDTH / 2)
        if int(m.sum()) >= min_n:
            t_out.append(c)
            y_out.append(snr_fast(vecs[m], N_FIX, rng))
    return np.array(t_out), np.array(y_out)


def p3_changepoint(od, m33, nboot, skip_boot, cache_dir):
    print("\n" + "=" * 78)
    print("P3  threshold-shape change-point comparison (PL15; sliding points overlap 75%, descriptive only)")
    print("=" * 78)
    tc = pd.read_csv(TAB / "timing_curve.csv")
    pl = tc[tc.source == "PL15"].sort_values("center_min")
    t, y = pl.center_min.values.astype(float), pl.snr.values.astype(float)
    se_med = float(np.median((pl.hi - pl.lo) / 3.92))
    print(f"PL15 curve: {len(t)} points, centres {t.min():.1f}-{t.max():.1f}', SNR {y.min():.3f}-{y.max():.3f}; "
          f"median single-point bootstrap SE ~ {se_med:.3f} (curve range only {np.ptp(y):.3f})")
    emit("P3_changepoint", "curve", "PL15", "n_points", len(t))
    emit("P3_changepoint", "curve", "PL15", "point_se_median", se_med)
    res = fit_all(t, y, "PL15")

    # reference: La Liga main curve, same fits, no bootstrap
    ll = tc[tc.source == "La Liga 2023/24"].sort_values("center_min")
    fit_all(ll.center_min.values.astype(float), ll.snr.values.astype(float), "LaLiga2324")

    if skip_boot:
        print("\n(--skip-boot: batch-level bootstrap skipped)")
        return res
    print(f"\nbatch-level bootstrap: resample PL15 batches, rebuild curve x{nboot} (SEED {SEED_BOOT}; "
          f"inner SNR {NB_IN}/{NN_IN} resamples)")
    mins, vecs = pl15_vectors(od, m33, cache_dir)
    print(f"  PL15 batch vectors rebuilt: {len(mins)} batches")
    rng0 = np.random.default_rng(SEED_BOOT)
    t_chk, y_chk = build_curve(mins, vecs, t, rng0)
    dev = np.max(np.abs(y_chk - y)) if len(t_chk) == len(t) else np.nan
    print(f"  rebuilt curve vs timing_curve.csv, max deviation: {dev:.3f} "
          f"(different rng path, approximate; should be well below 2 x single-point SE {se_med:.3f})")
    emit("P3_changepoint", "curve", "PL15", "rebuild_max_dev", dev)

    rng = np.random.default_rng(SEED_BOOT)
    cs = {"step": [], "linear_ramp": [], "step_conv_boxcar10": [], "logistic": []}
    pref_step = 0
    n_ok = 0
    for it in range(nboot):
        ix = rng.integers(len(mins), size=len(mins))
        tb, yb = build_curve(mins[ix], vecs[ix], t, rng, min_n=50)
        if len(tb) < 8:
            continue
        n_ok += 1
        fits = {}
        for name, fn in [("step", fit_step), ("linear_ramp", fit_hinge),
                         ("step_conv_boxcar10", fit_stepconv), ("logistic", fit_logistic)]:
            f = fn(tb, yb)
            f["aicc"] = ic(f["rss"], len(tb), f["k"])[1]
            fits[name] = f
            cs[name].append(f["c"])
        if fits["step"]["aicc"] <= fits["logistic"]["aicc"]:
            pref_step += 1
    print(f"  valid reps {n_ok}/{nboot}; AICc prefers step over logistic: {pref_step}/{n_ok} "
          f"({100*pref_step/n_ok:.0f}%)")
    emit("P3_changepoint", "bootstrap", "PL15", "n_reps_ok", n_ok)
    emit("P3_changepoint", "bootstrap", "PL15", "pct_aicc_prefers_step_over_logistic",
         100 * pref_step / n_ok)
    print(f"  95% CI of change location c (batch bootstrap percentiles):")
    for name, arr in cs.items():
        a = np.array(arr)
        lo, mid, hi = np.percentile(a, [2.5, 50, 97.5])
        print(f"    {name:>20}: median {mid:5.1f}'  95% CI [{lo:5.1f}, {hi:5.1f}]  (span {hi-lo:.1f}')")
        for met, val in [("c_boot_median", mid), ("c_ci_lo", lo), ("c_ci_hi", hi)]:
            emit("P3_changepoint", name, "PL15_boot", met, val)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nboot", type=int, default=NBOOT_DEFAULT)
    ap.add_argument("--skip-boot", action="store_true")
    ap.add_argument("--cache", type=str, default=None,
                    help="optional cache dir for the PL15 batch vectors (speed only)")
    a = ap.parse_args()
    od, m33 = load_od_modules()
    p1_pred_vs_obs()
    p2_wsl_sampling(od, m33)
    p3_changepoint(od, m33, a.nboot, a.skip_boot, a.cache)
    out = pd.DataFrame(ROWS)
    out.to_csv(TAB / "wsl_collapse_changepoint.csv", index=False)
    print(f"\nwrote {TAB / 'wsl_collapse_changepoint.csv'} ({len(out)} rows)")


if __name__ == "__main__":
    main()
