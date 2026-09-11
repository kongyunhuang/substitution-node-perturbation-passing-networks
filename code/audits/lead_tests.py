"""Lead tests on the pre-pre window and pre-window spatial-state adjustments.

1: placebo DiD pre-pre -> pre (zone-flow SNR, d_terr). 2: treated vs control
pre-origin, ANCOVA adjustment (linear, spline), late-group tertiles.
3: negative control on pass volume. 4: subset without a prior substitution.
Inputs:  data/{events_pass,events_substitution,did_controls}.parquet, data/substitution_batches.csv, data/windows/*, results/tables/{control_robustness,dterr_audit}.csv
Outputs: results/tables/prepre_lead.csv, results/reports/lead_tests_report.md
Run: python code/audits/lead_tests.py
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from patsy import build_design_matrices
from scipy.stats import ttest_1samp

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
TAB = ROOT / "results" / "tables"
LOGDIR = ROOT / "results" / "reports"
LOGDIR.mkdir(parents=True, exist_ok=True)
NX, NY, PL, PW = 6, 4, 120, 80
SEED = 20260613
GROUPS = ["strategic", "regular", "late"]
W = 900.0  # 15 min in seconds


def zbin_vec(x, y):
    zx = np.clip((x / (PL / NX)).astype(int), 0, NX - 1)
    zy = np.clip((y / (PW / NY)).astype(int), 0, NY - 1)
    return zx * NY + zy


def zvec(w, ex_ids=None):
    """24-dim zone-flow vector, optional player exclusion."""
    if ex_ids is not None:
        w = w[~w.player_id.isin(ex_ids) & ~w.recipient_id.isin(ex_ids)]
    if len(w) == 0:
        return np.zeros(24)
    Z = np.zeros((24, 24))
    np.add.at(Z, (zbin_vec(w.x.values, w.y.values), zbin_vec(w.end_x.values, w.end_y.values)), 1.0)
    np.fill_diagonal(Z, 0)
    t = Z / Z.sum() if Z.sum() > 0 else Z
    return t.sum(0) + t.sum(1)


def terr(w, ex_ids=None):
    """Mean origin x of completed passes."""
    cw = w[w.outcome.isna()]
    if ex_ids is not None:
        cw = cw[~cw.player_id.isin(ex_ids) & ~cw.recipient_id.isin(ex_ids)]
    return cw.x.mean() if len(cw) >= 5 else np.nan


def make_win(index_dict):
    def win(mid, tid, lo, hi, lo_open):
        g = index_dict.get((mid, tid))
        if g is None:
            return None
        return g[(g.tc > lo) & (g.tc <= hi)] if lo_open else g[(g.tc >= lo) & (g.tc < hi)]
    return win


def main():
    # t1 from all passes, before dropping set pieces
    passes = pd.read_parquet(DATA / "events_pass.parquet")
    t1 = passes[passes.period == 1].groupby("match_id").t_period_sec.max()
    net = passes[passes.outcome.isna() & ~passes.is_set_piece].copy()  # zone-flow
    net["tc"] = np.where(net.period == 1, net.t_period_sec, net.match_id.map(t1) + net.t_period_sec)
    pall = passes[~passes.is_set_piece].copy()  # d_terr; terr() keeps completed passes
    pall["tc"] = np.where(pall.period == 1, pall.t_period_sec, pall.match_id.map(t1) + pall.t_period_sec)
    ni = {k: g for k, g in net.groupby(["match_id", "team_id"])}
    pi = {k: g for k, g in pall.groupby(["match_id", "team_id"])}
    winN, winT = make_win(ni), make_win(pi)

    b = pd.read_csv(DATA / "substitution_batches.csv").set_index("batch_id")
    sp = pd.read_parquet(DATA / "windows" / "window_specs.parquet")
    st = sp[(sp.kind == "time") & (sp.W_min == 15)].set_index(["batch_id", "side"])
    fl = pd.read_parquet(DATA / "windows" / "batch_flags.parquet")
    nontr = set(fl[(fl.W_min == 15) & ~fl.post_truncated].batch_id)
    ctrl = pd.read_parquet(DATA / "did_controls.parquet")
    ctrl = ctrl[ctrl.control_type != "none"].set_index("batch_id")

    # all sub events on the continuous clock, for the prior-sub flag
    se = pd.read_parquet(DATA / "events_substitution.parquet")
    se["t_period_sec"] = pd.to_timedelta(se.timestamp).dt.total_seconds()
    se["tc"] = np.where(se.period == 1, se.t_period_sec, se.match_id.map(t1) + se.t_period_sec)
    subt = {k: g.tc.values for k, g in se.groupby(["match_id", "team_id"])}

    # per batch: main windows + pre-pre windows
    recs = []
    for bid in b.index:
        if bid not in nontr or bid not in ctrl.index or (bid, "pre") not in st.index:
            continue
        r = b.loc[bid]
        pre, post = st.loc[(bid, "pre")], st.loc[(bid, "post")]
        c = ctrl.loc[bid]
        cm, ct_, tc = int(c.control_match_id), int(c.control_team_id), c.control_t_center
        swapped = ([int(x) for x in str(r.players_out_id).split("|")]
                   + [int(x) for x in str(r.players_in_id).split("|")])
        pp_lo, pp_hi = pre.t_start - W, pre.t_start          # treated pre-pre window
        cpp_lo, cpp_hi = tc - 2 * W, tc - W                  # control pre-pre window
        rec = dict(batch_id=bid, timing=r.timing_group,
                   clock_ok=bool(pp_lo >= 0 and cpp_lo >= 0),
                   prepre_spans_ht=bool(pp_lo < t1[r.match_id] < pp_hi))
        # prior sub: same team inside [pre.t_start-900, pre.t_end); own sub has tc >= pre.t_end
        stc = subt.get((r.match_id, r.team_id), np.array([]))
        rec["prior_sub_in_win"] = bool(((stc >= pp_lo) & (stc < pre.t_end)).any())

        # zone-flow main sample: completed passes, four windows >= 10
        wp, wq = winN(r.match_id, r.team_id, pre.t_start, pre.t_end, False), \
                 winN(r.match_id, r.team_id, post.t_start, post.t_end, True)
        wcp, wcq = winN(cm, ct_, tc - W, tc, False), winN(cm, ct_, tc, tc + W, True)
        rec["main_zone_ok"] = not any(w is None or len(w) < 10 for w in [wp, wq, wcp, wcq])
        if rec["main_zone_ok"]:
            rec["dz_main"] = (zvec(wq, swapped) - zvec(wp, swapped)) - (zvec(wcq) - zvec(wcp))
            wpp, wcpp = winN(r.match_id, r.team_id, pp_lo, pp_hi, False), winN(cm, ct_, cpp_lo, cpp_hi, False)
            rec["prepre_zone_npass_ok"] = not any(w is None or len(w) < 10 for w in [wpp, wcpp])
            if rec["clock_ok"] and rec["prepre_zone_npass_ok"]:
                rec["dz_placebo"] = (zvec(wp, swapped) - zvec(wpp, swapped)) - (zvec(wcp) - zvec(wcpp))
                rec["dvol_placebo"] = (len(wp) - len(wpp)) - (len(wcp) - len(wcpp))  # negative control: pass volume

        # d_terr main sample: all non-set-piece passes, four windows >= 10, terr not NaN
        vp, vq = winT(r.match_id, r.team_id, pre.t_start, pre.t_end, False), \
                 winT(r.match_id, r.team_id, post.t_start, post.t_end, True)
        vcp, vcq = winT(cm, ct_, tc - W, tc, False), winT(cm, ct_, tc, tc + W, True)
        if not any(w is None or len(w) < 10 for w in [vp, vq, vcp, vcq]):
            ctrl_d = terr(vcq) - terr(vcp)
            di = (terr(vq) - terr(vp)) - ctrl_d
            de = (terr(vq, swapped) - terr(vp, swapped)) - ctrl_d
            rec["main_terr_ok"] = not (np.isnan(di) or np.isnan(de))
            if rec["main_terr_ok"]:
                rec["did_terr_incl"], rec["did_terr_excl"] = di, de
                rec["pre_origin"] = terr(vp)          # treated pre-window origin, all players
                rec["ctrl_pre_origin"] = terr(vcp)    # matched control pre-window origin
                vpp, vcpp = winT(r.match_id, r.team_id, pp_lo, pp_hi, False), winT(cm, ct_, cpp_lo, cpp_hi, False)
                rec["prepre_terr_npass_ok"] = not any(w is None or len(w) < 10 for w in [vpp, vcpp])
                if rec["clock_ok"] and rec["prepre_terr_npass_ok"]:
                    cpl = terr(vcp) - terr(vcpp)
                    pi_ = (terr(vp) - terr(vpp)) - cpl
                    pe_ = (terr(vp, swapped) - terr(vpp, swapped)) - cpl
                    if not (np.isnan(pi_) or np.isnan(pe_)):
                        rec["pl_terr_incl"], rec["pl_terr_excl"] = pi_, pe_
        else:
            rec["main_terr_ok"] = False
        recs.append(rec)

    R = pd.DataFrame(recs)
    rows, md = [], []
    md.append("# Pre-pre placebo lead tests and pre-window spatial-state adjustment\n")
    md.append(f"Script: `code/audits/lead_tests.py` | seed: {SEED} | boot 800 / flip 400\n")
    md.append("Run: `python code/audits/lead_tests.py`\n")
    md.append("Motivation: the balance covariates are all topological and omit the pre-window spatial state; "
              "a team substituting while pinned back could rebound in the post window (regression to the mean) "
              "and mimic an upfield territorial effect. This file holds the lead tests and the spatial-state adjustment.\n")

    # replay check: main d_terr group means vs dterr_audit.csv
    Mt = R[R.main_terr_ok == True]
    ref = pd.read_csv(TAB / "dterr_audit.csv")
    chk_new = Mt.groupby("timing").did_terr_excl.mean().round(3)
    chk_ref = ref.groupby("timing").did_terr_excl.mean().round(3)
    repro_ok = bool((chk_new == chk_ref).all() and len(Mt) == len(ref))
    print(f"[replay check] main did_terr_excl group means match dterr_audit.csv: {repro_ok} "
          f"(n={len(Mt)} vs {len(ref)})\n  new: {chk_new.to_dict()}  ref: {chk_ref.to_dict()}")
    md.append(f"## 0. Replay check\n\nThe main d_terr DiD is rebuilt here; the three did_terr_excl group means match "
              f"`results/tables/dterr_audit.csv`: **{repro_ok}** (n={len(Mt)}). "
              f"strategic/regular/late = {chk_ref.get('strategic')}/{chk_ref.get('regular')}/{chk_ref.get('late')}.\n")

    # attrition
    md.append("## 1. Pre-pre window eligibility and attrition\n")
    md.append("Extra conditions: the treated pre-pre window [pre.t_start-900, pre.t_start) and the control pre-pre window [tc-1800, tc-900) both hold "
              ">=10 passes (completed passes for zone-flow, all non-set-piece passes for d_terr, as in the respective main analysis), and both start at >=0 "
              "on the continuous clock. Pre-pre windows may span halftime.\n")
    for ana, okc, valc in [("zone-flow", "main_zone_ok", "dz_placebo"),
                           ("d_terr", "main_terr_ok", "pl_terr_excl")]:
        md.append(f"\n### {ana}\n\n| group | entry n | clock-origin drop | pass-count/NaN drop | kept n | attrition |")
        md.append("|---|---|---|---|---|---|")
        E = R[R[okc] == True]
        for g in GROUPS:
            Eg = E[E.timing == g]
            n0 = len(Eg)
            n_clock = int((~Eg.clock_ok).sum())
            kept = Eg[valc].notna().sum()
            n_pass = n0 - n_clock - kept
            loss = (1 - kept / n0) * 100 if n0 else np.nan
            md.append(f"| {g} | {n0} | {n_clock} | {n_pass} | {kept} | {loss:.1f}% |")
            rows.append(dict(section="attrition", variant=ana, timing=g, n=kept,
                             est=n0, stat=n_clock, p=np.nan,
                             note=f"entry={n0},clock_fail={n_clock},npass_fail={n_pass},loss_pct={loss:.1f}"))
        ht = E[E[valc].notna()].prepre_spans_ht.mean() * 100 if E[valc].notna().any() else np.nan
        md.append(f"\nShare of kept pre-pre windows spanning halftime: {ht:.1f}%.")
    print("\n[attrition] see md; kept n:", {g: int(R[(R.main_zone_ok == True) & (R.timing == g)].dz_placebo.notna().sum()) for g in GROUPS})

    # 1(i) zone-flow placebo SNR
    md.append("\n## 2. Lead test (i): zone-flow placebo SNR (DiD of pre-pre -> pre)\n")
    md.append("Same estimator as the main analysis: teammates-only zone-flow vectors, DiD against the control displacement, common N = min group n, "
              "bootstrap 800 for the mean magnitude, sign-flip 400 for the null, p = P(null >= obs).\n")
    cr_ref = pd.read_csv(TAB / "control_robustness.csv")
    cr_all = cr_ref[cr_ref.subset == "all"].set_index("timing")
    rng = np.random.default_rng(SEED)

    def snr(arr, N, rg):
        mags = [np.linalg.norm(arr[rg.integers(len(arr), size=N)].mean(0)) for _ in range(800)]
        nf = []
        for _ in range(400):
            idx = rg.integers(len(arr), size=N)
            sgn = rg.choice([1, -1], size=len(arr))
            nf.append(np.linalg.norm((arr[idx] * sgn[idx, None]).mean(0)))
        obs = np.mean(mags)
        return obs / np.mean(nf), 1 - (obs > np.array(nf)).mean()

    Z = R[R.dz_placebo.notna()]
    by = {g: np.stack(Z[Z.timing == g].dz_placebo.values) if (Z.timing == g).any() else np.empty((0, 24))
          for g in GROUPS}
    ns = {g: len(by[g]) for g in GROUPS}
    N = min(n for n in ns.values() if n >= 10)
    md.append(f"Common N = {N} (group n = {ns}). Main SNR reference from control_robustness.csv (subset all).\n")
    md.append("| group | n | placebo SNR | placebo p | main SNR (pre -> post) | main p |")
    md.append("|---|---|---|---|---|---|")
    print(f"\n=== 1(i) zone-flow placebo SNR  common N={N}  group n={ns} ===")
    for g in GROUPS:
        s, p = snr(by[g], N, rng)
        mres = cr_all.loc[g]
        flag = "*" if p < .05 else "ns"
        print(f"  {g:10s}: placebo SNR={s:.2f} p={p:.3f} {flag}   (main SNR={mres.snr_excl} p={mres.p})")
        md.append(f"| {g} | {ns[g]} | {s:.3f} | {p:.4f} | {mres.snr_excl} | {mres.p} |")
        rows.append(dict(section="placebo_zone_snr", variant=f"N={N}", timing=g, n=ns[g],
                         est=round(s, 3), stat=np.nan, p=round(p, 4),
                         note=f"main_snr={mres.snr_excl},main_p={mres.p}"))

    # 1(ii) d_terr placebo DiD
    md.append("\n## 3. Lead test (ii): territorial placebo DiD (d_terr, pre-pre -> pre)\n")
    md.append("| version | group | n | placebo mean | t | p | main effect (pre -> post) |")
    md.append("|---|---|---|---|---|---|---|")
    print("\n=== 1(ii) d_terr placebo DiD (pre-pre -> pre) ===")
    main_means = Mt.groupby("timing")[["did_terr_incl", "did_terr_excl"]].mean()
    for col, lab in [("pl_terr_excl", "excl (perturber excluded)"), ("pl_terr_incl", "incl (perturber included)")]:
        mcol = "did_terr_excl" if "excl" in col else "did_terr_incl"
        for g in GROUPS:
            x = R[(R.timing == g)][col].dropna()
            t, p = ttest_1samp(x, 0)
            mm = main_means.loc[g, mcol]
            flag = "*" if p < .05 else "ns"
            print(f"  {lab:12s} {g:10s}: n={len(x):3d} mean {x.mean():+.3f} t={t:+.2f} p={p:.3f} {flag} (main={mm:+.3f})")
            md.append(f"| {lab} | {g} | {len(x)} | {x.mean():+.3f} | {t:+.2f} | {p:.4f} | {mm:+.3f} |")
            rows.append(dict(section="placebo_dterr", variant=lab, timing=g, n=len(x),
                             est=round(x.mean(), 3), stat=round(t, 2), p=round(p, 4),
                             note=f"main={mm:+.3f}"))

    # 2(a) selection strength: pre-window origin, treated vs control
    md.append("\n## 4. Pre-window spatial state (2a): treated vs matched control (selection strength)\n")
    md.append("Mean origin x of the treated team's completed passes in the pre window (all players) minus the same for the matched control; paired t-test. "
              "Negative = the substituting team sat deeper (pinned back) in the pre window, the selection channel for regression to the mean.\n")
    md.append("| group | n | treated mean x | control mean x | diff | t | p |")
    md.append("|---|---|---|---|---|---|---|")
    print("\n=== 2(a) pre-origin: treated vs matched control ===")
    A = Mt[Mt.pre_origin.notna() & Mt.ctrl_pre_origin.notna()]
    for g in GROUPS:
        Ag = A[A.timing == g]
        d = Ag.pre_origin - Ag.ctrl_pre_origin
        t, p = ttest_1samp(d, 0)
        flag = "*" if p < .05 else "ns"
        print(f"  {g:10s}: n={len(Ag):3d} treated={Ag.pre_origin.mean():.2f} ctrl={Ag.ctrl_pre_origin.mean():.2f} "
              f"diff={d.mean():+.2f} t={t:+.2f} p={p:.4f} {flag}")
        md.append(f"| {g} | {len(Ag)} | {Ag.pre_origin.mean():.2f} | {Ag.ctrl_pre_origin.mean():.2f} | "
                  f"{d.mean():+.2f} | {t:+.2f} | {p:.4f} |")
        rows.append(dict(section="selection_pre_origin", variant="treated-ctrl", timing=g, n=len(Ag),
                         est=round(d.mean(), 3), stat=round(t, 2), p=round(p, 4),
                         note=f"treated={Ag.pre_origin.mean():.2f},ctrl={Ag.ctrl_pre_origin.mean():.2f}"))

    # 2(b) regression adjustment
    md.append("\n## 5. Regression adjustment of the territorial effect for pre-origin (2b)\n")
    md.append("ANCOVA: did_terr_excl ~ C(timing) + f(pre_origin), f linear or natural cubic spline cr(df=4); "
              "adjusted group effect = model prediction per group at the overall mean pre_origin (t-test vs 0). Sample = main d_terr sample.\n")
    md.append("| version | group | n | territorial effect | SE | p |")
    md.append("|---|---|---|---|---|---|")
    print("\n=== 2(b) adjusted group effects of d_terr_excl ===")
    D = Mt[Mt.pre_origin.notna()][["did_terr_excl", "timing", "pre_origin"]].rename(
        columns={"did_terr_excl": "did", "pre_origin": "x"})
    xbar = D.x.mean()
    versions = [("unadjusted", "did ~ 0 + C(timing)"),
                ("linear", "did ~ C(timing) + x"),
                ("spline_cr4", "did ~ C(timing) + cr(x, df=4)")]
    for vname, f in versions:
        res = smf.ols(f, data=D).fit()
        for g in GROUPS:
            row = np.asarray(build_design_matrices(
                [res.model.data.design_info], pd.DataFrame({"timing": [g], "x": [xbar]}))[0])
            tt = res.t_test(row)
            e, se, p = float(tt.effect), float(tt.sd), float(tt.pvalue)
            ng = int((D.timing == g).sum())
            flag = "*" if p < .05 else "ns"
            print(f"  {vname:10s} {g:10s}: {e:+.3f} (SE {se:.3f}) p={p:.2e} {flag}")
            md.append(f"| {vname} | {g} | {ng} | {e:+.3f} | {se:.3f} | {p:.2e} |")
            rows.append(dict(section="dterr_adjusted", variant=vname, timing=g, n=ng,
                             est=round(e, 3), stat=round(se, 3), p=p, note=f"at_xbar={xbar:.2f}"))

    # 2(c) late group by pre-origin tertile
    md.append("\n## 6. Late group by pre-origin tertile (2c)\n")
    md.append("Regression-to-the-mean fingerprint: a much larger effect in the lowest tertile (T1, most pinned back) decreasing monotonically across tertiles.\n")
    md.append("| tertile | n | mean pre-origin | mean did_terr_excl | t | p |")
    md.append("|---|---|---|---|---|---|")
    print("\n=== 2(c) late group by pre-origin tertile ===")
    L = Mt[(Mt.timing == "late") & Mt.pre_origin.notna()].copy()
    L["tert"] = pd.qcut(L.pre_origin, 3, labels=["T1_low", "T2_mid", "T3_high"])
    tert_means = []
    for lv in ["T1_low", "T2_mid", "T3_high"]:
        Lg = L[L.tert == lv]
        t, p = ttest_1samp(Lg.did_terr_excl, 0)
        tert_means.append(Lg.did_terr_excl.mean())
        flag = "*" if p < .05 else "ns"
        print(f"  {lv:8s}: n={len(Lg):2d} mean_x={Lg.pre_origin.mean():.2f} did={Lg.did_terr_excl.mean():+.3f} "
              f"t={t:+.2f} p={p:.3f} {flag}")
        md.append(f"| {lv} | {len(Lg)} | {Lg.pre_origin.mean():.2f} | {Lg.did_terr_excl.mean():+.3f} | "
                  f"{t:+.2f} | {p:.4f} |")
        rows.append(dict(section="late_tertile", variant=lv, timing="late", n=len(Lg),
                         est=round(Lg.did_terr_excl.mean(), 3), stat=round(t, 2), p=round(p, 4),
                         note=f"mean_pre_origin={Lg.pre_origin.mean():.2f}"))
    mono_desc = tert_means[0] > tert_means[1] > tert_means[2]

    # 3 negative-control outcome
    md.append("\n## 7. Negative-control outcome: placebo DiD of completed-pass volume (pre-pre -> pre)\n")
    md.append("A non-spatial quantity; zero means no systematic activity pre-trend in the pre-pre window.\n")
    md.append("| group | n | mean (passes) | t | p |")
    md.append("|---|---|---|---|---|")
    print("\n=== 3 negative control: completed-pass volume placebo DiD ===")
    for g in GROUPS:
        x = R[R.timing == g].dvol_placebo.dropna()
        t, p = ttest_1samp(x, 0)
        flag = "*" if p < .05 else "ns"
        print(f"  {g:10s}: n={len(x):3d} mean {x.mean():+.2f} t={t:+.2f} p={p:.3f} {flag}")
        md.append(f"| {g} | {len(x)} | {x.mean():+.2f} | {t:+.2f} | {p:.4f} |")
        rows.append(dict(section="negctrl_volume", variant="completed_passes", timing=g, n=len(x),
                         est=round(x.mean(), 2), stat=round(t, 2), p=round(p, 4), note=""))

    # supplement: carryover from a prior substitution
    md.append("\n## 8. Supplement: prior substitution inside the window (carryover) as an alternative explanation of the pre-drift\n")
    md.append("Alternative: the pre/pre-pre windows of mid/late batches often contain an earlier substitution of the same team, so a significant placebo may be "
              "the carryover response to the previous substitution rather than anticipation or selection. Flag: any earlier substitution of the same team "
              "(events_substitution.parquet, first-half subs included) inside pre-pre U pre = [pre.t_start-900, pre.t_end)"
              " (= [t_end_pre-1800, t_end_pre)); the batch's own substitution has tc >= pre.t_end and is excluded by the half-open interval.\n")
    Zp = R[R.dz_placebo.notna()]
    md.append("| group | placebo sample n | prior sub in window | share |")
    md.append("|---|---|---|---|")
    print("\n=== supplement: share of placebo batches with a prior sub inside pre-pre U pre ===")
    for g in GROUPS:
        Zg = Zp[Zp.timing == g]
        k = int(Zg.prior_sub_in_win.sum())
        share = k / len(Zg) * 100
        print(f"  {g:10s}: {k}/{len(Zg)} = {share:.1f}%")
        md.append(f"| {g} | {len(Zg)} | {k} | {share:.1f}% |")
        rows.append(dict(section="prior_sub_share", variant="prepre_union_pre", timing=g, n=len(Zg),
                         est=round(share, 1), stat=k, p=np.nan,
                         note="share_pct; window=[pre.t_start-900,pre.t_end)"))

    S = Zp[~Zp.prior_sub_in_win]
    ns2 = {g: int((S.timing == g).sum()) for g in GROUPS}
    elig = [g for g in GROUPS if ns2[g] >= 20]
    N2 = min(ns2[g] for g in elig)
    md.append(f"\nSubset without a prior substitution: group n = {ns2}; SNR only for groups with n>=20 (common N = {N2}, "
              f"boot 800/flip 400, seed {SEED} re-initialised, placebo round first then main SNR round, group order "
              "strategic -> regular -> late); groups with n<20 report only the descriptive mean magnitude. Placebo and main SNR use **the same batch set**, "
              "so the two columns differ only in the window (pre-pre -> pre vs pre -> post).\n")
    md.append("| group | n | placebo SNR | placebo p | main SNR (same subset) | main p | descriptive norm(mean dz) placebo/main |")
    md.append("|---|---|---|---|---|---|---|")
    print(f"\n=== supplement: SNR on the subset without a prior sub  common N={N2}  group n={ns2} ===")
    rng2 = np.random.default_rng(SEED)
    res2 = {}
    for col, key in [("dz_placebo", "placebo"), ("dz_main", "main")]:
        for g in GROUPS:
            arr = np.stack(S[S.timing == g][col].values)
            desc = np.linalg.norm(arr.mean(0))
            if g in elig:
                s, p = snr(arr, N2, rng2)
            else:
                s, p = np.nan, np.nan
            res2[(key, g)] = (s, p, desc)
            rows.append(dict(section="zone_snr_noprior", variant=key, timing=g, n=ns2[g],
                             est=(round(s, 3) if np.isfinite(s) else np.nan),
                             stat=round(desc, 4), p=(round(p, 4) if np.isfinite(p) else np.nan),
                             note=f"N_common={N2}" if g in elig else "n<20: descriptive_only(stat=||mean dz||)"))
    for g in GROUPS:
        sp_, pp_, dp_ = res2[("placebo", g)]
        sm_, pm_, dm_ = res2[("main", g)]
        if g in elig:
            print(f"  {g:10s}: placebo SNR={sp_:.2f} p={pp_:.3f} {'*' if pp_ < .05 else 'ns'} | "
                  f"main SNR={sm_:.2f} p={pm_:.3f} {'*' if pm_ < .05 else 'ns'}")
            md.append(f"| {g} | {ns2[g]} | {sp_:.3f} | {pp_:.4f} | {sm_:.3f} | {pm_:.4f} | "
                      f"{dp_:.4f} / {dm_:.4f} |")
        else:
            print(f"  {g:10s}: n={ns2[g]}<20, descriptive ||mean dz|| only: placebo={dp_:.4f} main={dm_:.4f}")
            md.append(f"| {g} | {ns2[g]} | n<20 | n/a | n<20 | n/a | {dp_:.4f} / {dm_:.4f} |")

    # verdict
    def sig_groups(section, variant_key=None):
        return [r["timing"] for r in rows if r["section"] == section
                and (variant_key is None or variant_key in r["variant"]) and r["p"] < .05]

    zone_fail = sig_groups("placebo_zone_snr")
    terr_fail = sig_groups("placebo_dterr", "excl")
    neg_fail = sig_groups("negctrl_volume")
    pl_zone_pass, pl_terr_pass = not zone_fail, not terr_fail
    adj_rows = [r for r in rows if r["section"] == "dterr_adjusted"]
    surv = {v: [g for g in GROUPS for r in adj_rows
                if r["variant"] == v and r["timing"] == g and r["p"] < .05 and r["est"] > 0]
            for v in ["unadjusted", "linear", "spline_cr4"]}
    late_terr_p = [r["p"] for r in rows if r["section"] == "placebo_dterr"
                   and "excl" in r["variant"] and r["timing"] == "late"][0]
    md.append("\n## 9. Verdict\n")
    md.append(f"- **Lead test (zone-flow placebo SNR)**: {'pass' if pl_zone_pass else f'not passed in all groups; significant = {zone_fail}'}. "
              "A significant placebo window means part of that group's zone-flow reorganization was already drifting before the substitution "
              "(anticipation or selection dynamics), so the main SNR cannot be attributed entirely to the substitution.")
    md.append(f"- **Lead test (d_terr placebo, excl)**: {'pass' if pl_terr_pass else f'not passed in all groups; significant = {terr_fail}'}. "
              f"Lead-test p for the headline late group = {late_terr_p:.3f}"
              f"{' (pass: no significant pre-trend before +3.79)' if late_terr_p >= .05 else ' (not passed)'}.")
    md.append(f"- **Territorial effect surviving adjustment**: significant unadjusted = {surv['unadjusted']}; after linear adjustment = {surv['linear']}; "
              f"after spline (cr df=4) adjustment = {surv['spline_cr4']}.")
    md.append(f"- **Monotone decrease across late tertiles (regression-to-the-mean fingerprint)**: {'yes, disclose in the text' if mono_desc else 'no, T1>T2>T3 does not hold'}"
              f" (T1/T2/T3 = {tert_means[0]:+.2f}/{tert_means[1]:+.2f}/{tert_means[2]:+.2f}); "
              "heterogeneity across strata is compatible with both regression to the mean and a ceiling effect (upper bound of x); adjusted effects at the mean are in the table above.")
    md.append(f"- **Negative control (completed-pass volume placebo)**: {'not significant in any group' if not neg_fail else f'significant = {neg_fail}, activity pre-trend in that group'}.")
    # carryover verdict: does a full-sample significant placebo vanish in the no-prior-sub subset
    co_lines = []
    for g in zone_fail:
        if g in elig:
            sp_, pp_, _ = res2[("placebo", g)]
            verdict = ("vanishes -> pre-drift attributable to carryover from a prior substitution (compatible with the threshold narrative)" if pp_ >= .05
                       else "persists -> genuine anticipation/selection; soften the causal wording in the text")
            co_lines.append(f"{g}: subset placebo SNR={sp_:.2f} p={pp_:.3f}, {verdict}")
        else:
            _, _, dp_ = res2[("placebo", g)]
            _, _, dm_ = res2[("main", g)]
            co_lines.append(f"{g}: no-prior-sub subset n={ns2[g]}<20, inconclusive (descriptive norm(mean dz) "
                            f"placebo={dp_:.4f} vs main={dm_:.4f})")
    md.append(f"- **Carryover check (prior substitution)**: subset re-test of groups with a significant full-sample placebo; " + "; ".join(co_lines) + ".")
    md.append("")

    pd.DataFrame(rows).to_csv(TAB / "prepre_lead.csv", index=False)
    (LOGDIR / "lead_tests_report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n[output] {TAB / 'prepre_lead.csv'}")
    print(f"[output] {LOGDIR / 'lead_tests_report.md'}")
    print(f"\nverdict: lead(zone)={'PASS' if pl_zone_pass else 'FAIL'}  lead(terr)={'PASS' if pl_terr_pass else 'FAIL'}  "
          f"surviving adjustment: linear={surv['linear']} spline={surv['spline_cr4']}  late tertiles monotone={mono_desc}")
    for ln in co_lines:
        print("  carryover re-test:", ln)


if __name__ == "__main__":
    main()
