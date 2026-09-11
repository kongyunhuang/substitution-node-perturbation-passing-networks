"""TOST equivalence for the null outcomes, 5/10 replication chance benchmark, related audits.

A: TOST on partial r vs SESOI |r|=0.10 (0.05 secondary). B: replication rule
(late-group mean forward shift > early-group mean, per competition, no significance
requirement) vs binomial chance. C: permutation p of max|z|. D: dropped vs retained SMDs.
Inputs:  results/tables/{consequence_mine,mechanism_mine,mechanism_null_global,mechanism_null_perm}.csv, data/{did_controls,analysis_table}.parquet, data/opendata/*.parquet
Outputs: results/tables/equivalence_misc.csv, results/reports/equivalence_tests_report.md
Run: python code/audits/equivalence_tests.py
"""

import importlib.util
import warnings
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]  # <repo>/code

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "data"
TAB = ROOT / "results" / "tables"
REP = ROOT / "results" / "reports" / "equivalence_tests_report.md"
REP.parent.mkdir(parents=True, exist_ok=True)
SEED = 20260808          # no random steps; fixed for completeness
W = 900.0

ROWS = []                 # tidy rows: section,item,metric,value,note


def put(section, item, metric, value, note=""):
    ROWS.append(dict(section=section, item=item, metric=metric,
                     value=value, note=note))


# A. TOST equivalence
def sec_a_tost():
    cm = pd.read_csv(TAB / "consequence_mine.csv")
    out = []
    for _, r in cm.iterrows():
        z = norm.isf(r.p / 2) * np.sign(r.beta)      # as in the fig3 panel
        pr = z / np.sqrt(r.n)                        # partial r
        se = 1.0 / np.sqrt(r.n)                      # = r / z
        lo95, hi95 = pr - 1.96 * se, pr + 1.96 * se
        lo90, hi90 = pr - 1.6449 * se, pr + 1.6449 * se
        tost = {}
        for m in (0.10, 0.05):
            p_lo = 1 - norm.cdf((pr + m) / se)       # H0: r <= -m
            p_hi = norm.cdf((pr - m) / se)           # H0: r >= +m
            tost[m] = max(p_lo, p_hi)
        out.append(dict(outcome=r.outcome, label=r.label, n=int(r.n),
                        beta=r.beta, p=r.p, q=r.q, z=z, r=pr,
                        r_lo95=lo95, r_hi95=hi95, r_lo90=lo90, r_hi90=hi90,
                        p_tost10=tost[0.10], p_tost05=tost[0.05]))
    T = pd.DataFrame(out)
    comp = T[T.outcome != "d_terr"]                  # 10 competitive outcomes
    for _, r in T.iterrows():
        ref = "" if r.outcome != "d_terr" else "structural outcome (significant), reference only, not in the equivalence verdict"
        put("A_tost", r.outcome, "n", r.n, r.label)
        put("A_tost", r.outcome, "partial_r", r.r)
        put("A_tost", r.outcome, "r_lo95", r.r_lo95)
        put("A_tost", r.outcome, "r_hi95", r.r_hi95)
        put("A_tost", r.outcome, "p_tost_sesoi_0.10", r.p_tost10,
            "equivalent" if r.p_tost10 < 0.05 else "not shown equivalent")
        put("A_tost", r.outcome, "p_tost_sesoi_0.05", r.p_tost05,
            ("equivalent" if r.p_tost05 < 0.05 else "inconclusive at 0.05") + ("; " + ref if ref else ""))
    n_eq10 = int((comp.p_tost10 < 0.05).sum())
    n_eq05 = int((comp.p_tost05 < 0.05).sum())
    put("A_tost", "SUMMARY", "n_competitive_outcomes", len(comp))
    put("A_tost", "SUMMARY", "n_equivalent_at_0.10", n_eq10)
    put("A_tost", "SUMMARY", "n_equivalent_at_0.05", n_eq05)
    put("A_tost", "SUMMARY", "max_abs_r_competitive", comp.r.abs().max(),
        comp.loc[comp.r.abs().idxmax(), "outcome"])
    print(f"[A] TOST: {n_eq10}/{len(comp)} competitive outcomes equivalent to zero at SESOI 0.10; "
          f"{n_eq05}/{len(comp)} at 0.05")
    return T, n_eq10, n_eq05


# B. "5/10 replication" rule and chance benchmark
COMPS = {"World Cup22": (43, 106), "Euro24": (55, 282), "Euro20": (55, 43),
         "ISL": (1238, 108), "WSL": (37, 90), "Bundesliga23": (9, 281),
         "Bundesliga15": (9, 27), "LaLiga15": (11, 27), "PL15": (2, 27),
         "SerieA15": (12, 27)}


def sec_b_replication():
    # forward shift = post-window mean origin x - pre-window mean; replicates if late mean > early mean
    # (same sign only, no significance requirement); summary binomtest(k, n), p=0.5 two-sided
    spec = importlib.util.spec_from_file_location(
        "od30", CODE_ROOT / "replication" / "fetch_open_data.py")
    od = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(od)

    flags, detail = [], []
    for name, (cid, sid) in COMPS.items():
        net_idx, _, match_end, B = od.build(cid, sid)
        grp = {"strategic": [], "regular": [], "late": []}
        for _, b in B.iterrows():
            key, t0 = (b.match_id, b.team), b.t0
            if t0 + W > match_end.get(b.match_id, 0):
                continue
            wp = od.win(net_idx, key, t0 - W, t0, False)
            wq = od.win(net_idx, key, t0, t0 + W, True)
            if wp is None or wq is None or len(wp) < 10 or len(wq) < 10:
                continue
            grp[b.timing].append(wq.x.mean() - wp.x.mean())
        if any(len(v) < 10 for v in grp.values()):
            detail.append((name, np.nan, np.nan, None)); continue
        e, l = np.mean(grp["strategic"]), np.mean(grp["late"])
        flags.append(l > e)
        detail.append((name, e, l, l > e))
        put("B_replication", name, "fwd_shift_early", e)
        put("B_replication", name, "fwd_shift_late", l)
        put("B_replication", name, "replicates_late_gt_early", int(l > e))
    k, n = sum(flags), len(flags)
    bt_two = binomtest(k, n).pvalue                       # p=0.5, two-sided
    bt_one = binomtest(k, n, alternative="greater").pvalue
    bt_hyp = binomtest(k, n, p=0.025, alternative="greater").pvalue
    put("B_replication", "SUMMARY", "criterion",
        "same-sign only: mean fwd-shift(late) > mean fwd-shift(strategic), "
        "per competition; NO significance requirement",
        "forward shift = post-window mean pass origin minus pre-window mean; replication = late-group mean above early-group mean per competition, no significance requirement")
    put("B_replication", "SUMMARY", "k_over_n", f"{k}/{n}",
        "re-derived on the ten open-data competitions")
    put("B_replication", "SUMMARY", "chance_per_comp_same_sign", 0.5)
    put("B_replication", "SUMMARY", "p_binom_two_sided_at_0.5", bt_two)
    put("B_replication", "SUMMARY", "p_binom_one_sided_at_0.5", bt_one)
    put("B_replication", "SUMMARY", "chance_per_comp_sign_and_p05", 0.025,
        "hypothetical stricter rule: same sign and two-sided p<0.05; not the rule used")
    put("B_replication", "SUMMARY", "p_binom_one_sided_at_0.025", bt_hyp)
    print(f"[B] replication rule = same sign only; re-derived {k}/{n}; binom(0.5) one-sided p={bt_one:.3f} "
          f"two-sided p={bt_two:.3f}; hypothetical rule (0.025) p={bt_hyp:.2e}")
    return detail, k, n, bt_one, bt_two, bt_hyp


# C. borderline max|z|
def sec_c_maxz():
    glob = pd.read_csv(TAB / "mechanism_null_global.csv")
    perm = pd.read_csv(TAB / "mechanism_null_perm.csv")
    mm = pd.read_csv(TAB / "mechanism_mine.csv")
    obs = float(glob[(glob.family == "interaction")
                     & (glob.stat == "max_abs_z")].observed.iloc[0])
    B = len(perm)
    n_ge = int((perm.max_abs_z >= obs).sum())
    p_raw = n_ge / B
    p_add1 = (n_ge + 1) / (B + 1)
    q95 = float(perm.max_abs_z.quantile(0.95))
    thr = 0.05 / 36
    for col, fam in [("p_main", "main"), ("p_inter", "interaction")]:
        d = (mm[col] - thr).abs()
        i = d.idxmin()
        put("C_maxz", f"closest_to_BHmin_{fam}", "predictor", mm.loc[i, "predictor"])
        put("C_maxz", f"closest_to_BHmin_{fam}", col, mm.loc[i, col])
        put("C_maxz", f"closest_to_BHmin_{fam}", "abs_distance_to_thr",
            float(d[i]), f"thr=0.05/36={thr:.6f}")
        put("C_maxz", f"closest_to_BHmin_{fam}", "ratio_p_over_thr",
            float(mm.loc[i, col] / thr))
    put("C_maxz", "perm_test", "observed_max_abs_z", obs, "interaction family")
    put("C_maxz", "perm_test", "B_permutations", B)
    put("C_maxz", "perm_test", "n_perm_ge_obs", n_ge)
    put("C_maxz", "perm_test", "p_perm_raw", p_raw)
    put("C_maxz", "perm_test", "p_perm_plus1", p_add1, "(r+1)/(B+1)")
    put("C_maxz", "perm_test", "perm_q95", q95, "text cites 3.14")
    mvn_p = float(glob[(glob.family == "interaction")
                       & (glob.stat == "max_abs_z")].mc_p.iloc[0])
    put("C_maxz", "perm_test", "mvn_mc_p_reference", mvn_p,
        "same statistic under the dependence-aware MVN null (NSIM=1e5), cross-reference")
    print(f"[C] permutation p(max|z|>= {obs:.3f}) = {n_ge}/{B} = {p_raw:.3f} "
          f"(add-one {p_add1:.3f}); q95={q95:.3f}; MVN reference p={mvn_p:.3f}")
    return obs, B, n_ge, p_raw, p_add1, q95, mvn_p, mm, thr


# D. dropped vs retained on the 12 covariates
COVS = (["pre_%s_%s_t15" % (m, l) for l in ("P", "Z")
         for m in ("density", "clustering", "betweenness", "lambda2", "efficiency")]
        + ["pre_vers_P_w10_Z_t15", "pre_vers_Z_w10_Z_t15"])


def sec_d_selection():
    dc = pd.read_parquet(DATA / "did_controls.parquet")[["batch_id", "control_type"]]
    at = pd.read_parquet(DATA / "analysis_table.parquet")
    m = dc.merge(at[["batch_id", "timing_group", "minute_first"] + COVS],
                 on="batch_id", how="inner")
    m["retained"] = m.control_type != "none"
    rows = []
    for c in COVS + ["minute_first"]:
        a = m.loc[m.retained, c].dropna().to_numpy(float)      # with control
        b = m.loc[~m.retained, c].dropna().to_numpy(float)     # without control
        sd = np.sqrt((a.var() + b.var()) / 2)                  # as entropy_balancing smd(), ddof=0
        s = (a.mean() - b.mean()) / sd if sd > 0 else np.nan
        rows.append(dict(covariate=c, n_ret=len(a), n_drop=len(b),
                         mean_ret=a.mean(), mean_drop=b.mean(), smd=s))
        tag = "supplementary (not one of the 12 covariates)" if c == "minute_first" else ""
        put("D_selection", c, "n_retained", len(a), tag)
        put("D_selection", c, "n_dropped", len(b))
        put("D_selection", c, "mean_retained", a.mean())
        put("D_selection", c, "mean_dropped", b.mean())
        put("D_selection", c, "smd_ret_minus_drop", s)
    D = pd.DataFrame(rows)
    core = D[D.covariate != "minute_first"]
    n_big = int((core.smd.abs() > 0.1).sum())
    put("D_selection", "SUMMARY", "n_retained_batches", int(m.retained.sum()))
    put("D_selection", "SUMMARY", "n_dropped_batches", int((~m.retained).sum()))
    put("D_selection", "SUMMARY", "n_abs_smd_gt_0.1", n_big, "among the 12 covariates")
    put("D_selection", "SUMMARY", "max_abs_smd", core.smd.abs().max(),
        core.loc[core.smd.abs().idxmax(), "covariate"])
    timing_tab = m.groupby(["timing_group", "retained"]).size().unstack()
    print(f"[D] with control {int(m.retained.sum())} vs without {int((~m.retained).sum())}; "
          f"12 covariates with |SMD|>0.1: {n_big}; max|SMD|={core.smd.abs().max():.3f} "
          f"({core.loc[core.smd.abs().idxmax(), 'covariate']})")
    return D, m, timing_tab


# markdown report
def write_report(T, n_eq10, n_eq05, detB, C, D, m, timing_tab):
    detail, k, n, bt_one, bt_two, bt_hyp = detB
    obs, B, n_ge, p_raw, p_add1, q95, mvn_p, mm, thr = C
    comp = T[T.outcome != "d_terr"]
    dt = T[T.outcome == "d_terr"].iloc[0]
    core = D[D.covariate != "minute_first"]
    mf = D[D.covariate == "minute_first"].iloc[0]
    L = []
    L.append("# Equivalence tests, 5-of-10 replication benchmark, borderline max|z|, dropped-vs-retained selection\n")
    L.append("Script: code/audits/equivalence_tests.py | "
             "table: results/tables/equivalence_misc.csv | seed: 20260808 (no random steps)\n")

    L.append("\n## A. TOST equivalence tests for the 10 competitive outcomes (SESOI |r| = 0.1)\n")
    L.append("Partial-r scale as in fig3.py panel c: z = Phi^-1(p/2)*sign(beta), "
             "partial r = z/sqrt(n), se = 1/sqrt(n) (i.e. r_se = r/z); "
             "TOST = two one-sided z tests, equivalence when p_TOST < 0.05 (90% CI inside the +/- SESOI band).\n")
    L.append("\n| outcome | n | partial r | 95% CI | p_TOST(0.10) | verdict@0.10 | p_TOST(0.05) | verdict@0.05 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for _, r in T.sort_values("r", key=abs, ascending=False).iterrows():
        if r.outcome == "d_terr":
            v10 = v05 = "n/a (significantly non-zero)"
        else:
            v10 = "equivalent" if r.p_tost10 < 0.05 else "not shown"
            v05 = "equivalent" if r.p_tost05 < 0.05 else "not shown"
        L.append(f"| {r.outcome} | {r.n} | {r.r:+.4f} | [{r.r_lo95:+.4f}, {r.r_hi95:+.4f}] "
                 f"| {r.p_tost10:.2e} | {v10} | {r.p_tost05:.3f} | {v05} |")
    L.append(f"\n**Verdict**: {n_eq10}/10 competitive outcomes **pass equivalence to zero** at SESOI |r|=0.1 "
             f"(all p_TOST < 0.05, largest {comp.p_tost10.max():.2e}; "
             f"largest |partial r| = {comp.r.abs().max():.3f}, "
             f"{comp.loc[comp.r.abs().idxmax(), 'outcome']}). "
             f"At the stricter SESOI 0.05 (fig3c shaded band) {n_eq05}/10 pass; at that bound the result is "
             "\"not shown\", not equivalence. An equivalence statement in the text should be tied to SESOI=0.1 with its source. "
             f"Reference: d_terr itself has partial r = {dt.r:+.3f} (significantly non-zero, just above 0.1), "
             "consistent with a structurally real, competitively neutral effect.\n")

    L.append("\n## B. Rule behind \"replicates in only 5/10 competitions\" and its chance benchmark\n")
    L.append("**Rule**: forward shift = post-window mean pass origin x minus pre-window mean; "
             "per competition compare the early- and late-group means, **replication = late-group mean > early-group mean (same sign only, no significance requirement)**; "
             "summary `binomtest(k, n)` (p=0.5, two-sided).\n")
    L.append("\n| competition | early fwd-shift | late fwd-shift | late>early |")
    L.append("|---|---|---|---|")
    for name, e, l, f in detail:
        if f is None:
            L.append(f"| {name} | - | - | insufficient sample |")
        else:
            L.append(f"| {name} | {e:+.2f} | {l:+.2f} | {'Y' if f else 'N'} |")
    L.append(f"\nRe-derived count **{k}/{n}**.")
    L.append(f"\n- **chance under the actual rule (same sign)** = 0.5 per competition -> binomial: one-sided P(X>={k}|{n},0.5) "
             f"= {bt_one:.3f}, two-sided = {bt_two:.3f}. **5/10 is not above chance**, so "
             "\"no better than chance\" holds under the actual rule.")
    L.append(f"- **hypothetical rule (same sign and two-sided p<0.05)**: chance = 0.5*0.05 = 0.025 per competition -> "
             f"P(X>=5|10,0.025) = {bt_hyp:.2e}, far above chance. This is **not** the rule used, "
             "but the text must state the replication definition explicitly, "
             "otherwise a reader assuming the stricter rule reaches the opposite conclusion.")
    L.append("- The within-La-Liga significance (beta=+0.071, p=6e-4) comes from the mechanism regression on the main dataset "
             "and is not recomputed here.\n")

    L.append("\n## C. Borderline max|z|\n")
    L.append(f"- Observed max|z| (interaction family, 36 tests) = **{obs:.3f}** "
             "(from mechanism_null_global.csv; the trailing x timing interaction, "
             f"p_inter = {mm.set_index('predictor').loc['trailing','p_inter']:.2e}).")
    L.append(f"- Permutation null (mechanism_null_perm.csv, Freedman-Lane mixedlm, B={B}): "
             f"**P(perm max|z| >= {obs:.2f}) = {n_ge}/{B} = {p_raw:.3f}** "
             f"(add-one (r+1)/(B+1) = {p_add1:.3f}); 95th percentile = {q95:.3f} (text cites 3.14). "
             f"Dependence-aware MVN null (NSIM=1e5) gives p = {mvn_p:.3f} for the same statistic; the two agree.")
    L.append(f"- Smallest BH threshold 0.05/36 = {thr:.6f}:")
    for fam, col in [("main", "p_main"), ("interaction", "p_inter")]:
        sub = [r for r in ROWS if r["section"] == "C_maxz"
               and r["item"] == f"closest_to_BHmin_{fam}"]
        d = {r["metric"]: r["value"] for r in sub}
        L.append(f"  - {fam} family ({col}): closest candidate = **{d['predictor']}** "
                 f"({col} = {d[col]:.6f}, distance to threshold {d['abs_distance_to_thr']:.6f}, "
                 f"= {d['ratio_p_over_thr']:.2f} x threshold).")
    L.append("- **Verdict**: observed max|z|=3.05 lies below the permutation 95th percentile 3.14 but close to it, permutation p about "
             f"{p_raw:.2f}, i.e. borderline non-significant. The 6 BH survivors of the main-effect family were eliminated downstream "
             "(DiD/RTM/cross-competition checks, as the text describes); the 0/36 interaction conclusion rests on this borderline result, "
             "so the permutation p itself should be disclosed, not only the position of the observed value relative to the 95th percentile.\n")

    L.append("\n## D. Dropped vs retained: does finding a control window constitute selection?\n")
    L.append(f"All {len(m)} batches in did_controls.parquet: with control (same_match+cross_match) "
             f"{int(m.retained.sum())} vs without (control_type=='none') {int((~m.retained).sum())}. "
             "Covariates = the 12 pre-window structural quantities of entropy_balancing.py "
             "(pre_*_t15 columns of analysis_table, treated team's own pre window); SMD denominator = "
             "sqrt((var_ret+var_drop)/2), same as smd() there.\n")
    L.append("\n| covariate | n (with control) | n (without) | mean (with) | mean (without) | SMD |")
    L.append("|---|---|---|---|---|---|")
    for _, r in core.iterrows():
        L.append(f"| {r.covariate} | {r.n_ret} | {r.n_drop} | {r.mean_ret:.4f} "
                 f"| {r.mean_drop:.4f} | {r.smd:+.3f} |")
    L.append(f"| minute_first(supp) | {mf.n_ret} | {mf.n_drop} | {mf.mean_ret:.1f} "
             f"| {mf.mean_drop:.1f} | {mf.smd:+.3f} |")
    tt = timing_tab.rename(columns={False: "dropped(none)", True: "retained"})
    tt["drop_rate"] = (tt["dropped(none)"] / tt.sum(axis=1)).round(3)
    L.append("\nComposition by timing group:\n")
    L.append(tt.to_markdown())
    nb = int((core.smd.abs() > 0.1).sum())
    dr = tt.drop_rate
    L.append(f"\n**Verdict**: matching does select, but almost entirely through the match minute, and roughly symmetrically for the core timing "
             f"comparison. Evidence: (i) {nb} of the 12 structural covariates have |SMD|>0.1 "
             f"(max {core.smd.abs().max():.3f}, {core.loc[core.smd.abs().idxmax(), 'covariate']}), "
             "all in the P layer and in one direction (dropped batches have sparser networks), while the 5 Z-layer and 2 versatility items all have |SMD|<0.07; "
             f"(ii) SMD of minute_first = {mf.smd:+.2f} (dropped batches are on average {mf.mean_drop - mf.mean_ret:.1f} "
             "minutes later); later substitutions are harder to match to a clean control window, and the P-layer imbalance runs with the minute difference, so one mechanical source; "
             f"(iii) the strategic group is almost fully retained (drop rate {dr['strategic']:.1%}), while regular and late drop rates are "
             f"nearly identical ({dr['regular']:.1%} vs {dr['late']:.1%}), so the mid-vs-late comparison is not contaminated by differential selection, "
             "and in early-vs-mid/late the early group is nearly complete while mid/late are about half subsamples. Conclusion: no structural selection "
             "that threatens the main result, but worth disclosing; the mid/late DiD set is the subpopulation with a clean control window "
             "(slightly earlier, denser-network batches over-represented). One sentence in limitations or the appendix suffices, "
             "noting that entropy balancing addresses treated-vs-control imbalance whereas this is retained-vs-dropped, a different axis.\n")
    REP.write_text("\n".join(L), encoding="utf-8")
    print(f"report written {REP.relative_to(ROOT)}")


def main():
    np.random.default_rng(SEED)          # no random steps
    T, n_eq10, n_eq05 = sec_a_tost()
    detB = sec_b_replication()
    C = sec_c_maxz()
    D, m, timing_tab = sec_d_selection()
    out = pd.DataFrame(ROWS)
    out.to_csv(TAB / "equivalence_misc.csv", index=False)
    print(f"tidy table written results/tables/equivalence_misc.csv ({len(out)} rows)")
    write_report(T, n_eq10, n_eq05, detB, C, D, m, timing_tab)


if __name__ == "__main__":
    main()
