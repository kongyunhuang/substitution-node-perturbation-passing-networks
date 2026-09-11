"""
Pre-registered window-length convergence analysis (Supplementary Table S6, Figure S1).
Stratified sample of 300 batches, W in {5,8,10,12,15,20} min, pre-windows; density, clustering,
betweenness, lambda2 on the player and 6x4 zone layers. Criteria: adjacent-W CV change < 10%
and Spearman rho vs the 20-min window >= 0.9; W = shortest passing both, else Spearman only.
Post-window curves on the untruncated-20-min subsample as supporting evidence.
Inputs: data/events_pass.parquet, substitution_batches.csv, events_substitution.parquet, windows/{window_specs,batch_flags}.parquet
Outputs: data/windows/convergence_metrics.parquet, results/tables/convergence_{cv_pre,rho_pre,decision}.csv, results/figures/convergence_curves.{pdf,png}
Run: python code/pipeline/04b_window_convergence.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics_core import (
    betweenness_inv_w,
    density_rate,
    fagiolo_clustering,
    lambda2_sym,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
TAB = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"

W_GRID = [5, 8, 10, 12, 15, 20]
N_SAMPLE = 300
SEED = 20260611
CV_PLATEAU = 0.10  # relative CV change between adjacent W
RHO_SAT = 0.90  # Spearman rho vs the 20-min window
NX, NY = 6, 4  # zone grid


def parse_ts_sec(ts: str) -> float:
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def player_adj(df: pd.DataFrame) -> np.ndarray:
    """Player-layer adjacency; nodes = players involved in window passes."""
    ids = pd.unique(pd.concat([df.player_id, df.recipient_id]).dropna())
    idx = {p: i for i, p in enumerate(ids)}
    W = np.zeros((len(ids), len(ids)))
    for p, r in zip(df.player_id, df.recipient_id):
        W[idx[p], idx[r]] += 1
    return W


def zone_adj(df: pd.DataFrame) -> np.ndarray:
    """6x4 zone-layer adjacency, self-loops removed."""
    zx = np.clip((df.x / (120 / NX)).astype(int), 0, NX - 1)
    zy = np.clip((df.y / (80 / NY)).astype(int), 0, NY - 1)
    ex = np.clip((df.end_x / (120 / NX)).astype(int), 0, NX - 1)
    ey = np.clip((df.end_y / (80 / NY)).astype(int), 0, NY - 1)
    src = zx * NY + zy
    dst = ex * NY + ey
    W = np.zeros((NX * NY, NX * NY))
    np.add.at(W, (src, dst), 1.0)
    np.fill_diagonal(W, 0.0)
    return W


def compute_metrics(df: pd.DataFrame, minutes: float) -> dict:
    out = {}
    for layer, W in [("P", player_adj(df)), ("Z", zone_adj(df))]:
        if W.shape[0] < 3 or W.sum() == 0:
            out.update({f"{m}_{layer}": np.nan for m in
                        ["density", "clustering", "betweenness", "lambda2"]})
            continue
        out[f"density_{layer}"] = density_rate(W, minutes)
        out[f"clustering_{layer}"] = float(np.mean(fagiolo_clustering(W)))
        out[f"betweenness_{layer}"] = float(np.mean(betweenness_inv_w(W, normalized=True)))
        out[f"lambda2_{layer}"] = lambda2_sym(W)
    return out


def main():
    passes = pd.read_parquet(DATA / "events_pass.parquet")
    batches = pd.read_csv(DATA / "substitution_batches.csv")
    specs = pd.read_parquet(DATA / "windows" / "window_specs.parquet")
    flags = pd.read_parquet(DATA / "windows" / "batch_flags.parquet")
    subs_all = pd.read_parquet(DATA / "events_substitution.parquet")
    subs_all["t_sec"] = subs_all["timestamp"].map(parse_ts_sec)

    t1_end = (
        pd.concat([
            passes.loc[passes.period == 1, ["match_id", "t_period_sec"]],
            subs_all.loc[subs_all.period == 1, ["match_id", "t_sec"]]
            .rename(columns={"t_sec": "t_period_sec"}),
        ]).groupby("match_id").t_period_sec.max())
    passes = passes.assign(
        t_cum=np.where(passes.period == 1, passes.t_period_sec,
                       passes.match_id.map(t1_end) + passes.t_period_sec))
    net = passes[passes.outcome.isna() & ~passes.is_set_piece]
    net_idx = {k: g for k, g in net.groupby(["match_id", "team_id"])}

    # ---- stratified sample of 300, proportional to timing groups
    rng = np.random.default_rng(SEED)
    sample_ids = []
    for tg, n_total in batches.timing_group.value_counts().items():
        n_take = round(N_SAMPLE * n_total / len(batches))
        pool = batches.loc[batches.timing_group == tg, "batch_id"].values
        sample_ids += list(rng.choice(pool, n_take, replace=False))
    print(f"Stratified sample of {len(sample_ids)} batches (seed={SEED}): "
          f"{batches[batches.batch_id.isin(sample_ids)].timing_group.value_counts().to_dict()}")

    # post-window subsample: 20-min post-window not truncated
    untrunc20 = set(flags[(flags.W_min == 20) & ~flags.post_truncated].batch_id)
    post_ids = [b for b in sample_ids if b in untrunc20]
    print(f"Post-window subsample (20-min post-window not truncated): {len(post_ids)} batches")

    # ---- per batch, per W
    spec_idx = specs[specs.kind == "time"].set_index(["batch_id", "W_min", "side"])
    rows = []
    for n_done, bid in enumerate(sample_ids, 1):
        b = batches[batches.batch_id == bid].iloc[0]
        g = net_idx[(b.match_id, b.team_id)]
        for W in W_GRID:
            for side in ["pre", "post"]:
                if side == "post" and bid not in untrunc20:
                    continue
                s = spec_idx.loc[(bid, W, side)]
                win = g[(g.t_cum >= s.t_start) & (g.t_cum < s.t_end)] if side == "pre" \
                    else g[(g.t_cum > s.t_start) & (g.t_cum <= s.t_end)]
                m = compute_metrics(win, minutes=s.actual_len_min)
                rows.append(dict(batch_id=bid, W_min=W, side=side,
                                 timing_group=b.timing_group, n_passes=len(win), **m))
        if n_done % 50 == 0:
            print(f"  ... {n_done}/{len(sample_ids)}")

    M = pd.DataFrame(rows)
    M.to_parquet(DATA / "windows" / "convergence_metrics.parquet", index=False)

    metrics = [f"{m}_{l}" for m in ["density", "clustering", "betweenness", "lambda2"]
               for l in ["P", "Z"]]

    # ---- criteria tables, main evidence = pre-windows
    def criteria_table(side):
        d = M[M.side == side]
        cv = d.groupby("W_min")[metrics].agg(lambda x: x.std() / abs(x.mean()))
        wide = d.pivot(index="batch_id", columns="W_min", values=metrics)
        rho = pd.DataFrame(index=W_GRID, columns=metrics, dtype=float)
        for m in metrics:
            for W in W_GRID:
                a = wide[(m, W)].astype(float)
                bb = wide[(m, 20)].astype(float)
                ok = a.notna() & bb.notna()
                rho.loc[W, m] = spearmanr(a[ok], bb[ok]).statistic if ok.sum() > 10 else np.nan
        return cv, rho

    cv_pre, rho_pre = criteria_table("pre")
    print("\n===== Criterion 1: CV (pre-windows) =====")
    print(cv_pre.round(3).to_string())
    print("\n===== Criterion 2: Spearman rho vs 20 min (pre-windows) =====")
    print(rho_pre.round(3).to_string())

    # joint decision
    decisions = {}
    for i, W in enumerate(W_GRID[:-1]):
        W_next = W_GRID[i + 1]
        cv_ok = (abs(cv_pre.loc[W_next] - cv_pre.loc[W]) / cv_pre.loc[W] < CV_PLATEAU).all()
        rho_ok = (rho_pre.loc[W] >= RHO_SAT).all()
        decisions[W] = dict(cv_plateau=bool(cv_ok), rho_sat=bool(rho_ok),
                            both=bool(cv_ok and rho_ok))
    dec_df = pd.DataFrame(decisions).T
    print("\n===== Joint criteria (pre-windows, all 8 metrics must pass) =====")
    print(dec_df.to_string())
    both_ok = [W for W, d in decisions.items() if d["both"]]
    rho_only = [W for W, d in decisions.items() if d["rho_sat"]]
    if both_ok:
        W_main = min(both_ok)
        rule = "both criteria"
    elif rho_only:
        W_main = min(rho_only)
        rule = "Spearman only (no W satisfies both criteria)"
    else:
        W_main = None
        rule = "no W satisfies either criterion"
    print(f"\n>>> Main window W = {W_main} min, rule: {rule}")

    # criteria tables to disk
    cv_pre.round(6).to_csv(TAB / "convergence_cv_pre.csv")
    rho_pre.round(6).to_csv(TAB / "convergence_rho_pre.csv")
    dec_df.to_csv(TAB / "convergence_decision.csv")
    print("saved convergence_{cv_pre,rho_pre,decision}.csv")

    # post-window supporting evidence
    cv_post, rho_post = criteria_table("post")
    print("\n===== Supporting evidence: Spearman rho (post-windows, 20-min untruncated subsample) =====")
    print(rho_post.round(3).to_string())

    # ---- three-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    tspec = specs[(specs.kind == "time")]
    med = tspec.groupby(["W_min", "side"]).n_passes.median().unstack()
    q1 = tspec.groupby(["W_min", "side"]).n_passes.quantile(0.25).unstack()
    q3 = tspec.groupby(["W_min", "side"]).n_passes.quantile(0.75).unstack()
    for side, c in [("pre", "C0"), ("post", "C1")]:
        axes[0].plot(W_GRID, med[side], "-o", color=c, label=f"{side} median")
        axes[0].fill_between(W_GRID, q1[side], q3[side], color=c, alpha=0.15)
    axes[0].set_xlabel("Window length (min)")
    axes[0].set_ylabel("Completed open-play passes")
    axes[0].set_title("(1) Passes per window (all 2,075 batches)")
    axes[0].legend()
    for m in metrics:
        axes[1].plot(W_GRID, cv_pre[m], "-o", ms=3, label=m)
        axes[2].plot(W_GRID, rho_pre[m], "-o", ms=3, label=m)
    axes[1].set_xlabel("Window length (min)")
    axes[1].set_ylabel("CV across batches")
    axes[1].set_title("(2) Between-batch CV (pre)")
    axes[2].axhline(RHO_SAT, color="red", ls="--", lw=1)
    axes[2].set_xlabel("Window length (min)")
    axes[2].set_ylabel("Spearman ρ vs 20-min window")
    axes[2].set_title("(3) Rank stability vs 20 min (pre)")
    axes[2].legend(fontsize=6, ncol=2)
    if W_main:
        for ax in axes[1:]:
            ax.axvline(W_main, color="green", ls=":", lw=1.5)
    fig.suptitle(f"Window convergence analysis (n={len(sample_ids)} stratified, seed={SEED}) "
                 f"→ W = {W_main} min")
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(FIG / f"convergence_curves.{ext}", dpi=300)
    print(f"\nFigure saved to {FIG}/convergence_curves.[pdf|png]")


if __name__ == "__main__":
    main()
