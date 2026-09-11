"""
Pipeline step 3: derive the analysis sample from the 3528 raw substitutions. E1 drop first
half, E2 drop injury (outcome "Injury" or an Injury Stoppage of the same player within 120 s),
E3 drop goalkeepers; chain-merge same-team substitutions with gap <= tau*, the valley of a
log10 KDE (Silverman and Sheather-Jones bandwidths; stop if the valleys differ by > 30 s).
Inputs: data/events_substitution.parquet, data/events_pass.parquet, raw events (Injury Stoppage only, cached)
Outputs: data/{substitution_batches.csv, injury_subsample.csv, exclusion_chain.csv},
         results/figures/kde_threshold.{pdf,png}, results/figures/qc/firsthalf_subs_outcomes.{pdf,png}
Run: python code/pipeline/03_build_sample.py [--tau 88.0]
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde

RAW_DIR = Path(
    os.environ.get("STATSBOMB_EVENTS_DIR", "path/to/statsbomb_event_data/11_281_2023_24")
)
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
FIG = ROOT / "results" / "figures"

INJURY_PROXY_WINDOW_SEC = 120.0  # injury stoppage lookback before the sub
FIRSTHALF_TACTICAL_STOP = 0.30  # E1 warning: tactical share of first-half subs
G1_VALLEY_DIFF_SEC = 30.0  # check: max gap between the two bandwidth valleys
MINUTE_RULE_GAP = 2  # sensitivity: integer-minute rule
# timing cut points, minutes
CUT_STRATEGIC, CUT_REGULAR = 60, 75


def parse_ts_sec(ts: str) -> float:
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


# ---------------------------------------------------------------- SJ bandwidth
def _phi4(x):
    return (x**4 - 6 * x**2 + 3) * np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)


def _phi6(x):
    return (x**6 - 15 * x**4 + 45 * x**2 - 15) * np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)


def bw_sheather_jones(x: np.ndarray) -> float:
    """Sheather and Jones (1991) solve-the-equation bandwidth, Gaussian kernel,
    per Wand and Jones (1995) 3.6.2; O(n^2) via broadcasting."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    lam = min(np.std(x, ddof=1), iqr / 1.349)
    a = 0.920 * lam * n ** (-1 / 7)
    b = 0.912 * lam * n ** (-1 / 9)
    diff = x[:, None] - x[None, :]
    tdb = -np.sum(_phi6(diff / b)) / (n * (n - 1) * b**7)
    sda = np.sum(_phi4(diff / a)) / (n * (n - 1) * a**5)

    def sd_alpha(h):
        alpha2 = 1.357 * (sda / tdb) ** (1 / 7) * h ** (5 / 7)
        return np.sum(_phi4(diff / alpha2)) / (n * (n - 1) * alpha2**5)

    def fixed_point(h):
        return h - (1.0 / (2 * np.sqrt(np.pi) * n * sd_alpha(h))) ** 0.2

    h0 = 1.06 * lam * n ** (-0.2)  # normal reference start
    lo, hi = h0 / 50, h0 * 5
    flo, fhi = fixed_point(lo), fixed_point(hi)
    if flo * fhi > 0:  # widen bracket once
        lo, hi = h0 / 500, h0 * 20
        flo, fhi = fixed_point(lo), fixed_point(hi)
        if flo * fhi > 0:
            raise RuntimeError("Sheather-Jones bandwidth: root bracketing failed")
    return brentq(fixed_point, lo, hi)


def _selftest_sj():
    """SJ bandwidth on a large N(0,1) sample should be near the normal reference 1.06 n^(-1/5)."""
    rng = np.random.default_rng(0)
    z = rng.normal(size=2000)
    h = bw_sheather_jones(z)
    h_ref = 1.06 * np.std(z, ddof=1) * 2000 ** (-0.2)
    assert 0.3 * h_ref < h < 1.7 * h_ref, f"SJ self-test failed: h={h:.4f}, ref={h_ref:.4f}"
    return h, h_ref


# ---------------------------------------------------------------- KDE valley
def kde_valley_log(log_gaps: np.ndarray, bw_log: float, grid_lo=-0.2, grid_hi=3.6):
    """log10-scale KDE at a log10 bandwidth -> (grid_log, density, valley s, two main peaks s).
    Valley and peaks found on the log grid, back-transformed with 10**x."""
    kde = gaussian_kde(log_gaps, bw_method=bw_log / np.std(log_gaps, ddof=1))
    grid = np.linspace(grid_lo, grid_hi, 2000)
    dens = kde(grid)
    peaks, props = find_peaks(dens, prominence=0)
    if len(peaks) < 2:
        return grid, dens, None, []
    top2 = sorted(peaks[np.argsort(props["prominences"])[-2:]])
    p_lo, p_hi = top2
    valley_log = grid[p_lo + np.argmin(dens[p_lo:p_hi])]
    return grid, dens, float(10**valley_log), [float(10 ** grid[p]) for p in top2]


# ---------------------------------------------------------------- injury stoppages
def load_injury_stoppages() -> pd.DataFrame:
    cache = DATA / "events_injury_stoppage.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    rows = []
    for fp in sorted(RAW_DIR.glob("*_events.json")):
        with open(fp) as f:
            events = json.load(f)
        mid = int(fp.stem.split("_")[0])
        for e in events:
            if e.get("type", {}).get("name") == "Injury Stoppage":
                rows.append(
                    {
                        "match_id": mid,
                        "period": e.get("period"),
                        "t_sec": parse_ts_sec(e["timestamp"]),
                        "team_id": e.get("team", {}).get("id"),
                        "player_id": (e.get("player") or {}).get("id"),
                    }
                )
    df = pd.DataFrame(rows)
    df.to_parquet(cache, index=False)
    return df


# ---------------------------------------------------------------- chain merge
def chain_merge(df: pd.DataFrame, by_seconds: bool, threshold: float) -> pd.Series:
    """Batch ids aligned with df (needs t_sec and minute, sorted by time).
    by_seconds: gap <= threshold s; else integer minute gap <= threshold."""
    key = df["t_sec"] if by_seconds else df["minute"]
    gap = key.groupby([df["match_id"], df["team_id"], df["period"]]).diff()
    new_batch = (gap.isna()) | (gap > threshold)
    return new_batch.cumsum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=None,
                    help="merge threshold tau* in seconds; default: Silverman KDE valley")
    args = ap.parse_args()

    (FIG / "qc").mkdir(parents=True, exist_ok=True)
    sub = pd.read_parquet(DATA / "events_substitution.parquet").copy()
    sub["t_sec"] = sub["timestamp"].map(parse_ts_sec)
    sub = sub.sort_values(["match_id", "team_id", "period", "t_sec"]).reset_index(drop=True)

    chain = [("raw substitution events", len(sub), "")]

    # ---- step 0: completeness
    n_matches = sub.match_id.nunique()
    assert n_matches == 380, f"substitution table covers {n_matches} matches, expected 380"
    chain.append(("data completeness", len(sub), "all 380 matches present, 0 excluded"))

    # ---- E1: first half
    fh = sub[sub.period == 1]
    oc = fh.outcome.value_counts()
    tactical_share = (fh.outcome == "Tactical").mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    oc.plot.bar(ax=ax, edgecolor="white")
    ax.set_title(f"First-half substitutions by outcome (n={len(fh)})")
    ax.set_ylabel("Count")
    for i, v in enumerate(oc.values):
        ax.text(i, v, str(v), ha="center", va="bottom")
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(FIG / "qc" / f"firsthalf_subs_outcomes.{ext}", dpi=300)
    print(f"[E1] First-half substitutions {len(fh)}; outcome distribution {oc.to_dict()}; "
          f"tactical share {tactical_share:.1%} (warning threshold {FIRSTHALF_TACTICAL_STOP:.0%})")
    if tactical_share > FIRSTHALF_TACTICAL_STOP:
        print("WARNING: tactical first-half substitutions exceed 30%, check the data before proceeding")
    sub = sub[sub.period != 1]
    chain.append(("E1 first-half substitutions", len(sub), f"excluded {len(fh)}; tactical share {tactical_share:.1%}"))

    # ---- E2: injury (outcome or proxy; proxy window on the continuous clock, may span half time)
    inj_stop = load_injury_stoppages()
    print(f"[E2] Injury Stoppage events {len(inj_stop)} (cached in events_injury_stoppage.parquet)")
    t1_end = pd.concat([
        pd.read_parquet(DATA / "events_pass.parquet",
                        columns=["match_id", "period", "t_period_sec"])
        .query("period == 1").groupby("match_id").t_period_sec.max(),
        inj_stop.query("period == 1").groupby("match_id").t_sec.max(),
    ], axis=1).max(axis=1)
    inj_stop = inj_stop.assign(
        t_cum=np.where(inj_stop.period == 1, inj_stop.t_sec,
                       inj_stop.match_id.map(t1_end) + inj_stop.t_sec))
    sub = sub.assign(t_cum=np.where(sub.period == 1, sub.t_sec,
                                    sub.match_id.map(t1_end) + sub.t_sec))
    inj_outcome = sub.outcome == "Injury"
    is_map = inj_stop.groupby(["match_id", "team_id"])
    proxy_flags, proxy_player_flags = [], []
    for _, r in sub.iterrows():
        try:
            g = is_map.get_group((r.match_id, r.team_id))
        except KeyError:
            proxy_flags.append(False)
            proxy_player_flags.append(False)
            continue
        win = g[(g.t_cum >= r.t_cum - INJURY_PROXY_WINDOW_SEC) & (g.t_cum <= r.t_cum)]
        proxy_flags.append(len(win) > 0)
        proxy_player_flags.append((win.player_id == r.player_out_id).any())
    sub = sub.assign(inj_outcome=inj_outcome.values,
                     inj_proxy_team=proxy_flags,
                     inj_proxy_player=proxy_player_flags)
    n_o, n_pt, n_pp = sub.inj_outcome.sum(), sub.inj_proxy_team.sum(), sub.inj_proxy_player.sum()
    n_both = (sub.inj_outcome & sub.inj_proxy_player).sum()
    excl_inj = sub.inj_outcome | sub.inj_proxy_player  # union of outcome and strict proxy
    print(f"     outcome {n_o}; same-team proxy (120 s) {n_pt}; strict proxy (same player) {n_pp}; "
          f"intersection {n_both}; union excluded {excl_inj.sum()}")
    print(f"     overlap: strict-proxy hits also labelled Injury = "
          f"{n_both / n_pp if n_pp else float('nan'):.1%}; "
          f"Injury-labelled also hit by the proxy = {n_both / n_o if n_o else float('nan'):.1%}")
    injury_sub = sub[excl_inj].drop(columns=["inj_outcome", "inj_proxy_team", "inj_proxy_player"])
    injury_sub.to_csv(DATA / "injury_subsample.csv", index=False)
    sub = sub[~excl_inj]
    chain.append(("E2 injury substitutions", len(sub),
                  f"excluded {len(injury_sub)} (outcome {n_o} or strict proxy {n_pp}, intersection {n_both}); subsample saved"))

    # ---- E3: goalkeepers
    gk = sub[sub.position_out == "Goalkeeper"]
    print(f"[E3] Goalkeeper substitutions {len(gk)} (reference 3): "
          f"{gk[['match_id', 'minute', 'player_out_name']].to_dict('records')}")
    sub = sub[sub.position_out != "Goalkeeper"]
    chain.append(("E3 goalkeeper substitutions", len(sub), f"excluded {len(gk)} (reference 3)"))

    # ---- KDE threshold
    gaps = sub.groupby(["match_id", "team_id", "period"]).t_sec.diff().dropna().values
    print(f"\n[KDE] Same-team consecutive substitution gaps n={len(gaps)}; "
          f"median {np.median(gaps):.0f}s, P25 {np.percentile(gaps, 25):.0f}s, "
          f"P75 {np.percentile(gaps, 75):.0f}s, max {gaps.max():.0f}s; "
          f"share <= 120 s {(gaps <= 120).mean():.1%}")
    h_sj_test = _selftest_sj()
    print(f"     SJ self-test passed (N(0,1) n=2000: h={h_sj_test[0]:.4f}, normal reference {h_sj_test[1]:.4f})")

    # zero gaps stay out of the KDE and always merge
    n_zero = int((gaps == 0).sum())
    log_gaps = np.log10(gaps[gaps > 0])
    print(f"     zero-gap (same second) substitutions {n_zero}; excluded from the KDE, always merged")
    h_silver = (0.9 * min(np.std(log_gaps, ddof=1),
                          np.subtract(*np.percentile(log_gaps, [75, 25])) / 1.349)
                * len(log_gaps) ** (-0.2))
    h_sj = bw_sheather_jones(log_gaps)
    g_s, d_s, v_s, peaks_s = kde_valley_log(log_gaps, h_silver)
    g_j, d_j, v_j, peaks_j = kde_valley_log(log_gaps, h_sj)
    print(f"     Silverman bandwidth {h_silver:.3f} (log10) -> main peaks {[f'{p:.0f}s' for p in peaks_s]}, valley {v_s:.1f}s")
    print(f"     Sheather-Jones bandwidth {h_sj:.3f} (log10) -> main peaks {[f'{p:.0f}s' for p in peaks_j]}, valley {v_j:.1f}s")
    g1 = v_s is None or v_j is None or abs(v_s - v_j) > G1_VALLEY_DIFF_SEC
    print(f"     G1 check (log scale, valleys back-transformed to seconds): valley difference "
          f"{abs(v_s - v_j):.1f}s (threshold {G1_VALLEY_DIFF_SEC}s) -> "
          f"{'G1 TRIGGERED, stop' if g1 else 'not triggered'}" if (v_s and v_j) else "     G1: valley missing, stop")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    bins_log = np.linspace(-0.2, 3.6, 60)
    axes[0].hist(log_gaps, bins=bins_log, density=True, alpha=0.35, color="grey",
                 label="log10(gap) histogram")
    axes[0].plot(g_s, d_s, lw=1.8, label=f"Silverman h={h_silver:.3f}, valley={v_s:.0f}s")
    axes[0].plot(g_j, d_j, lw=1.8, ls="--", label=f"Sheather-Jones h={h_sj:.3f}, valley={v_j:.0f}s")
    for v, c in [(v_s, "C0"), (v_j, "C1")]:
        axes[0].axvline(np.log10(v), color=c, ls=":", lw=1)
    axes[0].set_xlabel("log10(gap in seconds)")
    axes[0].set_ylabel("Density")
    axes[0].set_title(f"log-scale KDE (n={len(log_gaps)}, zero-gaps excluded: {n_zero})")
    axes[0].legend(fontsize=8)
    # right panel: linear seconds axis, same valleys back-transformed
    axes[1].hist(gaps, bins=np.arange(0, 1500, 15), density=True, alpha=0.35,
                 color="grey", label="gap histogram (15s bins)")
    for v, c, lbl in [(v_s, "C0", "Silverman"), (v_j, "C1", "SJ")]:
        axes[1].axvline(v, color=c, ls=":", lw=1.5, label=f"{lbl} valley = {v:.0f}s")
    axes[1].set_xlim(0, 1500)
    axes[1].set_xlabel("Gap between consecutive same-team substitutions (s)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("linear view with back-transformed valleys")
    axes[1].legend(fontsize=8)
    fig.suptitle(f"KDE batch-merge threshold, log10 scale (n={len(gaps)} gaps, after E1-E3)")
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(FIG / f"kde_threshold.{ext}", dpi=300)
    print(f"     figure saved to {FIG}/kde_threshold.[pdf|png]")
    if g1:
        print("G1 triggered: stopping, batch table not written")
        pd.DataFrame(chain, columns=["step", "N_after", "note"]).to_csv(
            DATA / "exclusion_chain.csv", index=False)
        return

    tau = args.tau if args.tau is not None else v_s
    print(f"\n[merge] tau* = {tau:.1f}s ({'user-specified' if args.tau else 'Silverman valley, pending confirmation'})")

    # ---- chain merge: KDE rule plus integer-minute reference rule
    sub = sub.assign(batch_kde=chain_merge(sub, True, tau),
                     batch_min=chain_merge(sub, False, MINUTE_RULE_GAP))

    def batch_table(df, col):
        agg = df.groupby(col).agg(
            match_id=("match_id", "first"), team_id=("team_id", "first"),
            team_name=("team_name", "first"), period=("period", "first"),
            n_subs=("event_uuid", "size"), minute_first=("minute", "first"),
            minute_last=("minute", "last"), t_sec_first=("t_sec", "first"),
            t_sec_last=("t_sec", "last"),
            players_out=("player_out_name", lambda s: "|".join(s)),
            players_out_id=("player_out_id", lambda s: "|".join(map(str, s))),
            players_in=("player_in_name", lambda s: "|".join(s)),
            players_in_id=("player_in_id", lambda s: "|".join(map(str, s))),
            event_uuids=("event_uuid", lambda s: "|".join(s)),
        ).reset_index(drop=True)
        agg["timing_group"] = np.where(agg.minute_first <= CUT_STRATEGIC, "strategic",
                              np.where(agg.minute_first <= CUT_REGULAR, "regular", "late"))
        return agg

    batches = batch_table(sub, "batch_kde")
    batches.insert(0, "batch_id", range(1, len(batches) + 1))
    n_min_rule = sub.batch_min.nunique()
    tg = batches.timing_group.value_counts()
    print(f"     KDE-rule batches N = {len(batches)}; integer-minute (<= 2) reference rule N = {n_min_rule}")
    print(f"     timing groups (cuts <={CUT_STRATEGIC}/{CUT_STRATEGIC + 1}-{CUT_REGULAR}/>{CUT_REGULAR}, by first substitution minute): "
          f"strategic {tg.get('strategic', 0)} / regular {tg.get('regular', 0)} / late {tg.get('late', 0)}")
    print(f"     reference: 2077 (445/829/803), directional comparison only")
    print(f"     batch size distribution: {batches.n_subs.value_counts().sort_index().to_dict()}")

    batches.to_csv(DATA / "substitution_batches.csv", index=False)
    pd.DataFrame(chain + [("KDE chain merge", len(batches), f"tau*={tau:.1f}s")],
                 columns=["step", "N_after", "note"]).to_csv(DATA / "exclusion_chain.csv", index=False)
    print(f"\nSaved: substitution_batches.csv ({len(batches)} batches), exclusion_chain.csv, "
          f"injury_subsample.csv ({len(injury_sub)})")


if __name__ == "__main__":
    main()
