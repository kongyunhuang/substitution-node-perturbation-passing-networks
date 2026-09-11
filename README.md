# Substitutions as natural node-perturbation experiments (soccer passing networks)

Code and frozen results for reproducing the paper:

> Liu C, Huang K, Tao R, Zhou C, Gómez-Ruano MÁ, Buldú JM, Gao B.
> *Substitutions as natural node-perturbation experiments: a timing threshold for the spatial
> reorganization of soccer passing networks.*

## What can be reproduced from what

| Tier | Needs | What runs |
| --- | --- | --- |
| 1. Figures and tables from the frozen results | nothing beyond `results/tables/` (shipped) | Figures 2, 4, 5, S1; Tables A.1, B.1, S1 to S6 |
| 2. Open-data replication (ten competitions) | StatsBomb open data, fetched automatically | `code/replication/` and the open-data audits |
| 3. Primary dataset (La Liga 2023/24) | licensed StatsBomb event data (see below) | `code/pipeline/`, `code/analysis/`, the remaining audits, Figures 1 and 3 |

## Data availability

The primary dataset (all 380 matches of La Liga 2023/24) was obtained from StatsBomb under a
commercial licence with Universidad Rey Juan Carlos and **cannot be redistributed**. Requests
should be directed to StatsBomb (https://statsbomb.com). To run Tier 3, point the environment
variable `STATSBOMB_EVENTS_DIR` at a directory holding the per-match event JSON files
(`<match_id>_events.json`) plus the match metadata file, then run the pipeline in order.

The ten replication competitions are taken from the StatsBomb open-data repository
(https://github.com/statsbomb/open-data). `code/replication/fetch_open_data.py`
downloads the event JSON directly from that repository and caches one parquet file per
competition under `data/opendata/`. Nothing from the open data is redistributed here; its use
is subject to StatsBomb's open-data licence and attribution terms.

The batch-level derived files of the primary dataset (`data/*.parquet`) are not shipped because
they are derived from the licensed events. They are available from the corresponding author on
reasonable request for verification.

`results/tables/` ships the frozen aggregate tables that the figures, the typeset tables and the
numbers in the text are read from (48 files, about 1 MB). Two of them hold one scalar per
substitution batch (`dterr_audit.csv`, the territorial shift; `bias_predictability.csv`, the
cross-layer alignment and activity share) because Figures 3 and 5 draw their point clouds from
them; they contain no event data.

## Structure

Pipeline scripts keep a numeric prefix that gives their order; every other script is named
after what it computes. Each file opens with a short docstring listing purpose, inputs, outputs
and the run command.

```
code/
  pipeline/      metrics_core.py and 01_ to 06b_: from licensed event files to windows, matched
                 controls, the two network layers, metrics and difference-in-differences responses
  analysis/      primary-dataset analyses reported in Sections 3 and 4 (mechanism and outcome
                 screens, self-exclusion check, territorial audit, bias predictability, game-state
                 and opponent mechanisms, entropy balancing, artifact exclusion, control-type
                 robustness, players-present-throughout sensitivity)
  replication/   the ten-competition open-data replication (fetch_open_data.py downloads and caches
                 the data; open_data_two_layer.py holds the shared conventions; the other four
                 produce the cross-competition tables)
  synthetic/     synthetic generator with planted coupling and the closed-form bias checks
  audits/        the robustness analyses reported in the paper (Monte-Carlo SE, calibration
                 inversion, cluster-level inference, early-group split, lead tests, equivalence
                 tests, zero-coupling generator check, window-collapse change-point, subtype screen)
  figures/       fig1.py to fig5.py and figS1.py draw the figures; the fig*_*.py files next to them
                 compute the frozen inputs each figure reads (named after the figure they serve)
results/tables/  frozen result tables (shipped)
results/figures/ output directory for the figure scripts
data/            created by Tier 2 and Tier 3 runs (not shipped)
```

## Conventions shared by the scripts

Every analysis script is self-contained on purpose: it loads the data it needs, defines the
few helpers it uses, and writes one table. That keeps each result reproducible in isolation
at the price of some repetition. The shared definitions are:

- Pitch grid: 6 x 4 zones on the provider's 120 x 80 coordinate system (`zbin_vec`), with a
  4 x 3 and a 10 x 7 grid used only in the robustness checks.
- Zone-flow vector `zvec`: the 24 x 24 zone-to-zone matrix of completed open-play passes
  in a window, diagonal removed, normalised to unit total, summarised as in-share plus
  out-share per zone. The teammates-only version drops every pass made or received by a
  substituted player.
- Response: difference-in-differences of `zvec` (post minus pre, treated minus matched
  control), 15-minute windows.
- Signal-to-noise ratio: bootstrap norm of the group mean at a common resampling size
  (800 draws) over the same statistic under per-batch sign flips (400 draws); seeds are
  fixed in each script.
- Timing groups: `strategic` (halftime to 60 min), `regular` (61 to 75), `late` (over 75);
  the paper calls them early, mid and late.

Scripts that re-implement a shared quantity check themselves against the frozen table of
the script that owns it (for example `players_present_sensitivity.py` aborts unless its
headline SNRs match `gradient_selfexcl.csv`, and `bias_replication.py` aborts unless its
naive alignments match the reference table), so the copies cannot drift apart silently.

## Figure and table map

| Output | Script | Reads |
| --- | --- | --- |
| Figure 1 | `figures/fig1.py` | `reorg_maps.npz` (from `fig1_fig2_zone_maps.py`) plus Tier 3 data |
| Figure 2 | `figures/fig2.py` | `role_did.npz` (`fig2_role_responses.py`), `reorg_maps.npz` (`fig1_fig2_zone_maps.py`), `coupling_didzone.npz` (`fig2_role_zone_coupling.py`), `role_positions_empirical.npz` (`fig2_role_positions.py`), `timing_curve.csv` (`fig2_timing_curve.py`), `control_robustness.csv`, `cross_gradient_clean.csv`, `entropy_balance_outcomes.csv` |
| Figure 3 | `figures/fig3.py` | `territory_profile.npz`, `territory_subgroups.csv` (`fig3_territory_profile.py`), `dterr_audit.csv` (`territorial_shift.py`), `consequence_mine.csv` (`outcome_screen.py`) plus `data/analysis_table.parquet` (Tier 3) |
| Figure 4 | `figures/fig4.py` | `mechanism_mine.csv` (`mechanism_screen.py`), `invariance_*.csv` (`fig4_invariance_curves.py`), `mechanism_null_*.csv` (`fig4_mechanism_nulls.py`), `mediation_plane.csv`, `volume_artifact.csv` |
| Figure 5 | `figures/fig5.py` | `bias_replication.csv` (`bias_replication.py`), `bias_predictability.csv` (`bias_predictability.py`), `synthetic_validation.csv` (`synthetic_generator.py`), `closedform_full.csv` (`closed_form_verification.py`), `mvp_closedform.csv` (`closed_form_check.py`) |
| Figure S1 | `figures/figS1.py` | `convergence_rho_pre.csv`, `convergence_cv_pre.csv` (`04b_`) |
| Table A.1 | typeset from `entropy_balance_smd.csv` (`entropy_balancing.py`) | |
| Table B.1 | typeset from `closedform_full.csv` (`closed_form_verification.py`) and `mvp_closedform.csv` (`closed_form_check.py`) | |
| Tables S1 to S6 | typeset from `did_placebo.csv`, `control_robustness.csv` (`control_robustness.py`), `fullwindow_sensitivity.csv` (`players_present_sensitivity.py`), `appendix_d_samples.csv`, `reconciliation_nflow.csv`, `convergence_rho_pre.csv` (`04b_`) | |

The figure scripts write `fig1.pdf` to `fig5.pdf` and `figS1.pdf` (plus PNG previews) to
`results/figures/`, the same files that accompany the manuscript.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Tier 1: redraw Figures 2, 4, 5 and S1 from the shipped tables
python code/figures/fig2.py
python code/figures/fig4.py
python code/figures/fig5.py
python code/figures/figS1.py

# Tier 2: fetch the open data and rerun the ten-competition replication
python code/replication/fetch_open_data.py              # downloads and caches data/opendata/*.parquet
python code/replication/cross_competition_threshold.py  # timing threshold across competitions
python code/replication/bias_replication.py             # shared-activity bias across competitions

# Synthetic generator and closed-form checks (no data needed)
python code/synthetic/synthetic_generator.py
python code/synthetic/closed_form_verification.py

# Tier 3: licensed primary data
export STATSBOMB_EVENTS_DIR=/path/to/statsbomb/11_281_2023_24
python code/pipeline/01_extract_substitutions.py   # then 02_, 03_, 04_, 04b_, 04c_, 05_, 06_, 06b_
```

Run everything from the repository root. Scripts resolve the repository root from their own
location, so the working directory does not matter. Bootstrap and sign-flip procedures use fixed
seeds, so the frozen tables are reproduced exactly on rerun. Python 3.10 was used for all results.

## Notes

- Every headline quantity excludes the passes of the substituted players (the teammates-only
  response); `analysis/self_exclusion_check.py` checks that the timing threshold survives that
  exclusion and `replication/bias_replication.py` quantifies the shared-activity bias that the
  exclusion removes.
- Every script here produces or consumes a number, table or figure that appears in the paper.
  Manuscript tooling and exploratory analyses that the paper does not report are not included.

## Licence

Code is released under the MIT License (see `LICENSE`). The licence covers the code only, not
the StatsBomb data, the open data, or any file derived from them.
