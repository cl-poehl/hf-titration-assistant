# Validation protocol (planned)

This document specifies the real-data validation this project has **not yet run**,
and is the repository's primary open item. It exists so the gap is stated precisely
rather than papered over: the current quantitative numbers (see
[`README.md`](README.md#results)) are a synthetic signal-recovery check and a
tiny-demo ETL smoke test — **neither establishes predictive validity**. What
follows is the design that would.

It is written to align with **TRIPOD+AI** [Collins et al., *BMJ* 2024] and to be
checked for risk of bias against **PROBAST** [Wolff et al., *Ann Intern Med* 2019].
The evaluation harness lives in [`src/validation/`](src/validation/) and runs only
against real data the user supplies (Zigong or MIMIC-IV); it ships no data.

## Why a new dataset is required

The model targets **hospital-at-home (HaH)** heart-failure monitoring, for which no
open dataset exists. The bundled MIMIC-IV / eICU **demo** subsets are (a) ICU, not
home, (b) too short (median stay ≈ 2.7 days vs. a 30-day HaH window), and (c) tiny
(~70 test patients). A defensible result needs a real cohort that is large enough
and matched to a clearly defined task.

## Two tracks

There is no single open dataset that both is large and matches the HaH trajectory
setting, so validation is split into two independent tracks with different
trade-offs. Do **Track A** first — it is a real result reachable in days.

| | Track A — Zigong | Track B — MIMIC-IV |
|---|---|---|
| **Task** | Admission-level readmission | Trajectory-based deterioration |
| **Access** | Restricted: login + DUA, **no** CITI/credentialing | Credentialed: CITI course + reference + review |
| **Cohort** | 2,008 real HF patients | thousands (HF subset) |
| **Data shape** | one row per admission (no time series) | real time series |
| **Validates** | that the ML/eval methodology holds on real HF data | the sliding-window approach itself |
| **Limitation** | not the trajectory model the app uses | ICU, not home (domain gap) |

Track A cannot validate the sliding-window features (Zigong has no time series), but
it does give an honest, literature-comparable readmission result on real patients —
enough to show the modelling and evaluation are sound. Track B validates the
trajectory thesis but needs credentialed data and carries an ICU-vs-home domain gap.

## Track A — Zigong (admission-level readmission)

**Dataset:** *Hospitalized patients with heart failure* (Zigong Fourth People's
Hospital), PhysioNet, **Restricted Access** — a free login and a signed data-use
agreement, **no** credentialing or CITI course. 2,008 patients, 166 baseline
variables, readmission (28 day / 3 / 6 month) and mortality outcomes.

**Task:** predict 28-day readmission (`re.admission.within.28.days`) from
admission-day predictors — the classic HF-readmission benchmark (literature AUROC
≈ 0.73–0.81, see [`DESIGN.md`](DESIGN.md#related-work)).

**Steps:**
1. Sign the DUA and download the dataset from PhysioNet.
2. Build the cohort: `ZIGONG_CSV=/path/to/main.csv python -m src.validation.zigong_cohort`
   → writes `data/processed/features_zigong.parquet`. Confirm the column names in
   `zigong_cohort.py` against the shipped variable-description file; it prints the
   columns it found and drops any outcome-derived column as a leakage guard.
3. Evaluate: `python -m src.validation.run_validation --features data/processed/features_zigong.parquet`
   → three-way patient-level split, held-out test, isotonic recalibration, a
   logistic-regression baseline, and patient-level bootstrap CIs.
4. Report the resulting AUROC / AUPRC / calibration **and the baseline** in the
   README, replacing the synthetic placeholder as the project's real number.

The methodology below (splits, metrics, calibration, baselines, reporting) applies
to both tracks; only the cohort construction differs.

## Track B — MIMIC-IV (trajectory monitoring)

### Data source

**Primary:** MIMIC-IV (full, credentialed) — access requires a free PhysioNet
account, completion of the CITI *"Data or Specimens Only Research"* course, and
signing the data-use agreement. **Secondary / external:** eICU-CRD (full) for an
independent multi-centre external-validation cohort. Neither is redistributed here.

## Target population and index time

- **Population:** adult admissions with a heart-failure diagnosis (ICD-9
  `428.*`; ICD-10 `I50.*`) recorded for the encounter.
- **Index time `t0`:** a fixed anchor per encounter (e.g. 24 h after admission, or
  ICU discharge for a post-ICU-ward analysis), chosen so that only data **before
  `t0`** is used for features — no lookahead.
- **Exclusions:** encounters with < 24 h of pre-`t0` data; in-hospital death before
  the prediction horizon (competing risk, analysed separately).

## Outcome

- **Primary:** clinical deterioration within a fixed horizon after `t0` — a
  composite of ICU transfer, initiation of IV vasoactive/inotropic therapy, or
  death, at **7 days** (matching the synthetic horizon for comparability).
- **Secondary:** 30-day unplanned readmission (the most-reported HF benchmark),
  for comparison against the literature (AUROC ≈ 0.73–0.81; see
  [`DESIGN.md`](DESIGN.md#related-work)).
- Outcome definitions are frozen **before** any modelling.

## Features

Reuse `src/features/build_features.py` unchanged — the same sliding-window vitals,
labs, symptom (where available), baseline-deviation, and missingness features — so
the feature contract is identical to the demo path. Any HaH-only feature without a
real-data analogue (e.g. patient-reported symptoms) is dropped and its absence
reported, not imputed silently.

## Study design and splits

- **Three-way, patient-level split**: train / validation / **held-out test**,
  grouped by `subject_id` so no patient spans folds (the current pipeline already
  does grouped splitting; the change is adding a *validation* fold distinct from
  test). Alternatively, grouped nested CV.
- **The test set is touched exactly once**, after all model and threshold selection
  on train+validation. This fixes a known weakness of the current
  `src/models/train.py`, which uses the test set as the training `eval_set` and for
  best-of-two model selection.
- **Class imbalance** handled by weighting only; prevalence is reported.

## Models and baselines

A model is only credible relative to a baseline. Report, on the same splits:

1. **Baseline A** — a single best clinical threshold (e.g. on a congestion/shock
   marker), the standard-of-care comparator.
2. **Baseline B** — penalised logistic regression on the same features.
3. **Candidate** — the gradient-boosted tree (XGBoost / LightGBM) used here.

The GBT must beat both baselines by a margin that survives the confidence interval
to justify its complexity.

## Metrics

- **Discrimination:** AUROC and **AUPRC** (AUPRC is the honest primary under low
  prevalence), each with a **patient-level bootstrap 95% CI** (already implemented).
- **Calibration:** a calibration curve **plus** calibration slope/intercept and
  Brier score. If miscalibrated (expected — the current models are), fit and report
  post-hoc **isotonic/Platt** recalibration on the validation fold. Calibration is
  currently plotted but never corrected; this closes that gap.
- **Clinical utility:** decision-curve analysis (net benefit) across the tier
  thresholds, and sensitivity / specificity / PPV / NPV at each operating band.
- **Subgroups:** metrics by sex, age band, and EF phenotype (HFrEF/HFmrEF/HFpEF) to
  surface differential performance.

## External validation

Repeat the frozen model on eICU-CRD without refitting, to estimate transportability
across hospitals. A drop from internal to external AUROC is expected and reported,
not hidden.

## Reporting checklist (TRIPOD+AI)

- [ ] Source, eligibility, and a participant flow diagram (n excluded at each step)
- [ ] Outcome and predictor definitions frozen pre-modelling
- [ ] Sample size / events-per-variable justification
- [ ] Handling of missing data stated (and not silently imputed)
- [ ] Model, hyperparameter search, and selection described reproducibly
- [ ] Discrimination **and** calibration **and** clinical utility reported
- [ ] Confidence intervals on all headline metrics
- [ ] Baseline comparators reported
- [ ] External-validation result reported
- [ ] Code and, where the DUA permits, derivation logic released

## Status

- **Track A (Zigong):** ready to run — the harness (`run_validation.py`) and cohort
  loader (`zigong_cohort.py`) are implemented and tested end-to-end on synthetic
  inputs. The only remaining step is signing the DUA and downloading the data; no
  credentialing is required.
- **Track B (MIMIC-IV):** `build_cohort.py` is a skeleton; blocked on credentialed
  access.

Until one of these tracks has actually been run and its numbers published here, no
result in this repository should be cited as clinical performance.
