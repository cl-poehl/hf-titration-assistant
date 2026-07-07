# Architecture

A file-by-file map of the codebase — commands, module responsibilities, data
layout, and gotchas. For product/clinical rationale see [`DESIGN.md`](DESIGN.md);
for scope and limitations see [`README.md`](README.md).

## Project Overview

HF Titration Assistant is a **research prototype** (not a medical device) that scores deterioration risk in heart-failure patients receiving acute care at home. It has two parts: a **Python/FastAPI backend** (ML pipeline + prediction API) and a **React/TypeScript dashboard** (clinician-facing UI). See `README.md` for the honest scope and limitations.

## Commands

### Backend (run from project root)

```bash
pip install -r requirements.txt
uvicorn src.api.main:app --reload --port 8000
```

### Frontend

```bash
cd dashboard && npm install
cd dashboard && npm run dev        # dev server at localhost:5173
cd dashboard && npm run build      # tsc + vite build → dist/
```

### ML Pipeline (run in order)

```bash
# 1. Generate synthetic data (or extract real clinical data)
python src/data_generation/generate_hf_patients.py

# 2. Build feature matrix
python src/features/build_features.py            # synthetic
python src/features/build_features.py --combined  # real clinical data

# 3. Train models
python src/models/train.py          # synthetic → models/best_model.pkl
python src/models/train.py --real   # real data → models/best_model_real.pkl

# 4. End-to-end validation (no server needed)
python scripts/test_api.py
```

### Manual Testing

```bash
curl http://localhost:8000/health
curl http://localhost:8000/patients
cd dashboard && npx tsc --noEmit   # frontend type check
```

No automated test suites, linters, or formatters are configured.

## Architecture

### Backend (`src/`)

- **`src/api/main.py`** — FastAPI app with endpoints: `/health`, `/patients`, `/patients/{id}`, `/patients/{id}/gdmt`, `/predict`. Loads model on startup, preferring `best_model_real.pkl` over `best_model.pkl`.
- **`src/api/predictor.py`** — `HFPredictor` class wrapping the trained model + SHAP TreeExplainer. Every prediction returns top-5 SHAP factors and a suggested clinical action.
- **`src/api/patient_registry.py`** — Loads parquet data from `data/combined/`, builds patient panel with live risk scores. Patient display names are derived from SHA-256 hash of patient ID (deterministic, no PHI).
- **`src/features/build_features.py`** — Sliding-window feature extraction (98 features from vitals/labs/symptoms). Uses `--combined` flag for real clinical data with adaptive 3-day windows (vs 7-day default for synthetic).
- **`src/models/train.py`** — XGBoost + LightGBM training with patient-level stratified splits (prevents data leakage across temporal windows of same patient).
- **`src/data_generation/`** — Synthetic patient generator + MIMIC-IV/eICU ETL pipelines.
- **`src/validation/`** — Real-data validation harness (see [`VALIDATION.md`](VALIDATION.md)). `run_validation.py` is a working, data-source-agnostic evaluator (three-way patient-level split, isotonic calibration, logistic baseline, bootstrap CIs); `zigong_cohort.py` builds an admission-level readmission cohort from the restricted-access **Zigong** dataset (Track A, runnable now); `build_cohort.py` is a skeleton for a trajectory cohort from credentialed **MIMIC-IV** (Track B). Ships no data and is not part of the demo pipeline.

### Frontend (`dashboard/`)

- React 19, TypeScript 5 (strict), Vite 7, Tailwind CSS v4, Recharts 3
- **Vite proxy**: `/api/*` requests are rewritten to `http://localhost:8000/*` (see `vite.config.ts`)
- **Demo-mode fallback**: `usePatients()` hook tries the live API first, falls back to `src/data/mockPatients.ts` with a yellow banner if the backend is unavailable
- Key components: `PatientList` (sidebar), `PatientDetail` (main panel with vitals charts), `ExplanationPanel` (SHAP bar charts + suggested actions), `RiskGauge` (circular risk score)

### Data

All data is file-based (parquet on disk, no database). Key directories:
- `data/raw/` — synthetic patient data
- `data/external/` — MIMIC-IV and eICU demo datasets
- `data/combined/` — merged real clinical data (what the API serves)
- `data/processed/` — ML feature matrices
- `models/` — trained model pickles, feature column lists, evaluation results

### Paths

Pipeline scripts default to locations relative to the project root, so they run
on any clone. Override the base data/model directories with the
`HFTA_DATA_DIR` and `HFTA_MODEL_DIR` environment variables (e.g. to
point at an HPC scratch workspace).

## Risk Tiers

The model classifies patients into tiers used throughout the API and dashboard.
Boundaries are defined once in `RISK_TIER_THRESHOLDS` (`src/api/predictor.py`):
- **Low**: <15% probability
- **Medium**: 15–59%
- **High**: 60–79%
- **Critical**: >=80%
