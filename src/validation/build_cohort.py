"""Build a heart-failure validation cohort from **full, credentialed** MIMIC-IV.

This is the skeleton referenced by ``VALIDATION.md``. It extracts an HF cohort and
writes it into the same long-format parquet schema the existing feature pipeline
(``src/features/build_features.py``) already consumes — so once real data is
present, feature building, training, and evaluation reuse the exact demo-path code.

It is intentionally not runnable without credentialed data: point ``MIMIC_IV_ROOT``
at a local MIMIC-IV v2.2+ download (the ``hosp/`` and ``icu/`` module CSVs or a
DuckDB/Postgres mirror). No MIMIC data is included in this repository, and the
PhysioNet data-use agreement governs anything produced by this script.

Steps (see ``VALIDATION.md`` for rationale):
  1. Select HF encounters by ICD-9 ``428.*`` / ICD-10 ``I50.*``.
  2. Fix an index time ``t0`` per encounter; use only pre-``t0`` data for features.
  3. Extract vitals and labs into ``{patient_id, day, <measurement columns>}``.
  4. Derive the composite deterioration outcome at the configured horizon.
  5. Write ``vitals.parquet`` / ``labs.parquet`` / ``symptoms.parquet`` /
     ``outcomes.parquet`` under ``$HFTA_DATA_DIR/mimic_full/``.

The ``itemid`` maps below MUST be confirmed against your MIMIC-IV ``d_labitems`` /
``d_items`` version before use — they are the one place a silent error would bias
results, so they are surfaced as explicit configuration rather than buried.
"""

from __future__ import annotations

import os
from pathlib import Path

# ICD codes defining the HF cohort (encounter-level diagnosis).
HF_ICD9_PREFIXES = ("428",)
HF_ICD10_PREFIXES = ("I50",)

# Prediction horizon for the primary composite outcome, in days.
OUTCOME_HORIZON_DAYS = 7

# MIMIC-IV itemids — CONFIRM against your d_items / d_labitems before running.
# (Values below are the commonly used MIMIC-IV itemids; verify per release.)
VITAL_ITEMIDS = {
    "heart_rate": 220045,
    "sbp": 220179,          # non-invasive systolic
    "dbp": 220180,          # non-invasive diastolic
    "resp_rate": 220210,
    "spo2": 220277,
    "weight_kg": 226512,    # admission weight; daily weights are sparser
}
LAB_ITEMIDS = {
    "bnp": 50963,           # NT-proBNP (confirm vs. BNP in your release)
    "creatinine": 50912,
    "potassium": 50971,
}


def _require_mimic_root() -> Path:
    root = os.environ.get("MIMIC_IV_ROOT")
    if not root or not Path(root).exists():
        raise RuntimeError(
            "MIMIC_IV_ROOT is unset or does not exist. This script needs a "
            "credentialed MIMIC-IV download (see VALIDATION.md); no data ships "
            "with this repository."
        )
    return Path(root)


def select_hf_encounters(root: Path):
    """Return HF encounters (subject_id, hadm_id, admittime, ...).

    Reads ``hosp/diagnoses_icd`` and keeps encounters whose icd_code matches an
    HF prefix for its icd_version, joined to ``hosp/admissions`` for timing.
    """
    raise NotImplementedError(
        "Implement against hosp/diagnoses_icd + hosp/admissions once "
        "MIMIC_IV_ROOT is available. Filter by HF_ICD9_PREFIXES / "
        "HF_ICD10_PREFIXES on the matching icd_version."
    )


def extract_timeseries(root: Path, encounters):
    """Extract pre-t0 vitals (icu/chartevents) and labs (hosp/labevents).

    Maps itemids via VITAL_ITEMIDS / LAB_ITEMIDS, bins to per-day rows keyed by
    ``patient_id`` and ``day`` (day 0 = t0 - window), and drops any measurement
    at or after t0 to prevent lookahead.
    """
    raise NotImplementedError(
        "Implement itemid extraction + per-day binning into the long schema "
        "consumed by src/features/build_features.py."
    )


def derive_outcome(root: Path, encounters):
    """Composite deterioration within OUTCOME_HORIZON_DAYS of t0.

    ICU transfer (icu/icustays intime > t0), initiation of IV vasoactive therapy
    (inputevents on the pressor/inotrope itemid set), or death
    (admissions.deathtime), whichever comes first within the horizon.
    """
    raise NotImplementedError(
        "Implement the composite outcome; freeze this definition before modelling."
    )


def main() -> None:
    root = _require_mimic_root()
    out_dir = Path(os.environ.get("HFTA_DATA_DIR", "data")) / "mimic_full"
    out_dir.mkdir(parents=True, exist_ok=True)

    encounters = select_hf_encounters(root)
    vitals, labs, symptoms = extract_timeseries(root, encounters)
    outcomes = derive_outcome(root, encounters)

    for name, frame in {
        "vitals": vitals, "labs": labs,
        "symptoms": symptoms, "outcomes": outcomes,
    }.items():
        frame.to_parquet(out_dir / f"{name}.parquet", index=False)
    print(f"Wrote HF validation cohort to {out_dir}")


if __name__ == "__main__":
    main()
