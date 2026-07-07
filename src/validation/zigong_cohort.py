"""Build an admission-level HF readmission cohort from the **Zigong** dataset.

The Zigong "Hospitalized patients with heart failure" dataset (PhysioNet, *restricted*
access — a login + signed DUA, no credentialing/CITI) provides **2,008 real HF
patients** with baseline (admission-day) variables and follow-up readmission /
mortality outcomes. Unlike MIMIC-IV it is **one row per hospitalization — no time
series**, so this validates an *admission-level readmission model*, not the
sliding-window trajectory approach the demo app uses. See ``VALIDATION.md`` (Track A).

This script reads the downloaded Zigong CSV and writes a features parquet in the
schema ``src/validation/run_validation.py`` consumes (a ``patient_id`` group column,
a binary ``label`` column, and predictor columns). No Zigong data ships with this
repo; point it at your download:

    ZIGONG_CSV=/path/to/zigong.csv python -m src.validation.zigong_cohort

Column names below use the dataset's R-style dotted spelling and match the variable
description file shipped with the download. Confirm them against that file — if one
is missing, the script prints the columns it *did* find so you can adjust the config.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

# --- Column configuration (confirm against the Zigong variable description file) ---
ID_COL = "inpatient.number"

# Outcome to predict. Options in the dataset: 28 days / 3 months / 6 months
# readmission, and the corresponding death columns. 28-day readmission is the
# classic benchmark (cf. the AUROC 0.73-0.81 literature in DESIGN.md).
LABEL_COL = "re.admission.within.28.days"

# Baseline predictors known to be admission-day measurements. This is a curated,
# leakage-safe whitelist; extend it from the variable dictionary as desired. Only
# columns present in the file are used (missing ones are skipped with a warning).
DEFAULT_PREDICTORS = [
    "gender", "ageCat", "BMI", "weight", "height",
    "body.temperature", "pulse", "respiration",
    "systolic.blood.pressure", "diastolic.blood.pressure", "map",
    "type.of.heart.failure", "NYHA.cardiac.function.classification",
    "Killip.grade", "GCS",
    "LVEF", "left.ventricular.end.diastolic.diameter.LV",
    "creatinine.enzymatic.method", "urea", "potassium", "sodium",
    "hemoglobin", "brain.natriuretic.peptide", "white.blood.cell",
]

# Columns that would LEAK the outcome or encode post-admission events — never use
# these as predictors, regardless of what the whitelist picks up.
LEAKAGE_SUBSTRINGS = (
    "re.admission", "readmission", "death", "time.of.death",
    "return.to.emergency", "destination", "discharge", "outcome",
    "days.from.admission", "dischargeday", "los",
)


def _binarize(series: pd.Series) -> pd.Series:
    """Coerce a yes/no, 1/2, or 0/1 outcome column to {0,1}.

    Uses a dtype check that is robust to pandas' object *and* string dtypes.
    """
    if pd.api.types.is_numeric_dtype(series):
        # Numeric codings: treat the dataset's 1 as the positive event, 0/2 as negative.
        return (pd.to_numeric(series, errors="coerce") == 1).astype("float")
    mapping = {"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0, "2": 0}
    return series.astype(str).str.strip().str.lower().map(mapping)


def _find_csv() -> Path:
    explicit = os.environ.get("ZIGONG_CSV")
    if explicit:
        return Path(explicit)
    root = os.environ.get("ZIGONG_ROOT")
    if not root:
        raise RuntimeError(
            "Set ZIGONG_CSV to the downloaded main data file, or ZIGONG_ROOT to its "
            "folder. No Zigong data ships here — download it from PhysioNet after "
            "signing the DUA (see VALIDATION.md)."
        )
    # The medication table is dat_md.csv; the main table is the other/largest CSV.
    candidates = [p for p in Path(root).glob("*.csv") if p.name != "dat_md.csv"]
    if not candidates:
        raise RuntimeError(f"No main CSV found under {root}.")
    return max(candidates, key=lambda p: p.stat().st_size)


def build() -> pd.DataFrame:
    csv = _find_csv()
    df = pd.read_csv(csv)

    if ID_COL not in df.columns:
        raise RuntimeError(
            f"ID column '{ID_COL}' not in {csv.name}. Columns found: {list(df.columns)}"
        )
    if LABEL_COL not in df.columns:
        raise RuntimeError(
            f"Label column '{LABEL_COL}' not in {csv.name}. Adjust LABEL_COL. "
            f"Columns found: {list(df.columns)}"
        )

    present = [c for c in DEFAULT_PREDICTORS if c in df.columns]
    missing = sorted(set(DEFAULT_PREDICTORS) - set(present))
    if missing:
        print(f"[warn] predictors not found, skipped: {missing}")

    # Leakage guard: drop any selected predictor whose name looks outcome-derived.
    safe = [c for c in present
            if not any(sub in c.lower() for sub in LEAKAGE_SUBSTRINGS)]

    out = pd.DataFrame({"patient_id": df[ID_COL].values,
                        "label": _binarize(df[LABEL_COL]).values})
    # One-hot low-cardinality categoricals; coerce the rest to numeric. Use a
    # dtype-agnostic test (is_numeric_dtype) so pandas' string dtype is handled.
    feats = df[safe].copy()
    cat = [c for c in feats.columns
           if not pd.api.types.is_numeric_dtype(feats[c])
           and feats[c].nunique(dropna=True) <= 12]
    num = [c for c in feats.columns if c not in cat]
    pieces = []
    if num:
        pieces.append(feats[num].apply(pd.to_numeric, errors="coerce"))
    if cat:
        pieces.append(pd.get_dummies(feats[cat].astype("string"), dummy_na=True))
    feats = pd.concat(pieces, axis=1) if pieces else pd.DataFrame(index=feats.index)
    feats = feats.dropna(axis=1, how="all")  # drop non-numeric cols that slipped into num

    out = pd.concat([out, feats.reset_index(drop=True)], axis=1)
    out = out.dropna(subset=["label"])
    out["label"] = out["label"].astype(int)
    return out


def main() -> None:
    out = build()
    dest = Path(os.environ.get("HFTA_DATA_DIR", "data")) / "processed" / "features_zigong.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dest, index=False)
    print(f"Wrote {len(out)} admissions x {out.shape[1]-2} predictors to {dest}")
    print(f"Positive rate ({LABEL_COL}): {out['label'].mean():.1%}")
    print(f"Next: python -m src.validation.run_validation --features {dest}")


if __name__ == "__main__":
    main()
