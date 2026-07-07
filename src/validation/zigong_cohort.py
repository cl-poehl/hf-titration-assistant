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

# Outcome to predict (override with ZIGONG_LABEL). 6-month readmission is the
# best-powered choice here (773 events / 2,008, ~18 events per variable); 28-day
# readmission has only 140 events and the mortality columns 11-57, too few to model.
LABEL_COL = os.environ.get("ZIGONG_LABEL", "re.admission.within.6.months")

# Baseline predictors known to be admission-day measurements. This is a curated,
# leakage-safe whitelist; extend it from the variable dictionary as desired. Only
# columns present in the file are used (missing ones are skipped with a warning).
# Deliberately kept small: with only ~140 readmission events, a ~25-predictor set
# keeps events-per-variable defensible (TRIPOD+AI / PROBAST), whereas all ~150
# columns would be ~1 EPV and overfit.
DEFAULT_PREDICTORS = [
    "gender", "ageCat", "BMI", "weight", "height",
    "body.temperature", "pulse", "respiration",
    "systolic.blood.pressure", "diastolic.blood.pressure", "map",
    "type.of.heart.failure", "NYHA.cardiac.function.classification",
    "Killip.grade", "GCS", "CCI.score",
    "myocardial.infarction", "diabetes",
    "moderate.to.severe.chronic.kidney.disease",
    "LVEF", "left.ventricular.end.diastolic.diameter.LV",
    "creatinine.enzymatic.method", "urea", "potassium", "sodium",
    "hemoglobin", "brain.natriuretic.peptide", "white.blood.cell",
]

# Columns that would LEAK the outcome or encode post-admission events — never use
# these as predictors, regardless of what the whitelist picks up.
LEAKAGE_SUBSTRINGS = (
    "re.admission", "readmission", "death", "time.of.death",
    "return.to.emergency", "emergency", "destination", "discharge",
    "outcome", "days.from.admission", "dischargeday", "los",
)


# Drug-name substrings → therapeutic class, for medication features derived from
# dat_md.csv. The four GDMT pillars mirror the repo's GDMT engine (predictor.py).
DRUG_CLASSES = {
    "beta_blocker": ["metoprolol", "bisoprolol", "carvedilol", "nebivolol", "atenolol"],
    "raasi": ["sacubitril", "enalapril", "captopril", "benazepril", "perindopril",
              "ramipril", "lisinopril", "fosinopril", "imidapril",
              "valsartan", "losartan", "irbesartan", "candesartan", "telmisartan", "olmesartan"],
    "mra": ["spironolactone", "eplerenone"],
    "sglt2i": ["dapagliflozin", "empagliflozin", "canagliflozin"],
    "loop_diuretic": ["furosemide", "torasemide", "torsemide", "bumetanide"],
    "digoxin": ["digoxin", "deslanoside", "digitalis"],
    "inotrope": ["milrinone", "dobutamine", "dopamine", "levosimendan"],
}
GDMT_PILLARS = ("beta_blocker", "raasi", "mra", "sglt2i")


def medication_features(md_csv: Path) -> pd.DataFrame:
    """Per-admission medication features from dat_md.csv (drug names → class flags).

    In-hospital medications are observed during the index stay, before the
    post-discharge readmission window, so they are valid predictors. Returns a
    frame indexed by patient_id with a polypharmacy count, per-class flags, and a
    GDMT-pillar count (0-4) mirroring the repo's four-pillar engine.
    """
    md = pd.read_csv(md_csv)
    id_col = next(c for c in md.columns if "inpatient" in c.lower())
    name = md["Drug_name"].astype(str).str.lower()
    md = md.assign(_name=name)

    rows = {"patient_id": [], "n_medications": []}
    for cls in DRUG_CLASSES:
        rows[f"on_{cls}"] = []
    for pid, grp in md.groupby(id_col):
        rows["patient_id"].append(pid)
        rows["n_medications"].append(grp["_name"].nunique())
        for cls, kws in DRUG_CLASSES.items():
            hit = grp["_name"].str.contains("|".join(kws), regex=True).any()
            rows[f"on_{cls}"].append(int(hit))
    out = pd.DataFrame(rows)
    out["n_gdmt_pillars"] = out[[f"on_{p}" for p in GDMT_PILLARS]].sum(axis=1)
    return out


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
    feats = feats.loc[:, feats.nunique(dropna=True) > 1]  # drop constant / empty-dummy columns

    out = pd.concat([out, feats.reset_index(drop=True)], axis=1)

    # Merge medication features if dat_md.csv is available alongside the main CSV.
    md_csv = Path(os.environ.get("ZIGONG_MD_CSV", csv.parent / "dat_md.csv"))
    if md_csv.exists():
        med = medication_features(md_csv)
        out = out.merge(med, on="patient_id", how="left")
        med_cols = [c for c in med.columns if c != "patient_id"]
        out[med_cols] = out[med_cols].fillna(0)  # no records == not on that drug
        print(f"[info] merged {len(med_cols)} medication features from {md_csv.name}")

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
