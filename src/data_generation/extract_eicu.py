"""
eICU Heart Failure Cohort Extraction

Extracts HF patients from eICU Collaborative Research Database (demo or full).
Maps vitals and labs to our standard pipeline schema.

eICU has a different structure than MIMIC-IV:
- vitalPeriodic: high-frequency vitals (every 5 min) with observationoffset from ICU admit
- lab: lab results with labresultoffset from ICU admit
- patient: demographics + outcomes
- diagnosis: text-based diagnosis strings
"""

import numpy as np
import pandas as pd
from pathlib import Path


def load_eicu_tables(base_dir: str) -> dict:
    """Load required eICU tables."""
    base = Path(base_dir)
    tables = {}

    tables["patient"] = pd.read_csv(base / "patient.csv.gz")
    tables["diagnosis"] = pd.read_csv(base / "diagnosis.csv.gz")
    tables["vitalPeriodic"] = pd.read_csv(base / "vitalPeriodic.csv.gz")
    tables["lab"] = pd.read_csv(base / "lab.csv.gz")

    for name, df in tables.items():
        print(f"  Loaded {name}: {df.shape}")

    return tables


def identify_hf_cohort(tables: dict) -> pd.DataFrame:
    """Identify heart failure patients via diagnosis strings."""
    dx = tables["diagnosis"]
    patient = tables["patient"]

    hf_terms = ["heart failure", "chf", "congestive heart"]
    hf_mask = dx["diagnosisstring"].str.lower().str.contains(
        "|".join(hf_terms), na=False
    )
    hf_dx = dx[hf_mask]
    hf_stay_ids = set(hf_dx["patientunitstayid"].unique())

    hf_patients = patient[patient["patientunitstayid"].isin(hf_stay_ids)].copy()

    print(f"  HF cohort: {len(hf_patients)} ICU stays")

    return hf_patients


def extract_vitals(tables: dict, hf_cohort: pd.DataFrame) -> pd.DataFrame:
    """
    Extract vitals from eICU vitalPeriodic table.

    observationoffset = minutes from ICU admission (can be negative for pre-ICU).
    We aggregate to 12-hour windows to match our pipeline.
    """
    vp = tables["vitalPeriodic"]
    stay_ids = set(hf_cohort["patientunitstayid"])

    vitals = vp[vp["patientunitstayid"].isin(stay_ids)].copy()

    # Convert offset to days
    vitals["hours_since_admit"] = vitals["observationoffset"] / 60
    vitals["day"] = (vitals["hours_since_admit"] / 24).astype(int)

    # Filter: first 30 days, positive offset only
    vitals = vitals[(vitals["day"] >= 0) & (vitals["day"] < 30)]

    # Classify morning/evening
    hour_in_day = vitals["hours_since_admit"] % 24
    vitals["measurement"] = hour_in_day.apply(
        lambda h: "morning" if 0 <= h < 12 else "evening"
    )

    vitals["patient_id"] = "eICU-" + vitals["patientunitstayid"].astype(str)

    # Rename eICU columns to our schema
    col_map = {
        "heartrate": "heart_rate",
        "respiration": "respiratory_rate",
        "systemicsystolic": "systolic_bp",
        "systemicdiastolic": "diastolic_bp",
        "sao2": "spo2",
    }
    vitals = vitals.rename(columns=col_map)

    # Aggregate to 12-hour windows (median)
    vital_cols = ["heart_rate", "respiratory_rate", "systolic_bp", "diastolic_bp", "spo2"]
    agg_dict = {col: "median" for col in vital_cols if col in vitals.columns}

    agg = vitals.groupby(["patient_id", "patientunitstayid", "day", "measurement"]).agg(
        agg_dict
    ).reset_index()

    # Add weight from patient table (admission weight)
    weight_map = hf_cohort.set_index("patientunitstayid")["admissionweight"].to_dict()
    agg["weight_kg"] = agg["patientunitstayid"].map(weight_map)

    # Add timestamp placeholder
    agg["timestamp"] = pd.Timestamp("2026-01-01") + pd.to_timedelta(agg["day"], unit="D")

    # Clinical validity filters
    for col, lo, hi in [
        ("heart_rate", 20, 250),
        ("respiratory_rate", 4, 60),
        ("systolic_bp", 40, 300),
        ("diastolic_bp", 20, 200),
        ("spo2", 50, 100),
        ("weight_kg", 30, 300),
    ]:
        if col in agg.columns:
            agg.loc[(agg[col] < lo) | (agg[col] > hi), col] = np.nan

    keep_cols = [
        "patient_id", "timestamp", "day", "measurement",
        "weight_kg", "spo2", "heart_rate", "systolic_bp",
        "diastolic_bp", "respiratory_rate"
    ]
    result = agg[[c for c in keep_cols if c in agg.columns]]

    print(f"  Vitals: {len(result)} records for {result['patient_id'].nunique()} stays")

    return result


def extract_labs(tables: dict, hf_cohort: pd.DataFrame) -> pd.DataFrame:
    """Extract labs from eICU lab table."""
    lab = tables["lab"]
    stay_ids = set(hf_cohort["patientunitstayid"])

    labs = lab[lab["patientunitstayid"].isin(stay_ids)].copy()

    # Map lab names to our schema
    lab_name_map = {
        "BNP": "bnp_pg_ml",
        "creatinine": "creatinine_mg_dl",
        "potassium": "potassium_meq_l",
    }

    labs = labs[labs["labname"].isin(lab_name_map.keys())]
    labs["lab_field"] = labs["labname"].map(lab_name_map)
    labs["labresult"] = pd.to_numeric(labs["labresult"], errors="coerce")
    labs = labs[labs["labresult"].notna()]

    # Convert offset to day
    labs["day"] = (labs["labresultoffset"] / (60 * 24)).astype(int)
    labs = labs[(labs["day"] >= 0) & (labs["day"] < 30)]

    labs["patient_id"] = "eICU-" + labs["patientunitstayid"].astype(str)

    # Pivot
    labs_pivot = labs.pivot_table(
        index=["patient_id", "day"],
        columns="lab_field",
        values="labresult",
        aggfunc="first",
    ).reset_index()

    labs_pivot["timestamp"] = pd.Timestamp("2026-01-01") + pd.to_timedelta(labs_pivot["day"], unit="D")

    # Validity filters
    if "bnp_pg_ml" in labs_pivot.columns:
        labs_pivot.loc[labs_pivot["bnp_pg_ml"] < 0, "bnp_pg_ml"] = np.nan
    if "creatinine_mg_dl" in labs_pivot.columns:
        labs_pivot.loc[labs_pivot["creatinine_mg_dl"] < 0, "creatinine_mg_dl"] = np.nan
        labs_pivot.loc[labs_pivot["creatinine_mg_dl"] > 30, "creatinine_mg_dl"] = np.nan
    if "potassium_meq_l" in labs_pivot.columns:
        labs_pivot.loc[labs_pivot["potassium_meq_l"] < 1.5, "potassium_meq_l"] = np.nan
        labs_pivot.loc[labs_pivot["potassium_meq_l"] > 10, "potassium_meq_l"] = np.nan

    keep_cols = ["patient_id", "timestamp", "day", "bnp_pg_ml", "creatinine_mg_dl", "potassium_meq_l"]
    result = labs_pivot[[c for c in keep_cols if c in labs_pivot.columns]]

    print(f"  Labs: {len(result)} records for {result['patient_id'].nunique()} stays")

    return result


def extract_outcomes(hf_cohort: pd.DataFrame) -> pd.DataFrame:
    """Extract outcomes from eICU patient table."""
    outcomes = []

    for _, row in hf_cohort.iterrows():
        patient_id = f"eICU-{row['patientunitstayid']}"
        died = str(row.get("unitdischargestatus", "")).lower() == "expired"

        # LOS in days
        discharge_offset = row.get("unitdischargeoffset", 0)
        los_days = max(0, discharge_offset) / (60 * 24)

        deterioration_day = None
        if died:
            deterioration_day = max(1, int(los_days - 2))

        # Parse age (eICU has '> 89' for elderly)
        age = row.get("age", "")
        if isinstance(age, str) and ">" in age:
            age = 90
        else:
            try:
                age = int(float(age))
            except (ValueError, TypeError):
                age = 0

        outcomes.append({
            "patient_id": patient_id,
            "deteriorated": int(died),
            "deterioration_day": deterioration_day,
            "age": age,
            "sex": str(row.get("gender", "Unknown")),
            "los_days": round(los_days, 1),
        })

    result = pd.DataFrame(outcomes)
    n_det = result["deteriorated"].sum()
    print(f"  Outcomes: {len(result)} stays, {n_det} ({100*n_det/max(len(result),1):.1f}%) died in unit")

    return result


def generate_dummy_symptoms(vitals: pd.DataFrame) -> pd.DataFrame:
    """Generate empty symptom placeholders (eICU has no patient-reported data)."""
    patient_days = vitals.groupby("patient_id")["day"].max().reset_index()
    records = []
    for _, row in patient_days.iterrows():
        for d in range(int(row["day"]) + 1):
            records.append({
                "patient_id": row["patient_id"],
                "timestamp": pd.Timestamp("2026-01-01") + pd.Timedelta(days=d, hours=9),
                "day": d,
                "reported": False,
                "dyspnea_score": np.nan,
                "orthopnea_pillows": np.nan,
                "ankle_edema": np.nan,
                "exercise_tolerance": np.nan,
                "medication_adherent": np.nan,
            })
    return pd.DataFrame(records)


def extract_eicu_cohort(base_dir: str, output_dir: str):
    """Main extraction pipeline."""
    print("Loading eICU tables...")
    tables = load_eicu_tables(base_dir)

    print("\nIdentifying HF cohort...")
    hf_cohort = identify_hf_cohort(tables)

    print("\nExtracting vitals...")
    vitals = extract_vitals(tables, hf_cohort)

    print("\nExtracting labs...")
    labs = extract_labs(tables, hf_cohort)

    print("\nExtracting outcomes...")
    outcomes = extract_outcomes(hf_cohort)

    print("\nGenerating symptom placeholders...")
    symptoms = generate_dummy_symptoms(vitals)

    # Save
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    vitals.to_parquet(out / "vitals.parquet", index=False)
    labs.to_parquet(out / "labs.parquet", index=False)
    symptoms.to_parquet(out / "symptoms.parquet", index=False)
    outcomes.to_parquet(out / "outcomes.parquet", index=False)

    print(f"\nSaved to {out}/")
    print(f"  Vitals: {len(vitals):,} records")
    print(f"  Labs: {len(labs):,} records")
    print(f"  Symptoms: {len(symptoms):,} records (placeholders)")
    print(f"  Outcomes: {len(outcomes)} stays")

    return {"vitals": vitals, "labs": labs, "symptoms": symptoms, "outcomes": outcomes}


if __name__ == "__main__":
    import os

    base_dir = Path(os.environ.get(
        "HFTA_DATA_DIR", Path(__file__).resolve().parents[2] / "data"))
    EICU_DIR = str(base_dir / "external" / "eicu-demo"
                   / "eicu-collaborative-research-database-demo-2.0.1")
    OUTPUT_DIR = str(base_dir / "eicu")

    extract_eicu_cohort(EICU_DIR, OUTPUT_DIR)
