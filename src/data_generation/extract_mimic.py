"""
MIMIC-IV Heart Failure Cohort Extraction

Extracts heart failure patients from MIMIC-IV (demo or full) and maps
vitals, labs, and outcomes to our standard pipeline schema.

Key design decisions:
- Uses ICU chartevents for vitals (highest temporal resolution)
- Aggregates to 12-hour windows (morning/evening) to match RPM cadence
- Outcome = in-hospital mortality OR readmission within 30 days
- Same output schema as synthetic generator → feeds directly into build_features.py
"""

import numpy as np
import pandas as pd
from pathlib import Path


# MIMIC-IV item IDs for vital signs
VITAL_ITEMIDS = {
    # Heart rate
    220045: "heart_rate",
    # Respiratory rate
    220210: "respiratory_rate",
    # Non-invasive BP (preferred for HaH analogy)
    220179: "systolic_bp",
    220180: "diastolic_bp",
    # Arterial BP (fallback)
    220050: "systolic_bp",
    220051: "diastolic_bp",
    # SpO2
    220277: "spo2",
    # Weight
    224639: "weight_kg",     # Daily Weight
    226512: "weight_kg",     # Admission Weight (Kg)
}

# MIMIC-IV lab item IDs
LAB_ITEMIDS = {
    50963: "bnp_pg_ml",       # NTproBNP
    50912: "creatinine_mg_dl",  # Creatinine (serum)
    50822: "potassium_meq_l",   # Potassium, Whole Blood
    50971: "potassium_meq_l",   # Potassium (serum, fallback)
}

# Heart failure ICD codes
HF_ICD9_PREFIX = "428"
HF_ICD10_PREFIX = "I50"


def load_mimic_tables(base_dir: str) -> dict:
    """Load required MIMIC-IV tables."""
    base = Path(base_dir)
    tables = {}

    hosp = base / "hosp"
    icu = base / "icu"

    tables["patients"] = pd.read_csv(hosp / "patients.csv.gz")
    tables["admissions"] = pd.read_csv(hosp / "admissions.csv.gz")
    tables["diagnoses"] = pd.read_csv(hosp / "diagnoses_icd.csv.gz")
    tables["labevents"] = pd.read_csv(hosp / "labevents.csv.gz")
    tables["icustays"] = pd.read_csv(icu / "icustays.csv.gz")
    tables["chartevents"] = pd.read_csv(icu / "chartevents.csv.gz")

    for name, df in tables.items():
        print(f"  Loaded {name}: {df.shape}")

    return tables


def identify_hf_cohort(tables: dict) -> pd.DataFrame:
    """Identify heart failure patients and their admissions."""
    dx = tables["diagnoses"]

    hf_dx = dx[
        dx["icd_code"].str.startswith(HF_ICD9_PREFIX) |
        dx["icd_code"].str.startswith(HF_ICD10_PREFIX)
    ]

    hf_admissions = hf_dx[["subject_id", "hadm_id"]].drop_duplicates()

    # Join with admissions for timing
    adm = tables["admissions"]
    hf_adm = hf_admissions.merge(adm, on=["subject_id", "hadm_id"])

    # Join with patients for demographics
    patients = tables["patients"]
    hf_cohort = hf_adm.merge(patients, on="subject_id")

    # Join with ICU stays
    icu = tables["icustays"]
    hf_cohort = hf_cohort.merge(icu, on=["subject_id", "hadm_id"], how="inner")

    print(f"  HF cohort: {hf_cohort['subject_id'].nunique()} patients, "
          f"{len(hf_cohort)} ICU stays")

    return hf_cohort


def extract_vitals(tables: dict, hf_cohort: pd.DataFrame) -> pd.DataFrame:
    """Extract and aggregate vital signs for HF cohort."""
    chart = tables["chartevents"]
    stay_ids = set(hf_cohort["stay_id"])

    # Filter to HF stays and vital sign items
    vital_items = set(VITAL_ITEMIDS.keys())
    vitals = chart[
        (chart["stay_id"].isin(stay_ids)) &
        (chart["itemid"].isin(vital_items)) &
        (chart["valuenum"].notna())
    ].copy()

    # Map item IDs to standard names
    vitals["vital_name"] = vitals["itemid"].map(VITAL_ITEMIDS)
    vitals["charttime"] = pd.to_datetime(vitals["charttime"])

    # Weight in lbs needs conversion (item 226531)
    # Items 224639 and 226512 are already in kg

    # Pivot: one row per (stay_id, charttime), columns = vital names
    # Take the first non-null value for duplicate vital types at same time
    vitals_pivot = vitals.pivot_table(
        index=["subject_id", "stay_id", "charttime"],
        columns="vital_name",
        values="valuenum",
        aggfunc="first",
    ).reset_index()

    # Join with ICU stay info to get admission time (for day calculation)
    stay_info = hf_cohort[["stay_id", "intime"]].drop_duplicates()
    stay_info["intime"] = pd.to_datetime(stay_info["intime"])
    vitals_pivot = vitals_pivot.merge(stay_info, on="stay_id")

    # Calculate day relative to ICU admission
    vitals_pivot["hours_since_admit"] = (
        (vitals_pivot["charttime"] - vitals_pivot["intime"]).dt.total_seconds() / 3600
    )
    vitals_pivot["day"] = (vitals_pivot["hours_since_admit"] / 24).astype(int)

    # Filter to first 30 days (matching our pipeline)
    vitals_pivot = vitals_pivot[(vitals_pivot["day"] >= 0) & (vitals_pivot["day"] < 30)]

    # Classify as morning/evening based on hour
    vitals_pivot["hour"] = vitals_pivot["charttime"].dt.hour
    vitals_pivot["measurement"] = vitals_pivot["hour"].apply(
        lambda h: "morning" if 4 <= h < 16 else "evening"
    )

    # Aggregate to 12-hour windows (median to reduce outliers)
    agg_vitals = vitals_pivot.groupby(
        ["subject_id", "stay_id", "day", "measurement"]
    ).agg({
        "charttime": "first",
        "heart_rate": "median",
        "respiratory_rate": "median",
        "systolic_bp": "median",
        "diastolic_bp": "median",
        "spo2": "median",
        "weight_kg": "last",  # most recent weight in window
    }).reset_index()

    # Forward-fill weight within each stay
    agg_vitals = agg_vitals.sort_values(["stay_id", "day", "measurement"])
    agg_vitals["weight_kg"] = agg_vitals.groupby("stay_id")["weight_kg"].ffill()

    # Create patient_id from subject_id + stay_id for uniqueness
    agg_vitals["patient_id"] = "MIMIC-" + agg_vitals["stay_id"].astype(str)

    # Rename to match our schema
    result = agg_vitals.rename(columns={"charttime": "timestamp"})

    # Ensure numeric types
    for col in ["heart_rate", "respiratory_rate", "systolic_bp", "diastolic_bp", "spo2", "weight_kg"]:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    # Apply clinical validity filters
    result.loc[result["heart_rate"] < 20, "heart_rate"] = np.nan
    result.loc[result["heart_rate"] > 250, "heart_rate"] = np.nan
    result.loc[result["spo2"] < 50, "spo2"] = np.nan
    result.loc[result["spo2"] > 100, "spo2"] = 100
    result.loc[result["systolic_bp"] < 40, "systolic_bp"] = np.nan
    result.loc[result["systolic_bp"] > 300, "systolic_bp"] = np.nan
    result.loc[result["weight_kg"] < 30, "weight_kg"] = np.nan
    result.loc[result["weight_kg"] > 300, "weight_kg"] = np.nan

    keep_cols = [
        "patient_id", "timestamp", "day", "measurement",
        "weight_kg", "spo2", "heart_rate", "systolic_bp",
        "diastolic_bp", "respiratory_rate"
    ]

    result = result[[c for c in keep_cols if c in result.columns]]

    print(f"  Vitals: {len(result)} records for {result['patient_id'].nunique()} stays")

    return result


def extract_labs(tables: dict, hf_cohort: pd.DataFrame) -> pd.DataFrame:
    """Extract lab values for HF cohort."""
    labevents = tables["labevents"]
    hadm_ids = set(hf_cohort["hadm_id"])
    lab_items = set(LAB_ITEMIDS.keys())

    labs = labevents[
        (labevents["hadm_id"].isin(hadm_ids)) &
        (labevents["itemid"].isin(lab_items)) &
        (labevents["valuenum"].notna())
    ].copy()

    labs["lab_name"] = labs["itemid"].map(LAB_ITEMIDS)
    labs["charttime"] = pd.to_datetime(labs["charttime"])

    # Join with stay info for day calculation
    # Map hadm_id to stay_id and intime
    stay_map = hf_cohort[["hadm_id", "stay_id", "intime"]].drop_duplicates()
    stay_map["intime"] = pd.to_datetime(stay_map["intime"])
    labs = labs.merge(stay_map, on="hadm_id")

    labs["hours_since_admit"] = (
        (labs["charttime"] - labs["intime"]).dt.total_seconds() / 3600
    )
    labs["day"] = (labs["hours_since_admit"] / 24).astype(int)
    labs = labs[(labs["day"] >= 0) & (labs["day"] < 30)]

    labs["patient_id"] = "MIMIC-" + labs["stay_id"].astype(str)

    # Pivot to wide format, one row per (patient, day)
    # Take first value per day if multiple
    labs_pivot = labs.pivot_table(
        index=["patient_id", "day"],
        columns="lab_name",
        values="valuenum",
        aggfunc="first",
    ).reset_index()

    # Add timestamp (use day midpoint)
    labs_pivot["timestamp"] = pd.Timestamp("2026-01-01") + pd.to_timedelta(labs_pivot["day"], unit="D")

    # Clinical validity filters
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
    """
    Extract outcome labels.

    For the demo dataset, we use in-hospital mortality as the outcome
    since we can't track 30-day readmissions across a 100-patient sample.
    For the full MIMIC-IV, we'll add readmission tracking.
    """
    outcomes = []

    for _, row in hf_cohort.iterrows():
        patient_id = f"MIMIC-{row['stay_id']}"
        died = row.get("hospital_expire_flag", 0) == 1

        # Calculate LOS to estimate deterioration timing
        intime = pd.to_datetime(row["intime"])
        outtime = pd.to_datetime(row["outtime"])
        los_days = (outtime - intime).total_seconds() / 86400

        # If patient died, estimate deterioration at 2 days before death/discharge
        deterioration_day = None
        if died:
            deterioration_day = max(1, int(los_days - 2))

        outcomes.append({
            "patient_id": patient_id,
            "deteriorated": int(died),
            "deterioration_day": deterioration_day,
            "age": row.get("anchor_age", 0),
            "sex": row.get("gender", "U"),
            "los_days": round(los_days, 1),
        })

    result = pd.DataFrame(outcomes)
    n_det = result["deteriorated"].sum()
    print(f"  Outcomes: {len(result)} stays, {n_det} ({100*n_det/max(len(result),1):.1f}%) deteriorated")

    return result


def generate_dummy_symptoms(vitals: pd.DataFrame) -> pd.DataFrame:
    """
    MIMIC-IV doesn't have patient-reported symptom scores.
    Generate empty symptom DataFrame with correct schema so the
    feature pipeline handles it gracefully.
    """
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


def extract_mimic_cohort(base_dir: str, output_dir: str):
    """Main extraction pipeline."""
    print("Loading MIMIC-IV tables...")
    tables = load_mimic_tables(base_dir)

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
    MIMIC_DIR = str(base_dir / "external" / "mimic-iv-demo"
                    / "mimic-iv-clinical-database-demo-2.2")
    OUTPUT_DIR = str(base_dir / "mimic")

    extract_mimic_cohort(MIMIC_DIR, OUTPUT_DIR)
