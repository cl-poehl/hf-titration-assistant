"""
Feature Extraction Pipeline for Heart Failure Deterioration Prediction

Transforms raw time-series vitals, labs, and symptoms into ML-ready features.
Uses a sliding window approach: for each patient, we generate features at
multiple prediction points (e.g., day 7, 10, 14, ...) to maximize training data.

Feature categories:
1. Current values (most recent measurement)
2. Trends (linear slope over last N days)
3. Variability (std dev over last N days)
4. Deviation from patient baseline
5. Rate of change (absolute and relative)
6. Cross-feature interactions
7. Symptom burden scores
8. Missing data indicators
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats


def load_raw_data(data_dir: str) -> dict:
    """Load raw parquet files."""
    data_dir = Path(data_dir)
    return {
        "vitals": pd.read_parquet(data_dir / "vitals.parquet"),
        "labs": pd.read_parquet(data_dir / "labs.parquet"),
        "symptoms": pd.read_parquet(data_dir / "symptoms.parquet"),
        "outcomes": pd.read_parquet(data_dir / "outcomes.parquet"),
    }


def compute_vital_features(vitals_patient: pd.DataFrame, up_to_day: int,
                           window_days: int = 3) -> dict:
    """
    Extract features from vitals for a single patient up to a given day.

    We look at data from [0, up_to_day] and compute features using
    the most recent `window_days` of data.
    """
    df = vitals_patient[vitals_patient["day"] <= up_to_day].copy()
    if len(df) == 0:
        return {}

    recent = df[df["day"] > up_to_day - window_days]
    early = df[df["day"] <= 3]  # first 3 days = baseline

    features = {}
    vital_cols = ["weight_kg", "spo2", "heart_rate", "systolic_bp", "diastolic_bp", "respiratory_rate"]

    for col in vital_cols:
        if col not in df.columns:
            continue

        prefix = col.replace("_", "")

        # Drop NaN for this column
        valid_df = df[col].dropna()
        valid_recent = recent[col].dropna() if len(recent) > 0 else pd.Series(dtype=float)
        valid_early = early[col].dropna() if len(early) > 0 else pd.Series(dtype=float)

        if len(valid_df) == 0:
            # No valid data for this vital
            features[f"{prefix}_current"] = 0.0
            features[f"{prefix}_mean_3d"] = 0.0
            features[f"{prefix}_std_3d"] = 0.0
            features[f"{prefix}_min_3d"] = 0.0
            features[f"{prefix}_max_3d"] = 0.0
            features[f"{prefix}_trend_3d"] = 0.0
            features[f"{prefix}_baseline_dev"] = 0.0
            features[f"{prefix}_baseline_pct_change"] = 0.0
            continue

        # Current value (most recent non-NaN)
        features[f"{prefix}_current"] = valid_df.iloc[-1]

        # Recent window stats
        if len(valid_recent) >= 2:
            features[f"{prefix}_mean_3d"] = valid_recent.mean()
            features[f"{prefix}_std_3d"] = valid_recent.std()
            features[f"{prefix}_min_3d"] = valid_recent.min()
            features[f"{prefix}_max_3d"] = valid_recent.max()

            # Trend: linear regression slope over recent window
            x = np.arange(len(valid_recent))
            slope, _, _, _, _ = stats.linregress(x, valid_recent.values)
            features[f"{prefix}_trend_3d"] = slope
        else:
            features[f"{prefix}_mean_3d"] = valid_df.iloc[-1]
            features[f"{prefix}_std_3d"] = 0.0
            features[f"{prefix}_min_3d"] = valid_df.iloc[-1]
            features[f"{prefix}_max_3d"] = valid_df.iloc[-1]
            features[f"{prefix}_trend_3d"] = 0.0

        # Deviation from baseline
        if len(valid_early) > 0:
            baseline_mean = valid_early.mean()
            features[f"{prefix}_baseline_dev"] = features[f"{prefix}_current"] - baseline_mean
            if baseline_mean != 0:
                features[f"{prefix}_baseline_pct_change"] = (
                    (features[f"{prefix}_current"] - baseline_mean) / abs(baseline_mean)
                )
            else:
                features[f"{prefix}_baseline_pct_change"] = 0.0
        else:
            features[f"{prefix}_baseline_dev"] = 0.0
            features[f"{prefix}_baseline_pct_change"] = 0.0

    # Cross-feature interactions (safe defaults if vitals missing)
    sbp = features.get("systolicbp_current", 0)
    dbp = features.get("diastolicbp_current", 0)
    hr = features.get("heartrate_current", 0)
    features["pulse_pressure"] = sbp - dbp
    features["rate_pressure_product"] = hr * sbp
    features["shock_index"] = hr / max(sbp, 1)

    # Weight change features (critical for HF)
    if "weight_kg" in df.columns:
        valid_early_wt = early["weight_kg"].dropna() if len(early) > 0 else pd.Series(dtype=float)
        valid_recent_wt = recent["weight_kg"].dropna() if len(recent) > 0 else pd.Series(dtype=float)

        if len(valid_early_wt) > 0 and len(valid_recent_wt) > 0:
            baseline_weight = valid_early_wt.mean()
            current_weight = valid_recent_wt.mean()
            features["weight_change_total_kg"] = current_weight - baseline_weight
            features["weight_change_total_pct"] = (
                (current_weight - baseline_weight) / baseline_weight if baseline_weight != 0 else 0.0
            )

            if len(valid_recent_wt) >= 2:
                x = np.arange(len(valid_recent_wt))
                slope, _, _, _, _ = stats.linregress(x, valid_recent_wt.values)
                features["weight_velocity_kg_per_measurement"] = slope
            else:
                features["weight_velocity_kg_per_measurement"] = 0.0
        else:
            features["weight_change_total_kg"] = 0.0
            features["weight_change_total_pct"] = 0.0
            features["weight_velocity_kg_per_measurement"] = 0.0
    else:
        features["weight_change_total_kg"] = 0.0
        features["weight_change_total_pct"] = 0.0
        features["weight_velocity_kg_per_measurement"] = 0.0

    return features


def compute_lab_features(labs_patient: pd.DataFrame, up_to_day: int,
                         age: int = 65, sex_male: int = 1) -> dict:
    """Extract features from lab values. Optionally computes eGFR if age/sex provided."""
    df = labs_patient[labs_patient["day"] <= up_to_day].copy()
    if len(df) == 0:
        return {}

    features = {}
    lab_cols = {
        "bnp_pg_ml": "bnp",
        "creatinine_mg_dl": "creat",
        "potassium_meq_l": "k",
    }

    for col, prefix in lab_cols.items():
        if col not in df.columns:
            features[f"{prefix}_current"] = 0.0
            features[f"{prefix}_trend"] = 0.0
            features[f"{prefix}_change"] = 0.0
            features[f"{prefix}_pct_change"] = 0.0
            continue

        valid = df[col].dropna()
        if len(valid) == 0:
            features[f"{prefix}_current"] = 0.0
            features[f"{prefix}_trend"] = 0.0
            features[f"{prefix}_change"] = 0.0
            features[f"{prefix}_pct_change"] = 0.0
            continue

        features[f"{prefix}_current"] = valid.iloc[-1]

        if len(valid) >= 2:
            x = np.arange(len(valid))
            slope, _, _, _, _ = stats.linregress(x, valid.values)
            features[f"{prefix}_trend"] = slope

            features[f"{prefix}_change"] = valid.iloc[-1] - valid.iloc[0]
            if valid.iloc[0] != 0:
                features[f"{prefix}_pct_change"] = (
                    (valid.iloc[-1] - valid.iloc[0]) / abs(valid.iloc[0])
                )
            else:
                features[f"{prefix}_pct_change"] = 0.0
        else:
            features[f"{prefix}_trend"] = 0.0
            features[f"{prefix}_change"] = 0.0
            features[f"{prefix}_pct_change"] = 0.0

    # BNP categories (clinically meaningful thresholds)
    bnp = features.get("bnp_current", 0)
    features["bnp_above_300"] = int(bnp > 300)
    features["bnp_above_600"] = int(bnp > 600)
    features["bnp_above_900"] = int(bnp > 900)

    # Creatinine > 1.5 indicates renal impairment
    features["creat_elevated"] = int(features.get("creat_current", 0) > 1.5)

    # Potassium abnormalities
    k_val = features.get("k_current", 4.0)
    features["k_abnormal"] = int(k_val < 3.5 or k_val > 5.0)

    # eGFR (CKD-EPI 2021, race-free)
    creat_val = features.get("creat_current", 1.0)
    features["egfr"] = compute_egfr(creat_val, age, sex_male)

    return features


def compute_egfr(creat: float, age: int, sex_male: int) -> float:
    """CKD-EPI 2021 race-free eGFR calculation."""
    if creat <= 0:
        return 90.0
    if sex_male:
        kappa, alpha, female_mult = 0.9, -0.302, 1.0
    else:
        kappa, alpha, female_mult = 0.7, -0.241, 1.012
    ratio = creat / kappa
    egfr = 142 * (min(ratio, 1.0) ** alpha) * (max(ratio, 1.0) ** -1.200) * (0.9938 ** age) * female_mult
    return round(max(egfr, 0), 1)


def compute_medication_features(medications_patient: pd.DataFrame, up_to_day: int) -> dict:
    """
    Extract per-drug-class features from medication/titration data.

    Features per drug class: is_prescribed, dose_pct_of_target,
    days_since_last_titration, n_prior_titrations, all_prior_titrations_tolerated.
    Aggregate: gdmt_n_classes, gdmt_avg_dose_pct.
    """
    features = {}

    # Baseline medication state (non-titration-event rows)
    baseline = medications_patient[
        medications_patient.get("status", pd.Series(dtype=str)) != "titration_event"
    ] if "status" in medications_patient.columns else medications_patient

    # Titration events up to prediction day
    titration_events = pd.DataFrame()
    if "titration_day" in medications_patient.columns:
        titration_events = medications_patient[
            (medications_patient["status"] == "titration_event") &
            (medications_patient["titration_day"] <= up_to_day)
        ]

    drug_classes = ["raasi", "beta_blocker", "mra", "sglt2i"]
    dose_pcts = []
    n_indicated = 0

    for dc in drug_classes:
        dc_baseline = baseline[baseline["drug_class"] == dc] if "drug_class" in baseline.columns else pd.DataFrame()
        dc_events = titration_events[titration_events["drug_class"] == dc] if len(titration_events) > 0 else pd.DataFrame()

        if len(dc_baseline) == 0:
            features[f"{dc}_is_prescribed"] = 0
            features[f"{dc}_dose_pct_of_target"] = 0.0
            features[f"{dc}_days_since_last_titration"] = -1
            features[f"{dc}_n_prior_titrations"] = 0
            features[f"{dc}_all_titrations_tolerated"] = 1
            continue

        row = dc_baseline.iloc[0]
        is_indicated = row.get("status", "") != "not_indicated"
        features[f"{dc}_is_prescribed"] = int(is_indicated)

        target = row.get("target_dose_mg", 1)
        current = row.get("current_dose_mg", 0)

        # Update current dose from titration events
        if len(dc_events) > 0 and "tolerated" in dc_events.columns:
            tolerated_events = dc_events[dc_events["tolerated"] == True]
            if len(tolerated_events) > 0:
                current = tolerated_events.iloc[-1].get("new_dose", current)

        dose_pct = min(current / target, 1.0) if target > 0 else 0.0
        features[f"{dc}_dose_pct_of_target"] = round(dose_pct, 3)

        if is_indicated:
            n_indicated += 1
            dose_pcts.append(dose_pct)

        # Titration history
        if len(dc_events) > 0:
            features[f"{dc}_n_prior_titrations"] = len(dc_events)
            last_day = dc_events["titration_day"].max()
            features[f"{dc}_days_since_last_titration"] = up_to_day - last_day
            if "tolerated" in dc_events.columns:
                features[f"{dc}_all_titrations_tolerated"] = int(dc_events["tolerated"].all())
            else:
                features[f"{dc}_all_titrations_tolerated"] = 1
        else:
            features[f"{dc}_n_prior_titrations"] = 0
            features[f"{dc}_days_since_last_titration"] = -1
            features[f"{dc}_all_titrations_tolerated"] = 1

    # Aggregate
    features["gdmt_n_classes"] = n_indicated
    features["gdmt_avg_dose_pct"] = round(np.mean(dose_pcts), 3) if dose_pcts else 0.0

    return features


def compute_symptom_features(symptoms_patient: pd.DataFrame, up_to_day: int,
                             window_days: int = 3) -> dict:
    """Extract features from patient-reported symptoms."""
    df = symptoms_patient[symptoms_patient["day"] <= up_to_day].copy()
    if len(df) == 0:
        return {}

    recent = df[df["day"] > up_to_day - window_days]
    features = {}

    # Reporting rate itself is a signal (sicker patients stop reporting)
    features["symptom_reporting_rate"] = df["reported"].mean()
    if len(recent) > 0:
        features["symptom_reporting_rate_recent"] = recent["reported"].mean()
    else:
        features["symptom_reporting_rate_recent"] = features["symptom_reporting_rate"]

    # Symptom scores (only from reported days)
    reported = df[df["reported"]].copy()
    reported_recent = recent[recent["reported"]].copy() if len(recent) > 0 else pd.DataFrame()

    symptom_cols = {
        "dyspnea_score": "dyspnea",
        "orthopnea_pillows": "orthopnea",
        "ankle_edema": "edema",
        "exercise_tolerance": "exercise",
    }

    for col, prefix in symptom_cols.items():
        if len(reported) > 0 and col in reported.columns:
            valid = reported[col].dropna()
            if len(valid) > 0:
                features[f"{prefix}_current"] = valid.iloc[-1]
                features[f"{prefix}_mean"] = valid.mean()

                if len(valid) >= 2:
                    x = np.arange(len(valid))
                    slope, _, _, _, _ = stats.linregress(x, valid.values)
                    features[f"{prefix}_trend"] = slope
                else:
                    features[f"{prefix}_trend"] = 0.0
            else:
                features[f"{prefix}_current"] = 0.0
                features[f"{prefix}_mean"] = 0.0
                features[f"{prefix}_trend"] = 0.0
        else:
            features[f"{prefix}_current"] = 0.0
            features[f"{prefix}_mean"] = 0.0
            features[f"{prefix}_trend"] = 0.0

    # Composite symptom burden score
    features["symptom_burden"] = (
        features.get("dyspnea_current", 0) / 10
        + features.get("orthopnea_current", 0) / 4
        + features.get("edema_current", 0) / 3
        + (10 - features.get("exercise_current", 10)) / 10
    ) / 4  # normalized 0-1

    # Medication adherence
    if len(reported) > 0 and "medication_adherent" in reported.columns:
        valid = reported["medication_adherent"].dropna()
        features["med_adherence_rate"] = valid.mean() if len(valid) > 0 else 1.0
    else:
        features["med_adherence_rate"] = 1.0

    # Days since last report (missingness signal)
    if len(reported) > 0:
        features["days_since_last_report"] = up_to_day - reported["day"].max()
    else:
        features["days_since_last_report"] = up_to_day

    return features


def build_feature_matrix(data: dict, prediction_days: list = None,
                         label_horizon: int = 7, adaptive_windows: bool = False,
                         use_tolerance_labels: bool = False) -> pd.DataFrame:
    """
    Build the full feature matrix for all patients.

    For each patient, we generate feature vectors at multiple time points
    (prediction_days) to maximize training data and simulate real-time
    prediction.

    Args:
        data: dict with vitals, labs, symptoms, outcomes DataFrames
              (optionally includes 'medications' DataFrame)
        prediction_days: fixed prediction points (used if adaptive_windows=False)
        label_horizon: days to look ahead for deterioration label
        adaptive_windows: if True, generate prediction points based on each
                         patient's actual data length (for short ICU stays)
        use_tolerance_labels: if True and medications data with titration events
                              exists, use tolerance as the label instead of
                              deterioration
    """
    if prediction_days is None:
        prediction_days = [7, 10, 14, 17, 21, 24, 27]  # multiple windows

    vitals = data["vitals"]
    labs = data["labs"]
    symptoms = data["symptoms"]
    outcomes = data["outcomes"]
    medications = data.get("medications", pd.DataFrame())
    has_medications = len(medications) > 0 and "drug_class" in medications.columns

    all_rows = []
    patient_ids = outcomes["patient_id"].unique()

    for i, pid in enumerate(patient_ids):
        patient_vitals = vitals[vitals["patient_id"] == pid]
        patient_labs = labs[labs["patient_id"] == pid]
        patient_symptoms = symptoms[symptoms["patient_id"] == pid]
        patient_meds = medications[medications["patient_id"] == pid] if has_medications else pd.DataFrame()

        outcome_rows = outcomes[outcomes["patient_id"] == pid]
        if len(outcome_rows) == 0:
            continue
        patient_outcome = outcome_rows.iloc[0]

        # Skip patients with no vitals data
        if len(patient_vitals) == 0:
            continue

        deterioration_day = patient_outcome["deterioration_day"]

        # Determine prediction points for this patient
        if adaptive_windows:
            max_day = int(patient_vitals["day"].max())
            patient_pred_days = list(range(1, max_day + 1))
        else:
            patient_pred_days = prediction_days

        # Static patient info for eGFR
        patient_age = int(patient_outcome.get("age", 65)) if not pd.isna(patient_outcome.get("age")) else 65
        patient_sex_male = int(patient_outcome.get("sex", "U") == "M")

        for pred_day in patient_pred_days:
            # Skip if this prediction point is after deterioration already happened
            if patient_outcome["deteriorated"] and deterioration_day is not None:
                if pred_day >= deterioration_day:
                    continue

            # Build features using only data up to pred_day
            vital_feats = compute_vital_features(patient_vitals, pred_day)
            lab_feats = compute_lab_features(patient_labs, pred_day, age=patient_age, sex_male=patient_sex_male)
            symptom_feats = compute_symptom_features(patient_symptoms, pred_day)

            if not vital_feats:
                continue

            # Medication features (if available)
            med_feats = {}
            if has_medications and len(patient_meds) > 0:
                med_feats = compute_medication_features(patient_meds, pred_day)

            # Merge all features
            row = {
                "patient_id": pid,
                "prediction_day": pred_day,
                **vital_feats,
                **lab_feats,
                **symptom_feats,
                **med_feats,
            }

            # Static features
            row["age"] = patient_age
            row["sex_male"] = patient_sex_male

            # Optional static features (may be missing in real clinical data)
            def _safe_get(outcome, col, default=0):
                val = outcome.get(col, np.nan)
                return default if pd.isna(val) else val

            row["ef"] = _safe_get(patient_outcome, "ejection_fraction", 0)
            row["nyha_class"] = _safe_get(patient_outcome, "nyha_class", 0)
            row["n_comorbidities"] = _safe_get(patient_outcome, "n_comorbidities", 0)
            row["n_medications"] = _safe_get(patient_outcome, "n_medications", 0)

            # EF category one-hot
            ef_cat = str(patient_outcome.get("ef_category", "Unknown"))
            row["ef_reduced"] = int(ef_cat == "HFrEF")
            row["ef_mid"] = int(ef_cat == "HFmrEF")
            row["ef_preserved"] = int(ef_cat == "HFpEF")

            # Label
            if use_tolerance_labels and has_medications and "titration_day" in patient_meds.columns:
                # Tolerance label: did the next titration event succeed?
                titration_events = patient_meds[
                    (patient_meds["status"] == "titration_event") &
                    (patient_meds["titration_day"] > pred_day) &
                    (patient_meds["titration_day"] <= pred_day + label_horizon)
                ]
                if len(titration_events) > 0 and "tolerated" in titration_events.columns:
                    row["label"] = int(titration_events.iloc[0]["tolerated"])
                else:
                    # No titration event in window — use inverse of deterioration
                    if patient_outcome["deteriorated"] and deterioration_day is not None:
                        days_until = deterioration_day - pred_day
                        row["label"] = int(not (0 < days_until <= label_horizon))
                    else:
                        row["label"] = 1  # stable = tolerant
            else:
                # Standard deterioration label
                if patient_outcome["deteriorated"] and deterioration_day is not None:
                    days_until = deterioration_day - pred_day
                    row["label"] = int(0 < days_until <= label_horizon)
                else:
                    row["label"] = 0

            all_rows.append(row)

        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(patient_ids)} patients...")

    feature_df = pd.DataFrame(all_rows)

    label_name = "tolerance" if use_tolerance_labels else "deterioration"
    print(f"\nFeature matrix shape: {feature_df.shape}")
    print(f"Positive labels ({label_name}): {feature_df['label'].sum()} ({100*feature_df['label'].mean():.1f}%)")
    print(f"Features per sample: {feature_df.shape[1] - 3}")  # minus patient_id, prediction_day, label

    return feature_df


if __name__ == "__main__":
    import os
    import sys

    # Data locations default to the project's data/ dir; override the base with
    # the HFTA_DATA_DIR environment variable (e.g. an HPC workspace).
    project_root = Path(__file__).resolve().parents[2]
    base_dir = Path(os.environ.get("HFTA_DATA_DIR", project_root / "data"))

    # Support both synthetic and combined real data
    if len(sys.argv) > 1 and sys.argv[1] == "--combined":
        data_dir = str(base_dir / "combined")
        output_path = str(base_dir / "processed" / "features_real.parquet")
        use_adaptive = True
    else:
        data_dir = str(base_dir / "raw")
        output_path = str(base_dir / "processed" / "features.parquet")
        use_adaptive = False

    print(f"Loading data from {data_dir}...")
    data = load_raw_data(data_dir)

    print("Building feature matrix...")
    if use_adaptive:
        # ICU stays are short (median 2.7 days), use adaptive windows and shorter horizon
        features = build_feature_matrix(data, label_horizon=3, adaptive_windows=True)
    else:
        features = build_feature_matrix(data)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)
    print(f"Saved to {output_path}")
