"""
Prediction engine that wraps the trained model.

Handles:
- Feature extraction from raw API input
- Model inference
- SHAP explanation generation
- Clinical action mapping
"""

import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import shap

from src.features.build_features import (
    compute_lab_features,
    compute_symptom_features,
    compute_vital_features,
)

# Human-readable feature name mapping for CDS display
FEATURE_DISPLAY_NAMES = {
    "weightkg_trend_3d": "Weight trend (3-day)",
    "weightkg_baseline_dev": "Weight change from baseline",
    "weight_change_total_kg": "Total weight change (kg)",
    "weight_change_total_pct": "Total weight change (%)",
    "weight_velocity_kg_per_measurement": "Weight gain velocity",
    "weightkg_std_3d": "Weight variability (3-day)",
    "weightkg_current": "Current weight",
    "spo2_trend_3d": "SpO2 trend (3-day)",
    "spo2_baseline_dev": "SpO2 change from baseline",
    "spo2_baseline_pct_change": "SpO2 % change from baseline",
    "spo2_current": "Current SpO2",
    "heartrate_trend_3d": "Heart rate trend (3-day)",
    "heartrate_current": "Current heart rate",
    "systolicbp_trend_3d": "Systolic BP trend (3-day)",
    "systolicbp_current": "Current systolic BP",
    "systolicbp_std_3d": "BP variability (3-day)",
    "respiratoryrate_trend_3d": "Respiratory rate trend (3-day)",
    "respiratoryrate_current": "Current respiratory rate",
    "pulse_pressure": "Pulse pressure",
    "shock_index": "Shock index (HR/SBP)",
    "rate_pressure_product": "Rate-pressure product",
    "bnp_current": "Current BNP",
    "bnp_trend": "BNP trend",
    "bnp_change": "BNP change from baseline",
    "bnp_above_300": "BNP > 300 pg/mL",
    "bnp_above_600": "BNP > 600 pg/mL",
    "creat_current": "Current creatinine",
    "creat_trend": "Creatinine trend",
    "creat_elevated": "Creatinine elevated (>1.5)",
    "k_current": "Current potassium",
    "k_trend": "Potassium trend",
    "k_abnormal": "Potassium abnormal",
    "symptom_burden": "Symptom burden score",
    "symptom_reporting_rate": "Symptom reporting rate",
    "dyspnea_current": "Current dyspnea score",
    "dyspnea_trend": "Dyspnea trend",
    "dyspnea_mean": "Average dyspnea",
    "orthopnea_current": "Current orthopnea (pillows)",
    "orthopnea_trend": "Orthopnea trend",
    "edema_current": "Current ankle edema",
    "exercise_current": "Current exercise tolerance",
    "exercise_trend": "Exercise tolerance trend",
    "med_adherence_rate": "Medication adherence rate",
    "days_since_last_report": "Days since last symptom report",
    "age": "Age",
    "sex_male": "Male sex",
    "ef": "Ejection fraction",
    "nyha_class": "NYHA class",
    "n_comorbidities": "Number of comorbidities",
    "n_medications": "Number of medications",
    # 3-day window statistics
    "weightkg_mean_3d": "Weight (3-day avg)",
    "weightkg_min_3d": "Weight (3-day min)",
    "weightkg_max_3d": "Weight (3-day max)",
    "weightkg_baseline_pct_change": "Weight % change from baseline",
    "spo2_mean_3d": "SpO2 (3-day avg)",
    "spo2_std_3d": "SpO2 variability (3-day)",
    "spo2_min_3d": "SpO2 (3-day min)",
    "spo2_max_3d": "SpO2 (3-day max)",
    "heartrate_mean_3d": "Heart rate (3-day avg)",
    "heartrate_std_3d": "Heart rate variability (3-day)",
    "heartrate_min_3d": "Heart rate (3-day min)",
    "heartrate_max_3d": "Heart rate (3-day max)",
    "heartrate_baseline_dev": "Heart rate change from baseline",
    "heartrate_baseline_pct_change": "Heart rate % change from baseline",
    "systolicbp_mean_3d": "Systolic BP (3-day avg)",
    "systolicbp_min_3d": "Systolic BP (3-day min)",
    "systolicbp_max_3d": "Systolic BP (3-day max)",
    "systolicbp_baseline_dev": "Systolic BP change from baseline",
    "systolicbp_baseline_pct_change": "Systolic BP % change from baseline",
    "diastolicbp_current": "Current diastolic BP",
    "diastolicbp_mean_3d": "Diastolic BP (3-day avg)",
    "diastolicbp_std_3d": "Diastolic BP variability (3-day)",
    "diastolicbp_min_3d": "Diastolic BP (3-day min)",
    "diastolicbp_max_3d": "Diastolic BP (3-day max)",
    "diastolicbp_trend_3d": "Diastolic BP trend (3-day)",
    "diastolicbp_baseline_dev": "Diastolic BP change from baseline",
    "diastolicbp_baseline_pct_change": "Diastolic BP % change from baseline",
    "respiratoryrate_mean_3d": "Respiratory rate (3-day avg)",
    "respiratoryrate_std_3d": "Respiratory rate variability (3-day)",
    "respiratoryrate_min_3d": "Respiratory rate (3-day min)",
    "respiratoryrate_max_3d": "Respiratory rate (3-day max)",
    "respiratoryrate_baseline_dev": "Respiratory rate change from baseline",
    "respiratoryrate_baseline_pct_change": "Respiratory rate % change from baseline",
    "bnp_pct_change": "BNP % change from baseline",
    "bnp_above_900": "BNP > 900 pg/mL",
    "creat_change": "Creatinine change from baseline",
    "creat_pct_change": "Creatinine % change from baseline",
    "k_change": "Potassium change from baseline",
    "k_pct_change": "Potassium % change from baseline",
    "symptom_reporting_rate_recent": "Recent symptom reporting rate",
    "orthopnea_mean": "Average orthopnea (pillows)",
    "edema_mean": "Average ankle edema",
    "edema_trend": "Ankle edema trend",
    "exercise_mean": "Average exercise tolerance",
    "ef_reduced": "Reduced EF (HFrEF)",
    "ef_mid": "Mid-range EF (HFmrEF)",
    "ef_preserved": "Preserved EF (HFpEF)",
}

# Token expansions for the humanizing fallback, so a raw feature name never
# reaches the UI even if a new column is added without a curated label.
_HUMANIZE_TOKENS = {
    "weightkg": "Weight", "spo2": "SpO2", "heartrate": "Heart rate",
    "systolicbp": "Systolic BP", "diastolicbp": "Diastolic BP",
    "respiratoryrate": "Respiratory rate", "bnp": "BNP", "creat": "Creatinine",
    "k": "Potassium", "ef": "EF", "nyha": "NYHA",
    "mean": "avg", "std": "variability", "min": "min", "max": "max",
    "dev": "deviation", "pct": "%", "3d": "(3-day)", "7d": "(7-day)",
}


def humanize_feature(name: str) -> str:
    """Fallback: turn a snake_case feature name into a readable label."""
    words = [_HUMANIZE_TOKENS.get(tok, tok) for tok in name.split("_")]
    label = " ".join(w for w in words if w)
    return label[:1].upper() + label[1:] if label else name

# ---------------------------------------------------------------------------
# GDMT Catalog — ACC/AHA 4-pillar guideline-directed medical therapy
# ---------------------------------------------------------------------------
GDMT_CATALOG = {
    "raasi": {
        "label": "RAASi",
        "drugs": {
            "lisinopril": {
                "dose_steps_mg": [2.5, 5, 10, 20, 40],
                "target_dose_mg": 40,
            },
            "sacubitril_valsartan": {
                "dose_steps_mg": [24, 49, 97],
                "target_dose_mg": 97,
            },
            "losartan": {
                "dose_steps_mg": [25, 50, 100, 150],
                "target_dose_mg": 150,
            },
        },
        "safety_checks": [
            {"check_name": "SBP >= 100", "feature": "systolicbp_current", "op": ">=", "threshold": 100},
            {"check_name": "K+ < 5.0", "feature": "k_current", "op": "<", "threshold": 5.0},
            {"check_name": "Creatinine stable", "feature": "creat_trend", "op": "<=", "threshold": 0.3},
        ],
        "ef_categories": ["HFrEF", "HFmrEF"],
    },
    "beta_blocker": {
        "label": "Beta-blocker",
        "drugs": {
            "carvedilol": {
                "dose_steps_mg": [3.125, 6.25, 12.5, 25],
                "target_dose_mg": 25,
            },
            "metoprolol_succinate": {
                "dose_steps_mg": [12.5, 25, 50, 100, 200],
                "target_dose_mg": 200,
            },
        },
        "safety_checks": [
            {"check_name": "HR >= 60", "feature": "heartrate_current", "op": ">=", "threshold": 60},
            {"check_name": "SBP >= 90", "feature": "systolicbp_current", "op": ">=", "threshold": 90},
            {"check_name": "No acute decompensation", "feature": "symptom_burden", "op": "<=", "threshold": 0.6},
        ],
        "ef_categories": ["HFrEF", "HFmrEF"],
    },
    "mra": {
        "label": "MRA",
        "drugs": {
            "spironolactone": {
                "dose_steps_mg": [12.5, 25, 50],
                "target_dose_mg": 50,
            },
            "eplerenone": {
                "dose_steps_mg": [25, 50],
                "target_dose_mg": 50,
            },
        },
        "safety_checks": [
            {"check_name": "K+ < 5.0", "feature": "k_current", "op": "<", "threshold": 5.0},
            {"check_name": "eGFR >= 30", "feature": "egfr", "op": ">=", "threshold": 30},
            {"check_name": "No AKI", "feature": "creat_trend", "op": "<=", "threshold": 0.3},
        ],
        "ef_categories": ["HFrEF", "HFmrEF"],
    },
    "sglt2i": {
        "label": "SGLT2i",
        "drugs": {
            "dapagliflozin": {
                "dose_steps_mg": [10],
                "target_dose_mg": 10,
            },
            "empagliflozin": {
                "dose_steps_mg": [10],
                "target_dose_mg": 10,
            },
        },
        "safety_checks": [
            {"check_name": "eGFR >= 20", "feature": "egfr", "op": ">=", "threshold": 20},
        ],
        "ef_categories": ["HFrEF", "HFmrEF", "HFpEF"],
    },
}

CLINICAL_ACTIONS = {
    "low": {
        "action": "Patient stable — consider GDMT optimization. Review titration opportunities.",
        "urgency": "routine",
        "rationale": "Vital signs stable and within expected range. Good window for GDMT uptitration if below target doses.",
    },
    "medium": {
        "action": "Schedule phone check-in within 4 hours. Hold GDMT uptitration until reassessed.",
        "urgency": "soon",
        "rationale": "Early warning signals detected. Reassess before any medication changes.",
    },
    "high": {
        "action": "Alert physician. Hold all GDMT uptitration. Consider diuretic adjustment and in-person assessment.",
        "urgency": "urgent",
        "rationale": "Multiple deterioration indicators trending abnormally. Not safe to uptitrate. Intervention likely needed within 24 hours.",
    },
    "critical": {
        "action": "Immediate physician notification. Hold all uptitration. Consider ED transfer if symptoms worsen.",
        "urgency": "emergent",
        "rationale": "High probability of acute decompensation. All GDMT changes on hold. Immediate clinical review required.",
    },
}


# Canonical risk-tier boundaries (probability of deterioration), defined in one
# place so the API, dashboard, and docs stay in sync regardless of what an older
# model bundle happened to be trained with:
#   Low <15%  |  Medium 15–59%  |  High 60–79%  |  Critical >=80%
RISK_TIER_THRESHOLDS = {
    "medium_risk": 0.15,
    "high_risk": 0.60,
    "critical_risk": 0.80,
}


class HFPredictor:
    def __init__(self, model_path: str):
        with open(model_path, "rb") as f:
            bundle = pickle.load(f)

        self.model = bundle["model"]
        self.model_name = bundle["model_name"]
        self.feature_cols = bundle["feature_cols"]
        self.metrics = bundle["metrics"]
        # Tiers are defined once here rather than read from the pickled bundle,
        # so the API, dashboard, and docs cannot drift out of sync with an
        # older or externally-supplied model file.
        self.tiers = RISK_TIER_THRESHOLDS
        self.explainer = shap.TreeExplainer(self.model)
        self.version = "0.1.0"

    def predict(self, vitals_df: pd.DataFrame, labs_df: pd.DataFrame,
                symptoms_df: pd.DataFrame, patient_context: dict) -> dict:
        """
        Run prediction for a single patient given their current data.

        Returns risk score, tier, contributing factors, and suggested action.
        """
        # Determine the current day (latest data point)
        current_day = vitals_df["day"].max()

        # Extract features
        vital_feats = compute_vital_features(vitals_df, current_day)
        lab_feats = compute_lab_features(
            labs_df, current_day,
            age=int(patient_context.get("age", 65)),
            sex_male=int(patient_context.get("sex_male", 1)),
        ) if len(labs_df) > 0 else {}
        symptom_feats = compute_symptom_features(symptoms_df, current_day) if len(symptoms_df) > 0 else {}

        # Merge
        features = {**vital_feats, **lab_feats, **symptom_feats, **patient_context}

        # Build feature vector in correct order
        feature_vector = []
        for col in self.feature_cols:
            feature_vector.append(features.get(col, 0.0))

        X = np.array([feature_vector])

        # Predict
        probability = float(self.model.predict_proba(X)[:, 1][0])
        risk_score = round(probability * 100, 1)

        # Determine tier
        if probability >= self.tiers["critical_risk"]:
            tier = "critical"
        elif probability >= self.tiers["high_risk"]:
            tier = "high"
        elif probability >= self.tiers["medium_risk"]:
            tier = "medium"
        else:
            tier = "low"

        # SHAP explanations
        shap_values = self.explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # Top contributing factors
        shap_flat = shap_values[0]
        abs_shap = np.abs(shap_flat)
        top_indices = np.argsort(abs_shap)[::-1][:5]

        top_factors = []
        for idx in top_indices:
            feat_name = self.feature_cols[idx]
            top_factors.append({
                "feature": feat_name,
                "display_name": FEATURE_DISPLAY_NAMES.get(feat_name) or humanize_feature(feat_name),
                "value": round(float(feature_vector[idx]), 3),
                "impact": round(float(shap_flat[idx]), 4),
                "direction": "increasing_risk" if shap_flat[idx] > 0 else "decreasing_risk",
            })

        # Suggested action
        action = CLINICAL_ACTIONS[tier]

        return {
            "risk_score": risk_score,
            "probability": round(probability, 4),
            "risk_tier": tier,
            "top_factors": top_factors,
            "suggested_action": action,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_version": f"{self.model_name}-v{self.version}",
        }

    def predict_tolerance(self, vitals_df: pd.DataFrame, labs_df: pd.DataFrame,
                          symptoms_df: pd.DataFrame, patient_context: dict) -> dict:
        """
        Predict whether a patient will tolerate GDMT uptitration.

        This is the inverse of deterioration risk: tolerance = 1 - deterioration.
        SHAP direction labels are flipped (what increases deterioration risk
        decreases tolerance, and vice versa).
        """
        result = self.predict(vitals_df, labs_df, symptoms_df, patient_context)

        tolerance_score = round(100 - result["risk_score"], 1)
        tolerance_probability = round(1 - result["probability"], 4)

        # Flip SHAP direction labels for tolerance framing
        tolerance_factors = []
        for f in result["top_factors"]:
            flipped_direction = (
                "increasing_tolerance" if f["direction"] == "decreasing_risk"
                else "decreasing_tolerance"
            )
            tolerance_factors.append({
                **f,
                "direction": flipped_direction,
            })

        return {
            "tolerance_score": tolerance_score,
            "tolerance_probability": tolerance_probability,
            "risk_score": result["risk_score"],
            "risk_tier": result["risk_tier"],
            "tolerance_factors": tolerance_factors,
            "suggested_action": result["suggested_action"],
            "timestamp": result["timestamp"],
            "model_version": result["model_version"],
        }


class GDMTEngine:
    """
    Rule-based GDMT titration recommendation engine.

    Implements ACC/AHA guideline logic: for each drug class, checks safety
    criteria against current vitals/labs, determines next dose step, and
    produces a recommendation (uptitrate / hold / initiate / at_target).
    """

    def __init__(self, predictor: HFPredictor):
        self.predictor = predictor

    def evaluate_titration(
        self,
        drug_class: str,
        medication: dict,
        features: dict,
    ) -> dict:
        """
        Evaluate whether a specific drug can be uptitrated.

        Args:
            drug_class: Key into GDMT_CATALOG (e.g. "raasi")
            medication: Dict with generic_name, current_dose_mg, target_dose_mg, status
            features: Current patient feature dict (from vitals/labs extraction)

        Returns:
            Recommendation dict with action, safety_checks, rationale, etc.
        """
        catalog_entry = GDMT_CATALOG[drug_class]
        drug_info = catalog_entry["drugs"][medication["generic_name"]]

        safety_checks = self._run_safety_checks(catalog_entry["safety_checks"], features)
        all_passed = all(c["passed"] for c in safety_checks)

        current_dose = medication["current_dose_mg"]
        target_dose = drug_info["target_dose_mg"]

        if current_dose == 0:
            # Not started yet
            next_dose = drug_info["dose_steps_mg"][0]
            if all_passed:
                action = "initiate"
                rationale = f"All safety checks passed. Initiate {medication['generic_name']} at {next_dose} mg."
            else:
                action = "hold"
                failed = [c["check_name"] for c in safety_checks if not c["passed"]]
                rationale = f"Cannot initiate — failed: {', '.join(failed)}."
                next_dose = None
        elif current_dose >= target_dose:
            action = "at_target"
            next_dose = None
            rationale = f"Already at target dose ({target_dose} mg). Continue current regimen."
        else:
            next_dose = self._get_next_dose(drug_info["dose_steps_mg"], current_dose)
            if next_dose is None:
                action = "at_target"
                rationale = f"At maximum available dose step. Continue current regimen."
            elif all_passed:
                action = "uptitrate"
                rationale = f"All safety checks passed. Increase to {next_dose} mg (target: {target_dose} mg)."
            else:
                action = "hold"
                failed = [c["check_name"] for c in safety_checks if not c["passed"]]
                rationale = f"Hold at {current_dose} mg — failed: {', '.join(failed)}."
                next_dose = None

        return {
            "drug_class": drug_class,
            "generic_name": medication["generic_name"],
            "action": action,
            "current_dose_mg": current_dose,
            "next_dose_mg": next_dose,
            "target_dose_mg": target_dose,
            "safety_checks": safety_checks,
            "rationale": rationale,
        }

    def _run_safety_checks(self, checks: list[dict], features: dict) -> list[dict]:
        """Evaluate each safety check against current feature values."""
        results = []
        for check in checks:
            value = features.get(check["feature"])
            if value is None:
                # Missing data — fail safe
                results.append({
                    "check_name": check["check_name"],
                    "passed": False,
                    "current_value": None,
                    "threshold": check["threshold"],
                })
                continue

            op = check["op"]
            threshold = check["threshold"]
            if op == ">=":
                passed = value >= threshold
            elif op == "<=":
                passed = value <= threshold
            elif op == "<":
                passed = value < threshold
            elif op == ">":
                passed = value > threshold
            else:
                passed = False

            results.append({
                "check_name": check["check_name"],
                "passed": bool(passed),
                "current_value": round(float(value), 2),
                "threshold": threshold,
            })
        return results

    @staticmethod
    def _get_next_dose(dose_steps: list[float], current_dose: float) -> float | None:
        """Find the next dose step above the current dose."""
        for step in dose_steps:
            if step > current_dose:
                return step
        return None

    @staticmethod
    def compute_optimization_score(medications: list[dict]) -> float:
        """
        Compute GDMT optimization as % of target doses achieved across all pillars.

        Only counts medications that are indicated (status != "not_indicated").
        """
        indicated = [m for m in medications if m["status"] != "not_indicated"]
        if not indicated:
            return 0.0
        total_pct = sum(
            min(m["current_dose_mg"] / m["target_dose_mg"], 1.0)
            for m in indicated
            if m["target_dose_mg"] > 0
        )
        return round(total_pct / len(indicated) * 100, 1)
