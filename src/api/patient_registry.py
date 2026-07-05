"""
Patient Registry — loads parquet data, runs ML predictions, caches results.

Provides an in-memory dict[patient_id, Patient] that matches the dashboard's
TypeScript Patient interface so the API can serve it directly as JSON.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.api.predictor import HFPredictor, GDMTEngine, GDMT_CATALOG

logger = logging.getLogger(__name__)

# Deterministic name pools (hashed from patient_id)
FIRST_NAMES_F = [
    "Margaret", "Dorothy", "Patricia", "Eleanor", "Alice", "Catherine",
    "Elizabeth", "Helen", "Barbara", "Virginia", "Ruth", "Frances",
    "Maria", "Rose", "Anna", "Jean", "Evelyn", "Gloria", "Janet", "Lillian",
    "Carol", "Diane", "Betty", "Sandra", "Linda", "Karen", "Nancy",
    "Sharon", "Donna", "Irene",
]
FIRST_NAMES_M = [
    "Robert", "James", "Harold", "William", "Charles", "George", "Thomas",
    "Richard", "Joseph", "Edward", "Frank", "Henry", "Walter", "Arthur",
    "Albert", "Paul", "Raymond", "Donald", "Eugene", "David",
    "Kenneth", "Gerald", "Larry", "Dennis", "Roger", "Samuel",
    "Ronald", "Daniel", "Philip", "Howard",
]
LAST_NAMES = [
    "Chen", "Williams", "Martinez", "Thompson", "Davis", "Kim", "Johnson",
    "Anderson", "Wilson", "Taylor", "Brown", "Garcia", "Miller", "Jones",
    "Lee", "Harris", "Clark", "Lewis", "Robinson", "Walker",
    "Young", "Allen", "King", "Wright", "Scott", "Hill", "Green",
    "Adams", "Baker", "Hall",
]


def _hash_int(patient_id: str, salt: str = "") -> int:
    """Deterministic integer from patient_id string."""
    return int(hashlib.sha256(f"{patient_id}{salt}".encode()).hexdigest(), 16)


def generate_name(patient_id: str, sex: str) -> str:
    """Deterministic full name from patient_id and sex."""
    h = _hash_int(patient_id)
    pool = FIRST_NAMES_F if sex == "F" else FIRST_NAMES_M
    first = pool[h % len(pool)]
    last = LAST_NAMES[_hash_int(patient_id, "last") % len(LAST_NAMES)]
    return f"{first} {last}"


def _ef_category(ef: float) -> str:
    if ef < 40:
        return "HFrEF"
    if ef < 50:
        return "HFmrEF"
    return "HFpEF"


def _diagnosis_text(ef_cat: str, nyha: int) -> str:
    base = {
        "HFrEF": "Heart failure with reduced EF",
        "HFmrEF": "Heart failure with mid-range EF",
        "HFpEF": "Heart failure with preserved EF",
    }.get(ef_cat, "Heart failure")
    return f"{base}, NYHA class {nyha}"


def _safe_float(val, default: float = 0.0) -> float:
    """Return val as float, falling back to default if NaN/None."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    return float(val)


def _safe_int(val, default: int = 0) -> int:
    """Return val as int, falling back to default if NaN/None."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    return int(val)


_VITAL_COLS = [
    "weight_kg", "spo2", "heart_rate", "systolic_bp", "diastolic_bp",
    "respiratory_rate",
]


def _fill_vitals_for_display(vitals_df: pd.DataFrame) -> pd.DataFrame:
    """Forward/back-fill missing vital values for DISPLAY only.

    Source data records vitals at different cadences, so any given row may have
    gaps (stored as NaN). Rendering those as 0 makes charts plunge to zero and
    the 'current value' read 0 bpm / 0% — clinically impossible. Last-observation-
    carried-forward (then back-filled) yields smooth trends and a sensible latest
    value. This is applied to a copy; the model still sees the raw data.
    """
    df = vitals_df.sort_values("day").copy() if "day" in vitals_df.columns else vitals_df.copy()
    for col in _VITAL_COLS:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()
    return df


def _round_opt(val, ndigits: int):
    """Round to ndigits, or return None for missing (NaN/None) values."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    return round(float(val), ndigits)


def format_vitals(vitals_df: pd.DataFrame) -> list[dict]:
    """Convert vitals DataFrame rows to the VitalReading interface."""
    readings = []
    for _, row in vitals_df.iterrows():
        readings.append({
            "timestamp": row["timestamp"].isoformat() if isinstance(row["timestamp"], (datetime, pd.Timestamp)) else str(row["timestamp"]),
            "day": _safe_int(row.get("day", 0)),
            "weight_kg": round(_safe_float(row.get("weight_kg", 0)), 1),
            "spo2": round(_safe_float(row.get("spo2", 0)), 1),
            "heart_rate": _safe_int(row.get("heart_rate", 0)),
            "systolic_bp": _safe_int(row.get("systolic_bp", 0)),
            "diastolic_bp": _safe_int(row.get("diastolic_bp", 0)),
            "respiratory_rate": _safe_int(row.get("respiratory_rate", 0)),
        })
    return readings


def format_labs(labs_df: pd.DataFrame) -> list[dict]:
    """Convert labs DataFrame rows to the LabReading interface."""
    readings = []
    for _, row in labs_df.iterrows():
        readings.append({
            "timestamp": row["timestamp"].isoformat() if isinstance(row["timestamp"], (datetime, pd.Timestamp)) else str(row["timestamp"]),
            "day": _safe_int(row.get("day", 0)),
            "bnp_pg_ml": _round_opt(row.get("bnp_pg_ml"), 1),
            "creatinine_mg_dl": _round_opt(row.get("creatinine_mg_dl"), 2),
            "potassium_meq_l": _round_opt(row.get("potassium_meq_l"), 1),
        })
    return readings


def compute_trend(vitals_df: pd.DataFrame) -> str:
    """Determine weight trend: 'rising', 'falling', or 'stable'."""
    if len(vitals_df) < 4 or "weight_kg" not in vitals_df.columns:
        return "stable"

    wt = vitals_df["weight_kg"].dropna()
    if len(wt) < 4:
        return "stable"

    n = len(wt)
    baseline = wt.iloc[: max(n // 3, 2)].mean()
    recent = wt.iloc[-max(n // 3, 2) :].mean()
    delta = recent - baseline

    if delta > 1.0:
        return "rising"
    if delta < -1.0:
        return "falling"
    return "stable"


def _generate_medication_state(patient_id: str, ef_cat: str) -> list[dict]:
    """
    Generate a deterministic medication profile for a patient based on EF category.

    HFrEF: all 4 pillars (RAASi, BB, MRA, SGLT2i)
    HFmrEF: RAASi, BB, SGLT2i (3 pillars)
    HFpEF: SGLT2i only
    """
    h = _hash_int(patient_id, "meds")
    medications = []

    for drug_class, catalog in GDMT_CATALOG.items():
        indicated = ef_cat in catalog["ef_categories"]
        if not indicated:
            # Pick the first drug but mark not indicated
            first_drug = list(catalog["drugs"].keys())[0]
            info = catalog["drugs"][first_drug]
            medications.append({
                "drug_class": drug_class,
                "generic_name": first_drug,
                "current_dose_mg": 0,
                "target_dose_mg": info["target_dose_mg"],
                "status": "not_indicated",
            })
            continue

        # Deterministically pick which drug in this class
        drug_names = list(catalog["drugs"].keys())
        drug_idx = h % len(drug_names)
        h = _hash_int(patient_id, f"drug_{drug_class}")
        generic_name = drug_names[drug_idx]
        info = catalog["drugs"][generic_name]

        # Deterministic dose level (0 = not started, up to len(steps) = at target)
        steps = info["dose_steps_mg"]
        dose_idx = h % (len(steps) + 1)  # 0 means not yet started
        if dose_idx == 0:
            current_dose = 0
            status = "not_started"
        else:
            current_dose = steps[min(dose_idx - 1, len(steps) - 1)]
            status = "at_target" if current_dose >= info["target_dose_mg"] else "sub_therapeutic"

        medications.append({
            "drug_class": drug_class,
            "generic_name": generic_name,
            "current_dose_mg": current_dose,
            "target_dose_mg": info["target_dose_mg"],
            "status": status,
        })

    return medications


class PatientRegistry:
    """Loads parquet data, runs predictions, and caches Patient dicts."""

    def __init__(self, data_dir: str, predictor: HFPredictor):
        self.data_dir = Path(data_dir)
        self.predictor = predictor
        self.gdmt_engine = GDMTEngine(predictor)
        self.patients: dict[str, dict] = {}

    def load(self) -> int:
        """Load data and populate the patient cache. Returns patient count."""
        logger.info("Loading parquet data from %s", self.data_dir)

        vitals = pd.read_parquet(self.data_dir / "vitals.parquet")
        labs = pd.read_parquet(self.data_dir / "labs.parquet")
        symptoms = pd.read_parquet(self.data_dir / "symptoms.parquet")
        outcomes = pd.read_parquet(self.data_dir / "outcomes.parquet")

        patient_ids = outcomes["patient_id"].unique()
        loaded = 0
        errors = 0

        for pid in patient_ids:
            try:
                patient = self._build_patient(
                    pid, outcomes, vitals, labs, symptoms,
                )
                if patient is not None:
                    self.patients[pid] = patient
                    loaded += 1
            except Exception:
                errors += 1
                logger.warning("Failed to build patient %s", pid, exc_info=True)

        logger.info(
            "Registry loaded: %d patients (%d skipped, %d errors)",
            loaded, len(patient_ids) - loaded - errors, errors,
        )
        return loaded

    def _build_patient(
        self,
        pid: str,
        outcomes: pd.DataFrame,
        vitals: pd.DataFrame,
        labs: pd.DataFrame,
        symptoms: pd.DataFrame,
    ) -> dict | None:
        """Build a single Patient dict matching the TS interface."""
        outcome = outcomes[outcomes["patient_id"] == pid].iloc[0]
        p_vitals = vitals[vitals["patient_id"] == pid].copy()
        p_labs = labs[labs["patient_id"] == pid].copy()
        p_symptoms = symptoms[symptoms["patient_id"] == pid].copy()

        # Need at least 2 vital readings for prediction
        if len(p_vitals) < 2:
            return None

        # --- Patient context ---
        age = int(outcome.get("age", 65)) if not pd.isna(outcome.get("age")) else 65
        raw_sex = str(outcome.get("sex", "U"))
        sex = raw_sex if raw_sex in ("M", "F") else "U"
        ef = float(outcome.get("ejection_fraction", 35)) if not pd.isna(outcome.get("ejection_fraction")) else 35.0
        ef_cat = str(outcome.get("ef_category", "")) if not pd.isna(outcome.get("ef_category")) else ""
        if ef_cat not in ("HFrEF", "HFmrEF", "HFpEF"):
            ef_cat = _ef_category(ef)
        nyha = int(outcome.get("nyha_class", 2)) if not pd.isna(outcome.get("nyha_class")) else 2
        n_comorbidities = int(outcome.get("n_comorbidities", 0)) if not pd.isna(outcome.get("n_comorbidities")) else 0
        n_medications = int(outcome.get("n_medications", 0)) if not pd.isna(outcome.get("n_medications")) else 0

        patient_context = {
            "age": age,
            "sex_male": int(sex == "M"),
            "ef": ef,
            "nyha_class": nyha,
            "n_comorbidities": n_comorbidities,
            "n_medications": n_medications,
            "ef_reduced": int(ef_cat == "HFrEF"),
            "ef_mid": int(ef_cat == "HFmrEF"),
            "ef_preserved": int(ef_cat == "HFpEF"),
        }

        # --- Run prediction ---
        result = self.predictor.predict(
            p_vitals, p_labs, p_symptoms, patient_context,
        )

        # --- Derive dashboard fields ---
        name = generate_name(pid, sex)
        trend = compute_trend(p_vitals)

        # Anchor timestamps to a recent synthetic "now". Source dates are either
        # de-identified/shifted (MIMIC uses ~2175) or relative to admission (eICU),
        # so we map the monitoring window onto the present using the `day` index:
        # the latest reading is "just now" and admission is `span` days earlier.
        # This keeps the day-to-day spacing while producing sensible display dates.
        now = datetime.now(timezone.utc)
        if "day" in p_vitals.columns and len(p_vitals) > 0:
            span_days = int(p_vitals["day"].max() - p_vitals["day"].min())
        else:
            span_days = 0
        # small deterministic recency (0–6 h) so the panel reads as live
        minutes_ago = _hash_int(pid, "updated") % 360
        last_updated_dt = now - timedelta(minutes=minutes_ago)
        last_updated = last_updated_dt.isoformat()
        admission_date = (last_updated_dt - timedelta(days=span_days)).strftime("%Y-%m-%d")

        # Alerts: derive from risk tier
        risk_tier = result["risk_tier"]
        alerts_total = {"critical": 4, "high": 2, "medium": 1, "low": 0}.get(risk_tier, 0)
        alerts_ack = max(0, alerts_total - _hash_int(pid, "alerts") % (alerts_total + 1)) if alerts_total > 0 else 0

        # --- GDMT medication state ---
        medications = _generate_medication_state(pid, ef_cat)

        # Build feature dict for safety checks (need vitals/labs features)
        from src.features.build_features import compute_vital_features, compute_lab_features
        current_day = p_vitals["day"].max()
        vital_feats = compute_vital_features(p_vitals, current_day)
        lab_feats = compute_lab_features(
            p_labs, current_day, age=age, sex_male=int(sex == "M"),
        ) if len(p_labs) > 0 else {}
        # compute_lab_features already derives eGFR (CKD-EPI, real age/sex) for the
        # MRA/SGLT2i safety checks.
        features = {**vital_feats, **lab_feats, **patient_context}

        # Get tolerance prediction
        tolerance = self.predictor.predict_tolerance(
            p_vitals, p_labs, p_symptoms, patient_context,
        )

        # Generate GDMT recommendations
        gdmt_recommendations = []
        for med in medications:
            if med["status"] == "not_indicated":
                gdmt_recommendations.append({
                    "drug_class": med["drug_class"],
                    "generic_name": med["generic_name"],
                    "action": "not_indicated",
                    "current_dose_mg": 0,
                    "next_dose_mg": None,
                    "target_dose_mg": med["target_dose_mg"],
                    "safety_checks": [],
                    "tolerance_score": None,
                    "tolerance_factors": None,
                    "rationale": f"Not indicated for {ef_cat}.",
                })
                continue

            rec = self.gdmt_engine.evaluate_titration(
                med["drug_class"], med, features,
            )
            # Attach tolerance score to actionable recommendations
            if rec["action"] in ("uptitrate", "initiate"):
                rec["tolerance_score"] = tolerance["tolerance_score"]
                rec["tolerance_factors"] = tolerance["tolerance_factors"]
            else:
                rec["tolerance_score"] = None
                rec["tolerance_factors"] = None
            gdmt_recommendations.append(rec)

        optimization_score = GDMTEngine.compute_optimization_score(medications)

        return {
            "id": pid,
            "name": name,
            "age": age,
            "sex": sex,
            "room": "Home",
            "ejection_fraction": round(ef, 1),
            "ef_category": ef_cat,
            "nyha_class": nyha,
            "admission_date": admission_date,
            "diagnosis": _diagnosis_text(ef_cat, nyha),
            "risk_score": result["risk_score"],
            "risk_tier": risk_tier,
            "trend": trend,
            "vitals": format_vitals(_fill_vitals_for_display(p_vitals)),
            "labs": format_labs(p_labs),
            "top_factors": result["top_factors"],
            "suggested_action": result["suggested_action"],
            "last_updated": last_updated,
            "alerts_acknowledged": alerts_ack,
            "alerts_total": alerts_total,
            "medications": medications,
            "gdmt_recommendations": gdmt_recommendations,
            "optimization_score": optimization_score,
        }

    def get_all(self) -> list[dict]:
        """Return all patients sorted by risk_score descending."""
        return sorted(
            self.patients.values(),
            key=lambda p: p["risk_score"],
            reverse=True,
        )

    def get(self, patient_id: str) -> dict | None:
        return self.patients.get(patient_id)
