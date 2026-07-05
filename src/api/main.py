"""
Hospital-at-Home Heart Failure Prediction API

Endpoints:
- GET  /health                — Service health check
- GET  /patients              — All patients with live risk predictions
- GET  /patients/{id}         — Single patient detail
- GET  /patients/{id}/gdmt    — GDMT titration status and recommendations
- POST /predict               — Single patient risk prediction
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.api.predictor import HFPredictor, GDMTEngine
from src.api.patient_registry import PatientRegistry
from src.api.schemas import (
    GDMTStatusResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH_REAL = PROJECT_ROOT / "models" / "best_model_real.pkl"
MODEL_PATH_FALLBACK = PROJECT_ROOT / "models" / "best_model.pkl"
MODEL_PATH = MODEL_PATH_REAL if MODEL_PATH_REAL.exists() else MODEL_PATH_FALLBACK
DATA_DIR = PROJECT_ROOT / "data" / "combined"

predictor: HFPredictor = None
registry: PatientRegistry = None
gdmt_engine: GDMTEngine = None

# CORS: default to the local Vite dev origins. Override with a comma-separated
# HFTA_CORS_ORIGINS env var when deploying behind a real frontend host.
CORS_ORIGINS = [
    o.strip() for o in os.environ.get(
        "HFTA_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",") if o.strip()
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor, registry, gdmt_engine
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found at {MODEL_PATH}")
    predictor = HFPredictor(str(MODEL_PATH))
    gdmt_engine = GDMTEngine(predictor)
    logger.info("Loaded model: %s (AUROC=%.4f)", predictor.model_name, predictor.metrics["auroc"])

    # Load patient registry from parquet data
    if DATA_DIR.exists():
        registry = PatientRegistry(str(DATA_DIR), predictor)
        count = registry.load()
        logger.info("Patient registry ready: %d patients", count)
    else:
        logger.warning("Data dir %s not found — /patients endpoints will return empty", DATA_DIR)
    yield


app = FastAPI(
    title="HF Titration Assistant — Heart Failure Risk API",
    description="Deterioration-risk scoring for hospital-at-home heart-failure patients (research prototype)",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=predictor is not None,
        model_name=predictor.model_name if predictor else "none",
        model_auroc=predictor.metrics["auroc"] if predictor else 0.0,
        version=predictor.version if predictor else "0.0.0",
    )


@app.get("/patients")
def list_patients():
    """Return all patients with risk predictions for the dashboard."""
    if registry is None:
        return {"patients": [], "count": 0}
    patients = registry.get_all()
    return {"patients": patients, "count": len(patients)}


@app.get("/patients/{patient_id}")
def get_patient(patient_id: str):
    """Return a single patient by ID."""
    if registry is None:
        raise HTTPException(status_code=503, detail="Patient registry not loaded")
    patient = registry.get(patient_id)
    if patient is None:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    return patient


@app.get("/patients/{patient_id}/gdmt", response_model=GDMTStatusResponse)
def get_patient_gdmt(patient_id: str):
    """Return GDMT titration status and recommendations for a patient."""
    if registry is None:
        raise HTTPException(status_code=503, detail="Patient registry not loaded")
    patient = registry.get(patient_id)
    if patient is None:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")

    return GDMTStatusResponse(
        patient_id=patient_id,
        ef_category=patient["ef_category"],
        optimization_score=patient.get("optimization_score", 0.0),
        medications=patient.get("medications", []),
        recommendations=patient.get("gdmt_recommendations", []),
        timestamp=datetime.now(timezone.utc),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert API input to DataFrames matching our feature pipeline format
        vitals_records = []
        start_time = min(v.timestamp for v in request.vitals)

        for v in request.vitals:
            day = (v.timestamp - start_time).days
            hour = v.timestamp.hour
            vitals_records.append({
                "patient_id": request.patient_id,
                "timestamp": v.timestamp,
                "day": day,
                "measurement": "morning" if hour < 14 else "evening",
                "weight_kg": v.weight_kg,
                "spo2": v.spo2,
                "heart_rate": v.heart_rate,
                "systolic_bp": v.systolic_bp,
                "diastolic_bp": v.diastolic_bp,
                "respiratory_rate": v.respiratory_rate,
            })
        vitals_df = pd.DataFrame(vitals_records)

        # Labs
        labs_records = []
        for lab in request.labs:
            day = (lab.timestamp - start_time).days
            labs_records.append({
                "patient_id": request.patient_id,
                "timestamp": lab.timestamp,
                "day": day,
                "bnp_pg_ml": lab.bnp_pg_ml,
                "creatinine_mg_dl": lab.creatinine_mg_dl,
                "potassium_meq_l": lab.potassium_meq_l,
            })
        labs_df = pd.DataFrame(labs_records) if labs_records else pd.DataFrame()

        # Symptoms
        symptom_records = []
        for s in request.symptoms:
            day = (s.timestamp - start_time).days
            reported = any([
                s.dyspnea_score is not None,
                s.orthopnea_pillows is not None,
                s.ankle_edema is not None,
                s.exercise_tolerance is not None,
            ])
            symptom_records.append({
                "patient_id": request.patient_id,
                "timestamp": s.timestamp,
                "day": day,
                "reported": reported,
                "dyspnea_score": s.dyspnea_score,
                "orthopnea_pillows": s.orthopnea_pillows,
                "ankle_edema": s.ankle_edema,
                "exercise_tolerance": s.exercise_tolerance,
                "medication_adherent": s.medication_adherent,
            })
        symptoms_df = pd.DataFrame(symptom_records) if symptom_records else pd.DataFrame()

        # Patient context
        patient_context = {
            "age": request.patient.age,
            "sex_male": int(request.patient.sex == "M"),
            "ef": request.patient.ejection_fraction,
            "nyha_class": request.patient.nyha_class,
            "n_comorbidities": request.patient.n_comorbidities,
            "n_medications": request.patient.n_medications,
            "ef_reduced": int(request.patient.ef_category == "HFrEF"),
            "ef_mid": int(request.patient.ef_category == "HFmrEF"),
            "ef_preserved": int(request.patient.ef_category == "HFpEF"),
        }

        result = predictor.predict(vitals_df, labs_df, symptoms_df, patient_context)

        return PredictionResponse(
            patient_id=request.patient_id,
            risk_score=result["risk_score"],
            risk_tier=result["risk_tier"],
            probability=result["probability"],
            top_factors=result["top_factors"],
            suggested_action=result["suggested_action"],
            timestamp=datetime.now(timezone.utc),
            model_version=result["model_version"],
        )

    except Exception:
        # Log the full error server-side; return a generic message to the client
        # so internal details (paths, stack context) are not leaked.
        logger.exception("Prediction failed for patient %s", request.patient_id)
        raise HTTPException(
            status_code=422,
            detail="Could not compute a prediction from the supplied data.",
        )
