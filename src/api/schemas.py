"""Pydantic schemas for the prediction API."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# GDMT (Guideline-Directed Medical Therapy) schemas
# ---------------------------------------------------------------------------

class Medication(BaseModel):
    drug_class: str  # "raasi", "beta_blocker", "mra", "sglt2i"
    generic_name: str
    current_dose_mg: float
    target_dose_mg: float
    status: str  # "not_started", "sub_therapeutic", "at_target", "not_indicated"


class TitrationSafetyCheck(BaseModel):
    check_name: str
    passed: bool
    current_value: Optional[float] = None
    threshold: Optional[float] = None


class TitrationRecommendation(BaseModel):
    drug_class: str
    generic_name: str
    action: str  # "uptitrate", "hold", "initiate", "at_target", "not_indicated"
    current_dose_mg: float
    next_dose_mg: Optional[float] = None
    target_dose_mg: float
    safety_checks: list[TitrationSafetyCheck]
    tolerance_score: Optional[float] = None
    tolerance_factors: Optional[list[dict]] = None
    rationale: str


class GDMTStatusResponse(BaseModel):
    patient_id: str
    ef_category: str
    optimization_score: float  # 0-100, % of target doses achieved
    medications: list[Medication]
    recommendations: list[TitrationRecommendation]
    timestamp: datetime


class VitalSign(BaseModel):
    timestamp: datetime
    weight_kg: float = Field(..., ge=30, le=300)
    spo2: float = Field(..., ge=50, le=100)
    heart_rate: int = Field(..., ge=30, le=250)
    systolic_bp: int = Field(..., ge=50, le=300)
    diastolic_bp: int = Field(..., ge=30, le=200)
    respiratory_rate: int = Field(..., ge=4, le=60)


class LabResult(BaseModel):
    timestamp: datetime
    bnp_pg_ml: float = Field(..., ge=0)
    creatinine_mg_dl: float = Field(..., ge=0)
    potassium_meq_l: float = Field(..., ge=1.0, le=9.0)


class SymptomReport(BaseModel):
    timestamp: datetime
    dyspnea_score: Optional[float] = Field(None, ge=0, le=10)
    orthopnea_pillows: Optional[int] = Field(None, ge=0, le=4)
    ankle_edema: Optional[int] = Field(None, ge=0, le=3)
    exercise_tolerance: Optional[float] = Field(None, ge=0, le=10)
    medication_adherent: Optional[int] = Field(None, ge=0, le=1)


class PatientContext(BaseModel):
    age: int = Field(..., ge=18, le=120)
    sex: str = Field(..., pattern="^(M|F)$")
    ejection_fraction: float = Field(..., ge=5, le=80)
    ef_category: str = Field(..., pattern="^(HFrEF|HFmrEF|HFpEF)$")
    nyha_class: int = Field(..., ge=1, le=4)
    n_comorbidities: int = Field(..., ge=0)
    n_medications: int = Field(..., ge=0)


class PredictionRequest(BaseModel):
    patient_id: str
    patient: PatientContext
    vitals: list[VitalSign] = Field(..., min_length=2)
    labs: list[LabResult] = Field(default_factory=list)
    symptoms: list[SymptomReport] = Field(default_factory=list)


class ContributingFactor(BaseModel):
    feature: str
    display_name: str
    value: float
    impact: float  # SHAP value
    direction: str  # "increasing_risk" or "decreasing_risk"


class SuggestedAction(BaseModel):
    action: str
    urgency: str
    rationale: str


class PredictionResponse(BaseModel):
    patient_id: str
    risk_score: float = Field(..., ge=0, le=100)
    risk_tier: str
    probability: float
    top_factors: list[ContributingFactor]
    suggested_action: SuggestedAction
    timestamp: datetime
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    model_auroc: float
    version: str
