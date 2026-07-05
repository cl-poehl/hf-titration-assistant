export type RiskTier = "low" | "medium" | "high" | "critical";

// ---------------------------------------------------------------------------
// GDMT types
// ---------------------------------------------------------------------------
export type GDMTDrugClass = "raasi" | "beta_blocker" | "mra" | "sglt2i";

export type TitrationStatus =
  | "not_started"
  | "sub_therapeutic"
  | "at_target"
  | "not_indicated";

export interface Medication {
  drug_class: GDMTDrugClass;
  generic_name: string;
  current_dose_mg: number;
  target_dose_mg: number;
  status: TitrationStatus;
}

export interface TitrationSafetyCheck {
  check_name: string;
  passed: boolean;
  current_value: number | null;
  threshold: number | null;
}

export interface TitrationRecommendation {
  drug_class: GDMTDrugClass;
  generic_name: string;
  action: "uptitrate" | "hold" | "initiate" | "at_target" | "not_indicated";
  current_dose_mg: number;
  next_dose_mg: number | null;
  target_dose_mg: number;
  safety_checks: TitrationSafetyCheck[];
  tolerance_score: number | null;
  tolerance_factors: ContributingFactor[] | null;
  rationale: string;
}

export interface VitalReading {
  timestamp: string;
  day: number;
  weight_kg: number;
  spo2: number;
  heart_rate: number;
  systolic_bp: number;
  diastolic_bp: number;
  respiratory_rate: number;
}

export interface LabReading {
  timestamp: string;
  day: number;
  // null when the lab was not measured that day (rendered as "—")
  bnp_pg_ml: number | null;
  creatinine_mg_dl: number | null;
  potassium_meq_l: number | null;
}

export interface ContributingFactor {
  feature: string;
  display_name: string;
  value: number;
  impact: number;
  direction: "increasing_risk" | "decreasing_risk";
}

export interface SuggestedAction {
  action: string;
  urgency: string;
  rationale: string;
}

export interface Patient {
  id: string;
  name: string;
  age: number;
  sex: string;
  room: string;
  ejection_fraction: number;
  ef_category: string;
  nyha_class: number;
  admission_date: string;
  diagnosis: string;
  risk_score: number;
  risk_tier: RiskTier;
  trend: "rising" | "falling" | "stable";
  vitals: VitalReading[];
  labs: LabReading[];
  top_factors: ContributingFactor[];
  suggested_action: SuggestedAction;
  last_updated: string;
  alerts_acknowledged: number;
  alerts_total: number;
  // GDMT fields (optional — present when backend provides them or in mock data)
  medications?: Medication[];
  gdmt_recommendations?: TitrationRecommendation[];
  optimization_score?: number;
}
