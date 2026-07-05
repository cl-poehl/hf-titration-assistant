import type { Patient, VitalReading, LabReading } from "../types/patient";

// Seeded PRNG (mulberry32) so the mock dashboard renders identical sample data
// on every load — deterministic demo data reads as deliberate, not jittery.
let _seed = 0x9e3779b9;
function rand(): number {
  _seed |= 0;
  _seed = (_seed + 0x6d2b79f5) | 0;
  let t = Math.imul(_seed ^ (_seed >>> 15), 1 | _seed);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}

function generateVitals(
  days: number,
  baseWeight: number,
  baseSpo2: number,
  baseHr: number,
  baseSbp: number,
  baseDbp: number,
  baseRr: number,
  deteriorateFrom?: number
): VitalReading[] {
  const vitals: VitalReading[] = [];
  for (let d = 0; d < days; d++) {
    for (const hour of [8, 20]) {
      const det =
        deteriorateFrom !== undefined && d >= deteriorateFrom
          ? Math.min((d - deteriorateFrom) / 4, 1)
          : 0;
      const ts = new Date(2026, 0, 15 + d, hour);
      vitals.push({
        timestamp: ts.toISOString(),
        day: d,
        weight_kg: +(baseWeight + (rand() - 0.5) * 0.6 + det * (2 + rand() * 3)).toFixed(1),
        spo2: +Math.min(100, Math.max(82, baseSpo2 + (rand() - 0.5) - det * (3 + rand() * 4))).toFixed(1),
        heart_rate: Math.round(baseHr + (rand() - 0.5) * 8 + det * (15 + rand() * 15)),
        systolic_bp: Math.round(baseSbp + (rand() - 0.5) * 10 - det * (8 + rand() * 12)),
        diastolic_bp: Math.round(baseDbp + (rand() - 0.5) * 6 - det * (4 + rand() * 6)),
        respiratory_rate: Math.round(baseRr + (rand() - 0.5) * 2 + det * (4 + rand() * 6)),
      });
    }
  }
  return vitals;
}

function generateLabs(
  days: number,
  baseBnp: number,
  baseCreat: number,
  baseK: number,
  deteriorateFrom?: number
): LabReading[] {
  const labs: LabReading[] = [];
  for (let d = 0; d < days; d += 7) {
    const det =
      deteriorateFrom !== undefined && d >= deteriorateFrom
        ? Math.min((d - deteriorateFrom) / 4, 1)
        : 0;
    const ts = new Date(2026, 0, 15 + d, 10);
    labs.push({
      timestamp: ts.toISOString(),
      day: d,
      bnp_pg_ml: +(baseBnp * (1 + (rand() - 0.5) * 0.1 + det * 2.5)).toFixed(1),
      creatinine_mg_dl: +(baseCreat + (rand() - 0.5) * 0.05 + det * 0.5).toFixed(2),
      potassium_meq_l: +(baseK + (rand() - 0.5) * 0.2 + det * 0.4).toFixed(1),
    });
  }
  return labs;
}

export const mockPatients: Patient[] = [
  {
    id: "HF-00142",
    name: "Margaret Chen",
    age: 78,
    sex: "F",
    room: "Home — 4521 Elm St",
    ejection_fraction: 28,
    ef_category: "HFrEF",
    nyha_class: 3,
    admission_date: "2026-01-15",
    diagnosis: "Acute decompensated HF, HFrEF",
    risk_score: 84.2,
    risk_tier: "critical",
    trend: "rising",
    vitals: generateVitals(14, 72, 95, 82, 118, 68, 18, 10),
    labs: generateLabs(14, 580, 1.4, 4.1, 10),
    top_factors: [
      { feature: "weight_change_total_kg", display_name: "Total weight change", value: 3.8, impact: 1.32, direction: "increasing_risk" },
      { feature: "spo2_trend_3d", display_name: "SpO2 trend (3-day)", value: -0.8, impact: 0.95, direction: "increasing_risk" },
      { feature: "bnp_trend", display_name: "BNP trend", value: 420, impact: 0.78, direction: "increasing_risk" },
      { feature: "dyspnea_trend", display_name: "Dyspnea worsening", value: 2.1, impact: 0.61, direction: "increasing_risk" },
      { feature: "creat_trend", display_name: "Creatinine trend", value: 0.3, impact: 0.42, direction: "increasing_risk" },
    ],
    suggested_action: {
      action: "Immediate physician notification. Consider ED transfer if symptoms worsen. Evaluate for inpatient readmission.",
      urgency: "emergent",
      rationale: "High probability of acute decompensation. Weight gain of 3.8 kg over 4 days with declining SpO2 and rising BNP indicate fluid overload.",
    },
    last_updated: "2026-01-29T08:32:00Z",
    alerts_acknowledged: 1,
    alerts_total: 3,
    medications: [
      { drug_class: "raasi", generic_name: "sacubitril_valsartan", current_dose_mg: 49, target_dose_mg: 97, status: "sub_therapeutic" },
      { drug_class: "beta_blocker", generic_name: "carvedilol", current_dose_mg: 12.5, target_dose_mg: 25, status: "sub_therapeutic" },
      { drug_class: "mra", generic_name: "spironolactone", current_dose_mg: 25, target_dose_mg: 50, status: "sub_therapeutic" },
      { drug_class: "sglt2i", generic_name: "dapagliflozin", current_dose_mg: 10, target_dose_mg: 10, status: "at_target" },
    ],
    gdmt_recommendations: [
      { drug_class: "raasi", generic_name: "sacubitril_valsartan", action: "hold", current_dose_mg: 49, next_dose_mg: null, target_dose_mg: 97, safety_checks: [
        { check_name: "SBP >= 100", passed: false, current_value: 96, threshold: 100 },
        { check_name: "K+ < 5.0", passed: true, current_value: 4.1, threshold: 5.0 },
        { check_name: "Creatinine stable", passed: false, current_value: 0.5, threshold: 0.3 },
      ], tolerance_score: null, tolerance_factors: null, rationale: "Hold at 49 mg — failed: SBP >= 100, Creatinine stable." },
      { drug_class: "beta_blocker", generic_name: "carvedilol", action: "hold", current_dose_mg: 12.5, next_dose_mg: null, target_dose_mg: 25, safety_checks: [
        { check_name: "HR >= 60", passed: true, current_value: 88, threshold: 60 },
        { check_name: "SBP >= 90", passed: true, current_value: 96, threshold: 90 },
        { check_name: "No acute decompensation", passed: false, current_value: 0.72, threshold: 0.6 },
      ], tolerance_score: null, tolerance_factors: null, rationale: "Hold at 12.5 mg — failed: No acute decompensation." },
      { drug_class: "mra", generic_name: "spironolactone", action: "hold", current_dose_mg: 25, next_dose_mg: null, target_dose_mg: 50, safety_checks: [
        { check_name: "K+ < 5.0", passed: true, current_value: 4.1, threshold: 5.0 },
        { check_name: "eGFR >= 30", passed: true, current_value: 42, threshold: 30 },
        { check_name: "No AKI", passed: false, current_value: 0.5, threshold: 0.3 },
      ], tolerance_score: null, tolerance_factors: null, rationale: "Hold at 25 mg — failed: No AKI." },
      { drug_class: "sglt2i", generic_name: "dapagliflozin", action: "at_target", current_dose_mg: 10, next_dose_mg: null, target_dose_mg: 10, safety_checks: [
        { check_name: "eGFR >= 20", passed: true, current_value: 42, threshold: 20 },
      ], tolerance_score: null, tolerance_factors: null, rationale: "Already at target dose (10 mg). Continue current regimen." },
    ],
    optimization_score: 52.8,
  },
  {
    id: "HF-00087",
    name: "Robert Williams",
    age: 65,
    sex: "M",
    room: "Home — 892 Oak Ave",
    ejection_fraction: 35,
    ef_category: "HFrEF",
    nyha_class: 3,
    admission_date: "2026-01-18",
    diagnosis: "HFrEF exacerbation, CAD",
    risk_score: 62.1,
    risk_tier: "high",
    trend: "rising",
    vitals: generateVitals(11, 95, 96, 76, 130, 78, 16, 8),
    labs: generateLabs(11, 350, 1.3, 4.3, 8),
    top_factors: [
      { feature: "weightkg_trend_3d", display_name: "Weight trend (3-day)", value: 0.45, impact: 0.88, direction: "increasing_risk" },
      { feature: "symptom_burden", display_name: "Symptom burden score", value: 0.72, impact: 0.65, direction: "increasing_risk" },
      { feature: "exercise_trend", display_name: "Exercise tolerance declining", value: -1.2, impact: 0.52, direction: "increasing_risk" },
      { feature: "k_trend", display_name: "Potassium trend", value: 0.3, impact: 0.41, direction: "increasing_risk" },
      { feature: "med_adherence_rate", display_name: "Medication adherence", value: 0.71, impact: 0.35, direction: "increasing_risk" },
    ],
    suggested_action: {
      action: "Alert physician. Consider in-person assessment and diuretic dose adjustment. Review most recent labs.",
      urgency: "urgent",
      rationale: "Multiple deterioration indicators trending abnormally. Weight gain with declining exercise tolerance.",
    },
    last_updated: "2026-01-29T07:15:00Z",
    alerts_acknowledged: 2,
    alerts_total: 2,
    medications: [
      { drug_class: "raasi", generic_name: "lisinopril", current_dose_mg: 10, target_dose_mg: 40, status: "sub_therapeutic" },
      { drug_class: "beta_blocker", generic_name: "metoprolol_succinate", current_dose_mg: 50, target_dose_mg: 200, status: "sub_therapeutic" },
      { drug_class: "mra", generic_name: "spironolactone", current_dose_mg: 0, target_dose_mg: 50, status: "not_started" },
      { drug_class: "sglt2i", generic_name: "empagliflozin", current_dose_mg: 10, target_dose_mg: 10, status: "at_target" },
    ],
    gdmt_recommendations: [
      { drug_class: "raasi", generic_name: "lisinopril", action: "hold", current_dose_mg: 10, next_dose_mg: null, target_dose_mg: 40, safety_checks: [
        { check_name: "SBP >= 100", passed: true, current_value: 118, threshold: 100 },
        { check_name: "K+ < 5.0", passed: true, current_value: 4.3, threshold: 5.0 },
        { check_name: "Creatinine stable", passed: true, current_value: 0.1, threshold: 0.3 },
      ], tolerance_score: null, tolerance_factors: null, rationale: "Hold — patient at high risk. Reassess after stabilization." },
      { drug_class: "beta_blocker", generic_name: "metoprolol_succinate", action: "hold", current_dose_mg: 50, next_dose_mg: null, target_dose_mg: 200, safety_checks: [
        { check_name: "HR >= 60", passed: true, current_value: 76, threshold: 60 },
        { check_name: "SBP >= 90", passed: true, current_value: 118, threshold: 90 },
        { check_name: "No acute decompensation", passed: false, current_value: 0.72, threshold: 0.6 },
      ], tolerance_score: null, tolerance_factors: null, rationale: "Hold at 50 mg — failed: No acute decompensation." },
      { drug_class: "mra", generic_name: "spironolactone", action: "hold", current_dose_mg: 0, next_dose_mg: null, target_dose_mg: 50, safety_checks: [
        { check_name: "K+ < 5.0", passed: true, current_value: 4.3, threshold: 5.0 },
        { check_name: "eGFR >= 30", passed: true, current_value: 55, threshold: 30 },
        { check_name: "No AKI", passed: true, current_value: 0.1, threshold: 0.3 },
      ], tolerance_score: null, tolerance_factors: null, rationale: "Hold initiation — patient at high risk. Reassess after stabilization." },
      { drug_class: "sglt2i", generic_name: "empagliflozin", action: "at_target", current_dose_mg: 10, next_dose_mg: null, target_dose_mg: 10, safety_checks: [
        { check_name: "eGFR >= 20", passed: true, current_value: 55, threshold: 20 },
      ], tolerance_score: null, tolerance_factors: null, rationale: "Already at target dose (10 mg). Continue current regimen." },
    ],
    optimization_score: 35.0,
  },
  {
    id: "HF-00203",
    name: "Dorothy Martinez",
    age: 82,
    sex: "F",
    room: "Home — 156 Pine Rd",
    ejection_fraction: 55,
    ef_category: "HFpEF",
    nyha_class: 2,
    admission_date: "2026-01-20",
    diagnosis: "HFpEF, hypertension, T2DM",
    risk_score: 38.5,
    risk_tier: "medium",
    trend: "rising",
    vitals: generateVitals(9, 68, 97, 72, 142, 82, 15, 7),
    labs: generateLabs(9, 210, 1.1, 4.5),
    top_factors: [
      { feature: "systolicbp_trend_3d", display_name: "Systolic BP trend", value: 0.32, impact: 0.55, direction: "increasing_risk" },
      { feature: "weightkg_baseline_dev", display_name: "Weight from baseline", value: 1.2, impact: 0.42, direction: "increasing_risk" },
      { feature: "orthopnea_trend", display_name: "Orthopnea worsening", value: 0.5, impact: 0.31, direction: "increasing_risk" },
      { feature: "age", display_name: "Age", value: 82, impact: 0.22, direction: "increasing_risk" },
      { feature: "spo2_current", display_name: "Current SpO2", value: 95.2, impact: -0.18, direction: "decreasing_risk" },
    ],
    suggested_action: {
      action: "Schedule phone check-in within 4 hours. Review medication adherence and fluid intake.",
      urgency: "soon",
      rationale: "Early warning signals detected. BP trending up with mild weight gain.",
    },
    last_updated: "2026-01-29T08:00:00Z",
    alerts_acknowledged: 0,
    alerts_total: 1,
    medications: [
      { drug_class: "raasi", generic_name: "lisinopril", current_dose_mg: 0, target_dose_mg: 40, status: "not_indicated" },
      { drug_class: "beta_blocker", generic_name: "carvedilol", current_dose_mg: 0, target_dose_mg: 25, status: "not_indicated" },
      { drug_class: "mra", generic_name: "spironolactone", current_dose_mg: 0, target_dose_mg: 50, status: "not_indicated" },
      { drug_class: "sglt2i", generic_name: "dapagliflozin", current_dose_mg: 10, target_dose_mg: 10, status: "at_target" },
    ],
    gdmt_recommendations: [
      { drug_class: "raasi", generic_name: "lisinopril", action: "not_indicated", current_dose_mg: 0, next_dose_mg: null, target_dose_mg: 40, safety_checks: [], tolerance_score: null, tolerance_factors: null, rationale: "Not indicated for HFpEF." },
      { drug_class: "beta_blocker", generic_name: "carvedilol", action: "not_indicated", current_dose_mg: 0, next_dose_mg: null, target_dose_mg: 25, safety_checks: [], tolerance_score: null, tolerance_factors: null, rationale: "Not indicated for HFpEF." },
      { drug_class: "mra", generic_name: "spironolactone", action: "not_indicated", current_dose_mg: 0, next_dose_mg: null, target_dose_mg: 50, safety_checks: [], tolerance_score: null, tolerance_factors: null, rationale: "Not indicated for HFpEF." },
      { drug_class: "sglt2i", generic_name: "dapagliflozin", action: "at_target", current_dose_mg: 10, next_dose_mg: null, target_dose_mg: 10, safety_checks: [
        { check_name: "eGFR >= 20", passed: true, current_value: 68, threshold: 20 },
      ], tolerance_score: null, tolerance_factors: null, rationale: "Already at target dose (10 mg). Continue current regimen." },
    ],
    optimization_score: 100.0,
  },
  {
    id: "HF-00315",
    name: "James Thompson",
    age: 70,
    sex: "M",
    room: "Home — 2200 Maple Dr",
    ejection_fraction: 42,
    ef_category: "HFmrEF",
    nyha_class: 2,
    admission_date: "2026-01-22",
    diagnosis: "HFmrEF, atrial fibrillation",
    risk_score: 12.3,
    risk_tier: "low",
    trend: "stable",
    vitals: generateVitals(7, 88, 97, 68, 125, 74, 14),
    labs: generateLabs(7, 180, 1.0, 4.2),
    top_factors: [
      { feature: "ef", display_name: "Ejection fraction", value: 42, impact: 0.15, direction: "increasing_risk" },
      { feature: "age", display_name: "Age", value: 70, impact: 0.12, direction: "increasing_risk" },
      { feature: "weightkg_baseline_dev", display_name: "Weight from baseline", value: -0.2, impact: -0.25, direction: "decreasing_risk" },
      { feature: "spo2_current", display_name: "Current SpO2", value: 97.1, impact: -0.30, direction: "decreasing_risk" },
      { feature: "bnp_change", display_name: "BNP change", value: -15, impact: -0.22, direction: "decreasing_risk" },
    ],
    suggested_action: {
      action: "Continue routine monitoring. No immediate intervention needed.",
      urgency: "routine",
      rationale: "Vital signs and symptoms within expected range. Patient is stable.",
    },
    last_updated: "2026-01-29T08:30:00Z",
    alerts_acknowledged: 0,
    alerts_total: 0,
    medications: [
      { drug_class: "raasi", generic_name: "losartan", current_dose_mg: 50, target_dose_mg: 150, status: "sub_therapeutic" },
      { drug_class: "beta_blocker", generic_name: "metoprolol_succinate", current_dose_mg: 25, target_dose_mg: 200, status: "sub_therapeutic" },
      { drug_class: "mra", generic_name: "spironolactone", current_dose_mg: 0, target_dose_mg: 50, status: "not_indicated" },
      { drug_class: "sglt2i", generic_name: "empagliflozin", current_dose_mg: 0, target_dose_mg: 10, status: "not_started" },
    ],
    gdmt_recommendations: [
      { drug_class: "raasi", generic_name: "losartan", action: "uptitrate", current_dose_mg: 50, next_dose_mg: 100, target_dose_mg: 150, safety_checks: [
        { check_name: "SBP >= 100", passed: true, current_value: 125, threshold: 100 },
        { check_name: "K+ < 5.0", passed: true, current_value: 4.2, threshold: 5.0 },
        { check_name: "Creatinine stable", passed: true, current_value: 0.05, threshold: 0.3 },
      ], tolerance_score: 87.7, tolerance_factors: null, rationale: "All safety checks passed. Increase to 100 mg (target: 150 mg)." },
      { drug_class: "beta_blocker", generic_name: "metoprolol_succinate", action: "uptitrate", current_dose_mg: 25, next_dose_mg: 50, target_dose_mg: 200, safety_checks: [
        { check_name: "HR >= 60", passed: true, current_value: 68, threshold: 60 },
        { check_name: "SBP >= 90", passed: true, current_value: 125, threshold: 90 },
        { check_name: "No acute decompensation", passed: true, current_value: 0.15, threshold: 0.6 },
      ], tolerance_score: 87.7, tolerance_factors: null, rationale: "All safety checks passed. Increase to 50 mg (target: 200 mg)." },
      { drug_class: "mra", generic_name: "spironolactone", action: "not_indicated", current_dose_mg: 0, next_dose_mg: null, target_dose_mg: 50, safety_checks: [], tolerance_score: null, tolerance_factors: null, rationale: "Not indicated for HFmrEF." },
      { drug_class: "sglt2i", generic_name: "empagliflozin", action: "initiate", current_dose_mg: 0, next_dose_mg: 10, target_dose_mg: 10, safety_checks: [
        { check_name: "eGFR >= 20", passed: true, current_value: 72, threshold: 20 },
      ], tolerance_score: 87.7, tolerance_factors: null, rationale: "All safety checks passed. Initiate empagliflozin at 10 mg." },
    ],
    optimization_score: 20.8,
  },
  {
    id: "HF-00198",
    name: "Patricia Davis",
    age: 74,
    sex: "F",
    room: "Home — 710 Cedar Ln",
    ejection_fraction: 25,
    ef_category: "HFrEF",
    nyha_class: 3,
    admission_date: "2026-01-17",
    diagnosis: "HFrEF, CKD Stage 3, anemia",
    risk_score: 8.7,
    risk_tier: "low",
    trend: "falling",
    vitals: generateVitals(12, 64, 96, 74, 112, 66, 16),
    labs: generateLabs(12, 290, 1.5, 4.0),
    top_factors: [
      { feature: "bnp_change", display_name: "BNP change", value: -80, impact: -0.55, direction: "decreasing_risk" },
      { feature: "weight_change_total_kg", display_name: "Weight change", value: -0.8, impact: -0.42, direction: "decreasing_risk" },
      { feature: "exercise_trend", display_name: "Exercise tolerance improving", value: 0.8, impact: -0.35, direction: "decreasing_risk" },
      { feature: "creat_current", display_name: "Creatinine", value: 1.5, impact: 0.20, direction: "increasing_risk" },
      { feature: "ef", display_name: "Ejection fraction", value: 25, impact: 0.18, direction: "increasing_risk" },
    ],
    suggested_action: {
      action: "Continue routine monitoring. No immediate intervention needed.",
      urgency: "routine",
      rationale: "Patient improving. BNP trending down, weight stable, exercise tolerance increasing.",
    },
    last_updated: "2026-01-29T07:45:00Z",
    alerts_acknowledged: 1,
    alerts_total: 1,
    medications: [
      { drug_class: "raasi", generic_name: "sacubitril_valsartan", current_dose_mg: 97, target_dose_mg: 97, status: "at_target" },
      { drug_class: "beta_blocker", generic_name: "carvedilol", current_dose_mg: 25, target_dose_mg: 25, status: "at_target" },
      { drug_class: "mra", generic_name: "spironolactone", current_dose_mg: 50, target_dose_mg: 50, status: "at_target" },
      { drug_class: "sglt2i", generic_name: "dapagliflozin", current_dose_mg: 10, target_dose_mg: 10, status: "at_target" },
    ],
    gdmt_recommendations: [
      { drug_class: "raasi", generic_name: "sacubitril_valsartan", action: "at_target", current_dose_mg: 97, next_dose_mg: null, target_dose_mg: 97, safety_checks: [
        { check_name: "SBP >= 100", passed: true, current_value: 112, threshold: 100 },
        { check_name: "K+ < 5.0", passed: true, current_value: 4.0, threshold: 5.0 },
        { check_name: "Creatinine stable", passed: true, current_value: 0.0, threshold: 0.3 },
      ], tolerance_score: null, tolerance_factors: null, rationale: "Already at target dose (97 mg). Continue current regimen." },
      { drug_class: "beta_blocker", generic_name: "carvedilol", action: "at_target", current_dose_mg: 25, next_dose_mg: null, target_dose_mg: 25, safety_checks: [
        { check_name: "HR >= 60", passed: true, current_value: 74, threshold: 60 },
        { check_name: "SBP >= 90", passed: true, current_value: 112, threshold: 90 },
        { check_name: "No acute decompensation", passed: true, current_value: 0.1, threshold: 0.6 },
      ], tolerance_score: null, tolerance_factors: null, rationale: "Already at target dose (25 mg). Continue current regimen." },
      { drug_class: "mra", generic_name: "spironolactone", action: "at_target", current_dose_mg: 50, next_dose_mg: null, target_dose_mg: 50, safety_checks: [
        { check_name: "K+ < 5.0", passed: true, current_value: 4.0, threshold: 5.0 },
        { check_name: "eGFR >= 30", passed: true, current_value: 38, threshold: 30 },
        { check_name: "No AKI", passed: true, current_value: 0.0, threshold: 0.3 },
      ], tolerance_score: null, tolerance_factors: null, rationale: "Already at target dose (50 mg). Continue current regimen." },
      { drug_class: "sglt2i", generic_name: "dapagliflozin", action: "at_target", current_dose_mg: 10, next_dose_mg: null, target_dose_mg: 10, safety_checks: [
        { check_name: "eGFR >= 20", passed: true, current_value: 38, threshold: 20 },
      ], tolerance_score: null, tolerance_factors: null, rationale: "Already at target dose (10 mg). Continue current regimen." },
    ],
    optimization_score: 100.0,
  },
  {
    id: "HF-00421",
    name: "Harold Kim",
    age: 68,
    sex: "M",
    room: "Home — 33 Birch Ct",
    ejection_fraction: 30,
    ef_category: "HFrEF",
    nyha_class: 4,
    admission_date: "2026-01-12",
    diagnosis: "Severe HFrEF, COPD, sleep apnea",
    risk_score: 91.7,
    risk_tier: "critical",
    trend: "rising",
    vitals: generateVitals(17, 102, 93, 88, 105, 62, 20, 13),
    labs: generateLabs(17, 820, 1.8, 3.8, 13),
    top_factors: [
      { feature: "weight_change_total_kg", display_name: "Total weight change", value: 5.2, impact: 1.65, direction: "increasing_risk" },
      { feature: "spo2_baseline_dev", display_name: "SpO2 change from baseline", value: -5.1, impact: 1.22, direction: "increasing_risk" },
      { feature: "weightkg_trend_3d", display_name: "Weight trend (3-day)", value: 1.1, impact: 1.05, direction: "increasing_risk" },
      { feature: "bnp_current", display_name: "Current BNP", value: 2150, impact: 0.88, direction: "increasing_risk" },
      { feature: "respiratoryrate_current", display_name: "Respiratory rate", value: 28, impact: 0.72, direction: "increasing_risk" },
    ],
    suggested_action: {
      action: "Immediate physician notification. Consider ED transfer if symptoms worsen. Evaluate for inpatient readmission.",
      urgency: "emergent",
      rationale: "Severe decompensation likely. 5.2 kg weight gain, SpO2 declining to 88%, BNP > 2000. High risk of respiratory failure.",
    },
    last_updated: "2026-01-29T08:45:00Z",
    alerts_acknowledged: 0,
    alerts_total: 4,
    medications: [
      { drug_class: "raasi", generic_name: "lisinopril", current_dose_mg: 5, target_dose_mg: 40, status: "sub_therapeutic" },
      { drug_class: "beta_blocker", generic_name: "carvedilol", current_dose_mg: 3.125, target_dose_mg: 25, status: "sub_therapeutic" },
      { drug_class: "mra", generic_name: "eplerenone", current_dose_mg: 0, target_dose_mg: 50, status: "not_started" },
      { drug_class: "sglt2i", generic_name: "empagliflozin", current_dose_mg: 10, target_dose_mg: 10, status: "at_target" },
    ],
    gdmt_recommendations: [
      { drug_class: "raasi", generic_name: "lisinopril", action: "hold", current_dose_mg: 5, next_dose_mg: null, target_dose_mg: 40, safety_checks: [
        { check_name: "SBP >= 100", passed: false, current_value: 88, threshold: 100 },
        { check_name: "K+ < 5.0", passed: true, current_value: 3.8, threshold: 5.0 },
        { check_name: "Creatinine stable", passed: false, current_value: 0.6, threshold: 0.3 },
      ], tolerance_score: null, tolerance_factors: null, rationale: "Hold at 5 mg — failed: SBP >= 100, Creatinine stable." },
      { drug_class: "beta_blocker", generic_name: "carvedilol", action: "hold", current_dose_mg: 3.125, next_dose_mg: null, target_dose_mg: 25, safety_checks: [
        { check_name: "HR >= 60", passed: true, current_value: 92, threshold: 60 },
        { check_name: "SBP >= 90", passed: false, current_value: 88, threshold: 90 },
        { check_name: "No acute decompensation", passed: false, current_value: 0.85, threshold: 0.6 },
      ], tolerance_score: null, tolerance_factors: null, rationale: "Hold at 3.125 mg — failed: SBP >= 90, No acute decompensation." },
      { drug_class: "mra", generic_name: "eplerenone", action: "hold", current_dose_mg: 0, next_dose_mg: null, target_dose_mg: 50, safety_checks: [
        { check_name: "K+ < 5.0", passed: true, current_value: 3.8, threshold: 5.0 },
        { check_name: "eGFR >= 30", passed: false, current_value: 28, threshold: 30 },
        { check_name: "No AKI", passed: false, current_value: 0.6, threshold: 0.3 },
      ], tolerance_score: null, tolerance_factors: null, rationale: "Cannot initiate — failed: eGFR >= 30, No AKI." },
      { drug_class: "sglt2i", generic_name: "empagliflozin", action: "at_target", current_dose_mg: 10, next_dose_mg: null, target_dose_mg: 10, safety_checks: [
        { check_name: "eGFR >= 20", passed: true, current_value: 28, threshold: 20 },
      ], tolerance_score: null, tolerance_factors: null, rationale: "Already at target dose (10 mg). Continue current regimen." },
    ],
    optimization_score: 21.0,
  },
];
