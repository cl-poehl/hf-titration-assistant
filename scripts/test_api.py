"""End-to-end test: prediction + GDMT titration engine."""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_generation.generate_hf_patients import HeartFailurePatientGenerator
from src.api.predictor import HFPredictor, GDMTEngine, GDMT_CATALOG
from src.features.build_features import compute_vital_features, compute_lab_features, compute_egfr

MODEL_PATH = str(Path(__file__).resolve().parents[1] / "models" / "best_model.pkl")

# Generate two test patients: one stable, one deteriorating
gen = HeartFailurePatientGenerator(seed=99)

print("=" * 60)
print("End-to-end prediction + GDMT test")
print("=" * 60)

predictor = HFPredictor(MODEL_PATH)
gdmt_engine = GDMTEngine(predictor)

for i, scenario in enumerate(["stable", "deteriorating"]):
    # Generate until we get the right type
    while True:
        profile = gen.generate_patient_profile(f"TEST-{scenario}")
        if scenario == "stable" and not profile.will_deteriorate:
            break
        if scenario == "deteriorating" and profile.will_deteriorate:
            break

    vitals = gen.generate_vitals_timeseries(profile, days=30)
    labs = gen.generate_labs_timeseries(profile, days=30)
    symptoms = gen.generate_symptoms_timeseries(profile, days=30)

    patient_context = {
        "age": profile.age,
        "sex_male": int(profile.sex == "M"),
        "ef": profile.ejection_fraction,
        "nyha_class": profile.nyha_class,
        "n_comorbidities": len(profile.comorbidities),
        "n_medications": len(profile.medications),
        "ef_reduced": int(profile.ef_category == "HFrEF"),
        "ef_mid": int(profile.ef_category == "HFmrEF"),
        "ef_preserved": int(profile.ef_category == "HFpEF"),
    }

    # --- Standard prediction ---
    result = predictor.predict(vitals, labs, symptoms, patient_context)

    print(f"\n{'='*60}")
    print(f"Patient: {scenario.upper()}")
    print(f"{'='*60}")
    print(f"  Age: {profile.age}, Sex: {profile.sex}, EF: {profile.ejection_fraction}% ({profile.ef_category}), NYHA: {profile.nyha_class}")
    if profile.will_deteriorate:
        print(f"  Ground truth: deteriorates on day {profile.deterioration_day}")
    else:
        print(f"  Ground truth: stable throughout")
    print(f"  Risk Score: {result['risk_score']}/100")
    print(f"  Risk Tier:  {result['risk_tier']}")
    print(f"  Suggested Action: {result['suggested_action']['action']}")
    print(f"  Top Contributing Factors:")
    for f in result["top_factors"]:
        arrow = "↑" if f["direction"] == "increasing_risk" else "↓"
        print(f"    {arrow} {f['display_name']}: {f['value']} (impact: {f['impact']:+.4f})")

    # --- Tolerance prediction ---
    tolerance = predictor.predict_tolerance(vitals, labs, symptoms, patient_context)
    print(f"\n  Tolerance Score: {tolerance['tolerance_score']}/100")
    print(f"  Tolerance Factors:")
    for f in tolerance["tolerance_factors"][:3]:
        arrow = "+" if f["direction"] == "increasing_tolerance" else "-"
        print(f"    {arrow} {f['display_name']}: {f['value']}")

    # --- GDMT titration evaluation ---
    print(f"\n  GDMT Titration Evaluation:")
    current_day = vitals["day"].max()
    vital_feats = compute_vital_features(vitals, current_day)
    lab_feats = compute_lab_features(labs, current_day, age=profile.age, sex_male=int(profile.sex == "M"))
    features = {**vital_feats, **lab_feats, **patient_context}
    features["egfr"] = compute_egfr(
        lab_feats.get("creat_current", 1.0), profile.age, int(profile.sex == "M")
    )

    for med in profile.gdmt_medications:
        if med.status == "not_indicated":
            print(f"    {med.drug_class:15s} — not indicated for {profile.ef_category}")
            continue

        med_dict = {
            "generic_name": med.generic_name,
            "current_dose_mg": med.current_dose_mg,
            "target_dose_mg": med.target_dose_mg,
            "status": med.status,
        }
        rec = gdmt_engine.evaluate_titration(med.drug_class, med_dict, features)
        checks_str = ", ".join(
            f"{'✓' if c['passed'] else '✗'} {c['check_name']}" for c in rec["safety_checks"]
        )
        print(f"    {med.drug_class:15s} {med.generic_name:25s} {rec['action']:12s} "
              f"{rec['current_dose_mg']:>6g} mg → {str(rec['next_dose_mg'] or '-'):>6s} mg  "
              f"[{checks_str}]")

    # Compute optimization score
    meds_as_dicts = [
        {"drug_class": m.drug_class, "generic_name": m.generic_name,
         "current_dose_mg": m.current_dose_mg, "target_dose_mg": m.target_dose_mg,
         "status": m.status}
        for m in profile.gdmt_medications
    ]
    opt_score = GDMTEngine.compute_optimization_score(meds_as_dicts)
    print(f"  GDMT Optimization Score: {opt_score}%")

    # --- Assertions ---
    if scenario == "stable":
        assert tolerance["tolerance_score"] > 50, \
            f"Stable patient should have high tolerance, got {tolerance['tolerance_score']}"
        print(f"\n  ✓ Stable patient has high tolerance score ({tolerance['tolerance_score']})")
    else:
        assert result["risk_score"] > 30, \
            f"Deteriorating patient should have elevated risk, got {result['risk_score']}"
        print(f"\n  ✓ Deteriorating patient has elevated risk ({result['risk_score']})")

print("\n" + "=" * 60)
print("All tests passed.")
