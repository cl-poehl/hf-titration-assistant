"""
Synthetic Heart Failure Hospital-at-Home Patient Generator

Generates realistic time-series data for heart failure patients being monitored
at home, including vitals, labs, patient-reported symptoms, and deterioration
events. Based on published clinical parameters from heart failure literature.

Clinical references:
- ACC/AHA Heart Failure Guidelines (2022)
- MIMIC-IV heart failure cohort characteristics
- Published RPM thresholds for HF exacerbation detection
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class GDMTMedication:
    """A single GDMT medication with titration state."""
    drug_class: str  # "raasi", "beta_blocker", "mra", "sglt2i"
    generic_name: str
    current_dose_mg: float
    target_dose_mg: float
    status: str  # "not_started", "sub_therapeutic", "at_target", "not_indicated"


@dataclass
class TitrationEvent:
    """Record of a dose change during monitoring."""
    day: int
    drug_class: str
    generic_name: str
    old_dose_mg: float
    new_dose_mg: float
    tolerated: bool


@dataclass
class PatientProfile:
    """Baseline characteristics for a heart failure patient."""
    patient_id: str
    age: int
    sex: str  # M/F
    ef_category: str  # HFrEF (<40%), HFmrEF (40-49%), HFpEF (>=50%)
    ejection_fraction: float
    nyha_class: int  # I-IV
    baseline_weight_kg: float
    baseline_systolic_bp: int
    baseline_diastolic_bp: int
    baseline_hr: int
    baseline_spo2: float
    baseline_rr: int
    baseline_bnp: float  # pg/mL
    baseline_creatinine: float  # mg/dL
    baseline_potassium: float  # mEq/L
    comorbidities: list = field(default_factory=list)
    medications: list = field(default_factory=list)
    will_deteriorate: bool = False
    deterioration_day: Optional[int] = None
    gdmt_medications: list = field(default_factory=list)
    titration_events: list = field(default_factory=list)


class HeartFailurePatientGenerator:
    """
    Generates synthetic but clinically realistic heart failure patient data.

    Each patient gets:
    - A baseline profile (demographics, comorbidities, meds)
    - 30 days of time-series vitals (measured 2x/day from RPM devices)
    - Periodic labs (weekly)
    - Daily patient-reported symptom scores
    - A binary outcome: deterioration requiring escalation (yes/no)

    ~30% of patients will have a deterioration event, matching published
    HF readmission rates within 30 days.
    """

    COMORBIDITY_POOL = [
        "hypertension", "diabetes_t2", "atrial_fibrillation", "ckd_stage3",
        "copd", "obesity", "coronary_artery_disease", "sleep_apnea",
        "anemia", "depression"
    ]

    HF_MEDICATIONS = [
        "furosemide", "lisinopril", "carvedilol", "spironolactone",
        "sacubitril_valsartan", "empagliflozin", "digoxin", "hydralazine",
        "isosorbide_dinitrate", "metolazone"
    ]

    # Comorbidity prevalence in HF population (approximate)
    COMORBIDITY_PROBS = [0.75, 0.45, 0.35, 0.40, 0.20, 0.40, 0.55, 0.25, 0.30, 0.20]

    # GDMT titration catalog (mirrors src/api/predictor.py GDMT_CATALOG)
    GDMT_CATALOG = {
        "raasi": {
            "drugs": {
                "lisinopril": {"dose_steps_mg": [2.5, 5, 10, 20, 40], "target_dose_mg": 40},
                "sacubitril_valsartan": {"dose_steps_mg": [24, 49, 97], "target_dose_mg": 97},
                "losartan": {"dose_steps_mg": [25, 50, 100, 150], "target_dose_mg": 150},
            },
            "ef_categories": ["HFrEF", "HFmrEF"],
        },
        "beta_blocker": {
            "drugs": {
                "carvedilol": {"dose_steps_mg": [3.125, 6.25, 12.5, 25], "target_dose_mg": 25},
                "metoprolol_succinate": {"dose_steps_mg": [12.5, 25, 50, 100, 200], "target_dose_mg": 200},
            },
            "ef_categories": ["HFrEF", "HFmrEF"],
        },
        "mra": {
            "drugs": {
                "spironolactone": {"dose_steps_mg": [12.5, 25, 50], "target_dose_mg": 50},
                "eplerenone": {"dose_steps_mg": [25, 50], "target_dose_mg": 50},
            },
            "ef_categories": ["HFrEF", "HFmrEF"],
        },
        "sglt2i": {
            "drugs": {
                "dapagliflozin": {"dose_steps_mg": [10], "target_dose_mg": 10},
                "empagliflozin": {"dose_steps_mg": [10], "target_dose_mg": 10},
            },
            "ef_categories": ["HFrEF", "HFmrEF", "HFpEF"],
        },
    }

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def generate_patient_profile(self, patient_id: str) -> PatientProfile:
        """Generate a single patient's baseline profile."""
        age = int(self.rng.normal(72, 12))
        age = np.clip(age, 40, 95)
        sex = self.rng.choice(["M", "F"], p=[0.58, 0.42])  # HF skews male

        # EF distribution in HF population
        ef_cat = self.rng.choice(
            ["HFrEF", "HFmrEF", "HFpEF"],
            p=[0.45, 0.15, 0.40]
        )
        if ef_cat == "HFrEF":
            ef = self.rng.uniform(15, 39)
        elif ef_cat == "HFmrEF":
            ef = self.rng.uniform(40, 49)
        else:
            ef = self.rng.uniform(50, 70)

        nyha = self.rng.choice([1, 2, 3, 4], p=[0.05, 0.35, 0.45, 0.15])

        # Baselines — vary by severity
        severity_factor = (nyha - 1) / 3  # 0 to 1

        weight = self.rng.normal(85 + severity_factor * 10, 15)
        weight = max(50, weight)

        sbp = int(self.rng.normal(125 - severity_factor * 15, 12))
        dbp = int(self.rng.normal(72 - severity_factor * 8, 8))
        hr = int(self.rng.normal(78 + severity_factor * 15, 10))
        spo2 = round(self.rng.normal(96 - severity_factor * 2, 1.2), 1)
        spo2 = min(100, max(88, spo2))
        rr = int(self.rng.normal(16 + severity_factor * 4, 2))

        # Labs
        bnp = self.rng.lognormal(np.log(150 + severity_factor * 400), 0.5)
        creatinine = self.rng.normal(1.2 + severity_factor * 0.5, 0.3)
        creatinine = max(0.6, creatinine)
        potassium = self.rng.normal(4.2, 0.4)
        potassium = np.clip(potassium, 3.0, 6.0)

        # Comorbidities
        comorbidities = [
            c for c, p in zip(self.COMORBIDITY_POOL, self.COMORBIDITY_PROBS)
            if self.rng.random() < p
        ]

        # Medications — more severe patients get more meds
        n_meds = min(len(self.HF_MEDICATIONS), int(self.rng.normal(3 + severity_factor * 3, 1)))
        n_meds = max(1, n_meds)
        medications = list(self.rng.choice(self.HF_MEDICATIONS, size=n_meds, replace=False))

        # Deterioration: ~30% base rate, higher with severity
        deterioration_prob = 0.15 + severity_factor * 0.30
        will_deteriorate = self.rng.random() < deterioration_prob
        deterioration_day = None
        if will_deteriorate:
            # Deterioration can happen anytime in the 30-day window
            # but more likely in weeks 2-3
            deterioration_day = int(self.rng.triangular(5, 18, 28))

        profile = PatientProfile(
            patient_id=patient_id,
            age=age, sex=sex,
            ef_category=ef_cat, ejection_fraction=round(ef, 1),
            nyha_class=nyha,
            baseline_weight_kg=round(weight, 1),
            baseline_systolic_bp=sbp, baseline_diastolic_bp=dbp,
            baseline_hr=hr, baseline_spo2=spo2, baseline_rr=rr,
            baseline_bnp=round(bnp, 1),
            baseline_creatinine=round(creatinine, 2),
            baseline_potassium=round(potassium, 1),
            comorbidities=comorbidities,
            medications=medications,
            will_deteriorate=will_deteriorate,
            deterioration_day=deterioration_day,
        )

        # Assign GDMT medications and generate titration events
        profile.gdmt_medications = self._assign_gdmt_medications(profile)
        profile.titration_events = self._generate_titration_events(profile)

        return profile

    def _assign_gdmt_medications(self, profile: PatientProfile) -> list:
        """Assign GDMT medications based on EF category and randomize initial doses."""
        meds = []
        for drug_class, catalog in self.GDMT_CATALOG.items():
            indicated = profile.ef_category in catalog["ef_categories"]
            if not indicated:
                first_drug = list(catalog["drugs"].keys())[0]
                info = catalog["drugs"][first_drug]
                meds.append(GDMTMedication(
                    drug_class=drug_class,
                    generic_name=first_drug,
                    current_dose_mg=0,
                    target_dose_mg=info["target_dose_mg"],
                    status="not_indicated",
                ))
                continue

            # Pick a drug from this class
            drug_names = list(catalog["drugs"].keys())
            generic_name = self.rng.choice(drug_names)
            info = catalog["drugs"][generic_name]
            steps = info["dose_steps_mg"]

            # Randomize starting dose: ~20% not started, rest distributed across steps
            if self.rng.random() < 0.2:
                current_dose = 0
                status = "not_started"
            else:
                idx = int(self.rng.integers(0, len(steps)))
                current_dose = steps[idx]
                status = "at_target" if current_dose >= info["target_dose_mg"] else "sub_therapeutic"

            meds.append(GDMTMedication(
                drug_class=drug_class,
                generic_name=generic_name,
                current_dose_mg=current_dose,
                target_dose_mg=info["target_dose_mg"],
                status=status,
            ))
        return meds

    def _generate_titration_events(self, profile: PatientProfile, days: int = 30) -> list:
        """Generate titration events during the monitoring period."""
        events = []
        severity_factor = (profile.nyha_class - 1) / 3

        for med in profile.gdmt_medications:
            if med.status in ("not_indicated", "at_target"):
                continue

            catalog = self.GDMT_CATALOG[med.drug_class]
            info = catalog["drugs"][med.generic_name]
            steps = info["dose_steps_mg"]
            current_dose = med.current_dose_mg

            # Titration attempts every ~7-14 days
            titration_day = int(self.rng.integers(5, 14))
            while titration_day < days:
                if current_dose >= info["target_dose_mg"]:
                    break

                # Find next step
                next_dose = None
                for s in steps:
                    if s > current_dose:
                        next_dose = s
                        break
                if next_dose is None:
                    break

                # Tolerance probability: higher severity = lower tolerance
                # Deteriorating patients near event have low tolerance
                base_tolerance = 0.80 - severity_factor * 0.25
                if profile.will_deteriorate and profile.deterioration_day is not None:
                    days_to_event = profile.deterioration_day - titration_day
                    if days_to_event < 5:
                        base_tolerance *= 0.3
                    elif days_to_event < 10:
                        base_tolerance *= 0.6

                tolerated = self.rng.random() < base_tolerance
                events.append(TitrationEvent(
                    day=titration_day,
                    drug_class=med.drug_class,
                    generic_name=med.generic_name,
                    old_dose_mg=current_dose,
                    new_dose_mg=next_dose,
                    tolerated=tolerated,
                ))

                if tolerated:
                    current_dose = next_dose
                # Next attempt in 7-14 days
                titration_day += int(self.rng.integers(7, 15))

        return events

    def generate_vitals_timeseries(self, profile: PatientProfile, days: int = 30) -> pd.DataFrame:
        """
        Generate twice-daily vitals for a patient over the monitoring period.

        If the patient will deteriorate, vitals trend abnormally starting
        ~3-5 days before the deterioration event, mimicking real clinical
        presentation of acute decompensated heart failure.
        """
        records = []
        measurements_per_day = 2  # morning and evening

        for day in range(days):
            for measurement in range(measurements_per_day):
                hour = 8 if measurement == 0 else 20  # 8am and 8pm
                timestamp = pd.Timestamp("2026-01-01") + pd.Timedelta(days=day, hours=hour)

                # Calculate deterioration effect
                det_effect = self._deterioration_curve(
                    day, profile.will_deteriorate, profile.deterioration_day
                )

                # Weight: fluid retention is THE key signal in HF
                # Normal daily variation: +/- 0.5 kg
                # Deterioration: gradual gain of 2-5 kg over 3-5 days
                weight = (
                    profile.baseline_weight_kg
                    + self.rng.normal(0, 0.3)  # daily noise
                    + det_effect * self.rng.uniform(3, 6)  # fluid retention
                )

                # SpO2: drops during decompensation (pulmonary congestion)
                spo2 = (
                    profile.baseline_spo2
                    + self.rng.normal(0, 0.5)
                    - det_effect * self.rng.uniform(3, 7)
                )
                spo2 = np.clip(spo2, 80, 100)

                # Heart rate: compensatory tachycardia
                hr = (
                    profile.baseline_hr
                    + self.rng.normal(0, 4)
                    + det_effect * self.rng.uniform(15, 30)
                )

                # Blood pressure: can drop (cardiogenic) or rise (fluid overload)
                sbp = (
                    profile.baseline_systolic_bp
                    + self.rng.normal(0, 6)
                    - det_effect * self.rng.uniform(5, 20)  # dropping in decompensation
                )
                dbp = (
                    profile.baseline_diastolic_bp
                    + self.rng.normal(0, 4)
                    - det_effect * self.rng.uniform(3, 10)
                )

                # Respiratory rate: increases with congestion
                rr = (
                    profile.baseline_rr
                    + self.rng.normal(0, 1.5)
                    + det_effect * self.rng.uniform(4, 10)
                )

                records.append({
                    "patient_id": profile.patient_id,
                    "timestamp": timestamp,
                    "day": day,
                    "measurement": "morning" if measurement == 0 else "evening",
                    "weight_kg": round(weight, 1),
                    "spo2": round(float(spo2), 1),
                    "heart_rate": int(np.clip(hr, 40, 180)),
                    "systolic_bp": int(np.clip(sbp, 70, 200)),
                    "diastolic_bp": int(np.clip(dbp, 40, 120)),
                    "respiratory_rate": int(np.clip(rr, 8, 40)),
                })

        return pd.DataFrame(records)

    def generate_labs_timeseries(self, profile: PatientProfile, days: int = 30) -> pd.DataFrame:
        """Generate weekly lab values (BNP, creatinine, potassium)."""
        records = []
        lab_days = list(range(0, days, 7))  # weekly

        for day in lab_days:
            timestamp = pd.Timestamp("2026-01-01") + pd.Timedelta(days=day, hours=10)
            det_effect = self._deterioration_curve(
                day, profile.will_deteriorate, profile.deterioration_day
            )

            # BNP rises dramatically during decompensation
            bnp = (
                profile.baseline_bnp
                * (1 + self.rng.normal(0, 0.1))  # normal variation ~10%
                * (1 + det_effect * self.rng.uniform(1.5, 4.0))  # can 2-5x during decompensation
            )

            # Creatinine rises (cardiorenal syndrome)
            creatinine = (
                profile.baseline_creatinine
                + self.rng.normal(0, 0.05)
                + det_effect * self.rng.uniform(0.3, 0.8)
            )

            # Potassium can swing with diuretic changes
            potassium = (
                profile.baseline_potassium
                + self.rng.normal(0, 0.15)
                + det_effect * self.rng.choice([-1, 1]) * self.rng.uniform(0.3, 0.8)
            )

            records.append({
                "patient_id": profile.patient_id,
                "timestamp": timestamp,
                "day": day,
                "bnp_pg_ml": round(max(10, bnp), 1),
                "creatinine_mg_dl": round(max(0.4, creatinine), 2),
                "potassium_meq_l": round(float(np.clip(potassium, 2.5, 7.0)), 1),
            })

        return pd.DataFrame(records)

    def generate_symptoms_timeseries(self, profile: PatientProfile, days: int = 30) -> pd.DataFrame:
        """Generate daily patient-reported symptom scores."""
        records = []

        for day in range(days):
            timestamp = pd.Timestamp("2026-01-01") + pd.Timedelta(days=day, hours=9)
            det_effect = self._deterioration_curve(
                day, profile.will_deteriorate, profile.deterioration_day
            )

            # Dyspnea scale 0-10
            dyspnea_base = profile.nyha_class * 1.5
            dyspnea = dyspnea_base + self.rng.normal(0, 0.8) + det_effect * 4
            dyspnea = np.clip(dyspnea, 0, 10)

            # Orthopnea: how many pillows (0-4)
            orthopnea_base = max(0, profile.nyha_class - 1)
            orthopnea = orthopnea_base + det_effect * 2 + self.rng.normal(0, 0.3)
            orthopnea = int(np.clip(orthopnea, 0, 4))

            # Ankle swelling: 0 (none), 1 (mild), 2 (moderate), 3 (severe)
            edema_base = max(0, profile.nyha_class - 2) * 0.5
            edema = edema_base + det_effect * 2 + self.rng.normal(0, 0.3)
            edema = int(np.clip(edema, 0, 3))

            # Exercise tolerance: 0 (bedbound) to 10 (normal)
            exercise_base = 10 - profile.nyha_class * 2
            exercise = exercise_base + self.rng.normal(0, 0.5) - det_effect * 4
            exercise = np.clip(exercise, 0, 10)

            # Medication adherence: probability of taking all meds
            adherence_prob = 0.85 - det_effect * 0.2  # sicker patients may miss doses
            med_adherent = int(self.rng.random() < adherence_prob)

            # Patient sometimes doesn't report (missingness ~5%, higher when deteriorating)
            missing_prob = 0.05 + det_effect * 0.15
            reported = self.rng.random() > missing_prob

            records.append({
                "patient_id": profile.patient_id,
                "timestamp": timestamp,
                "day": day,
                "reported": reported,
                "dyspnea_score": round(float(dyspnea), 1) if reported else np.nan,
                "orthopnea_pillows": orthopnea if reported else np.nan,
                "ankle_edema": edema if reported else np.nan,
                "exercise_tolerance": round(float(exercise), 1) if reported else np.nan,
                "medication_adherent": med_adherent if reported else np.nan,
            })

        return pd.DataFrame(records)

    def _deterioration_curve(self, current_day: int, will_deteriorate: bool,
                             deterioration_day: Optional[int]) -> float:
        """
        Returns a 0-1 deterioration effect factor.

        Mimics real decompensation: gradual worsening over 3-5 days before
        the acute event, with some day-to-day noise.
        """
        if not will_deteriorate or deterioration_day is None:
            return 0.0

        # Onset starts 3-5 days before the event
        onset_day = deterioration_day - 4
        if current_day < onset_day:
            return 0.0
        if current_day >= deterioration_day:
            return 1.0

        # Sigmoid-like ramp up
        progress = (current_day - onset_day) / (deterioration_day - onset_day)
        # Add slight noise to prevent perfect signals
        noise = self.rng.normal(0, 0.05)
        return float(np.clip(progress ** 1.5 + noise, 0, 1))

    def generate_outcomes(self, profile: PatientProfile) -> dict:
        """Generate the outcome label for this patient."""
        return {
            "patient_id": profile.patient_id,
            "deteriorated": int(profile.will_deteriorate),
            "deterioration_day": profile.deterioration_day,
            "age": profile.age,
            "sex": profile.sex,
            "ef_category": profile.ef_category,
            "ejection_fraction": profile.ejection_fraction,
            "nyha_class": profile.nyha_class,
            "n_comorbidities": len(profile.comorbidities),
            "comorbidities": ",".join(profile.comorbidities),
            "n_medications": len(profile.medications),
            "medications": ",".join(profile.medications),
        }

    def generate_cohort(self, n_patients: int = 2000, days: int = 30,
                        output_dir: Optional[str] = None) -> dict:
        """
        Generate a full cohort of heart failure patients.

        Returns dict of DataFrames: vitals, labs, symptoms, outcomes, medications.
        Optionally writes to parquet files.
        """
        all_vitals = []
        all_labs = []
        all_symptoms = []
        all_outcomes = []
        all_medications = []

        for i in range(n_patients):
            pid = f"HF-{i:05d}"
            profile = self.generate_patient_profile(pid)

            all_vitals.append(self.generate_vitals_timeseries(profile, days))
            all_labs.append(self.generate_labs_timeseries(profile, days))
            all_symptoms.append(self.generate_symptoms_timeseries(profile, days))
            all_outcomes.append(self.generate_outcomes(profile))

            # GDMT medication records
            for med in profile.gdmt_medications:
                all_medications.append({
                    "patient_id": pid,
                    "drug_class": med.drug_class,
                    "generic_name": med.generic_name,
                    "current_dose_mg": med.current_dose_mg,
                    "target_dose_mg": med.target_dose_mg,
                    "status": med.status,
                })

            # Titration events as additional medication records
            for evt in profile.titration_events:
                all_medications.append({
                    "patient_id": pid,
                    "drug_class": evt.drug_class,
                    "generic_name": evt.generic_name,
                    "current_dose_mg": evt.new_dose_mg,
                    "target_dose_mg": next(
                        (m.target_dose_mg for m in profile.gdmt_medications
                         if m.drug_class == evt.drug_class), 0
                    ),
                    "status": "titration_event",
                    "titration_day": evt.day,
                    "old_dose": evt.old_dose_mg,
                    "new_dose": evt.new_dose_mg,
                    "tolerated": evt.tolerated,
                })

            if (i + 1) % 500 == 0:
                print(f"  Generated {i + 1}/{n_patients} patients...")

        result = {
            "vitals": pd.concat(all_vitals, ignore_index=True),
            "labs": pd.concat(all_labs, ignore_index=True),
            "symptoms": pd.concat(all_symptoms, ignore_index=True),
            "outcomes": pd.DataFrame(all_outcomes),
            "medications": pd.DataFrame(all_medications),
        }

        # Summary stats
        n_deteriorated = result["outcomes"]["deteriorated"].sum()
        med_df = result["medications"]
        n_titration_events = len(med_df[med_df["status"] == "titration_event"]) if "status" in med_df.columns else 0
        print(f"\nCohort summary:")
        print(f"  Total patients: {n_patients}")
        print(f"  Deteriorated: {n_deteriorated} ({100*n_deteriorated/n_patients:.1f}%)")
        print(f"  Vitals records: {len(result['vitals']):,}")
        print(f"  Lab records: {len(result['labs']):,}")
        print(f"  Symptom records: {len(result['symptoms']):,}")
        print(f"  Medication records: {len(result['medications']):,} ({n_titration_events} titration events)")

        if output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            for name, df in result.items():
                path = out / f"{name}.parquet"
                df.to_parquet(path, index=False)
                print(f"  Saved {path}")

        return result


if __name__ == "__main__":
    import os

    # Defaults to <project>/data/raw; override with HFTA_DATA_DIR.
    project_root = Path(__file__).resolve().parents[2]
    base_dir = Path(os.environ.get("HFTA_DATA_DIR", project_root / "data"))

    print("Generating synthetic heart failure cohort...")
    generator = HeartFailurePatientGenerator(seed=42)
    data = generator.generate_cohort(
        n_patients=2000,
        days=30,
        output_dir=str(base_dir / "raw"),
    )
