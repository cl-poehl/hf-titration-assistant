"""
Combine extracted clinical datasets (MIMIC-IV, eICU, UCI) into a unified dataset.

Normalizes schemas across sources and handles missing fields gracefully.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def normalize_sex(sex_val):
    """Normalize sex values across datasets."""
    s = str(sex_val).strip().lower()
    if s in ("m", "male"):
        return "M"
    elif s in ("f", "female"):
        return "F"
    return "U"


def combine_vitals(data_dirs: list[Path]) -> pd.DataFrame:
    """Combine vitals from all sources."""
    frames = []
    for d in data_dirs:
        path = d / "vitals.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            source = d.name
            df["source"] = source
            frames.append(df)
            print(f"  {source}: {len(df)} vital records, {df['patient_id'].nunique()} patients")
    combined = pd.concat(frames, ignore_index=True)
    print(f"  Combined vitals: {len(combined)} records, {combined['patient_id'].nunique()} patients")
    return combined


def combine_labs(data_dirs: list[Path]) -> pd.DataFrame:
    """Combine labs from all sources."""
    frames = []
    for d in data_dirs:
        path = d / "labs.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            source = d.name
            df["source"] = source
            frames.append(df)
            print(f"  {source}: {len(df)} lab records, {df['patient_id'].nunique()} patients")
    combined = pd.concat(frames, ignore_index=True)
    print(f"  Combined labs: {len(combined)} records, {combined['patient_id'].nunique()} patients")
    return combined


def combine_symptoms(data_dirs: list[Path]) -> pd.DataFrame:
    """Combine symptoms from all sources."""
    frames = []
    for d in data_dirs:
        path = d / "symptoms.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            source = d.name
            df["source"] = source
            frames.append(df)
            print(f"  {source}: {len(df)} symptom records")
    combined = pd.concat(frames, ignore_index=True)
    print(f"  Combined symptoms: {len(combined)} records")
    return combined


def combine_outcomes(data_dirs: list[Path]) -> pd.DataFrame:
    """Combine outcomes from all sources, normalizing schema differences."""
    frames = []
    for d in data_dirs:
        path = d / "outcomes.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            source = d.name
            df["source"] = source
            # Normalize sex
            df["sex"] = df["sex"].apply(normalize_sex)
            frames.append(df)
            n_det = df["deteriorated"].sum()
            print(f"  {source}: {len(df)} stays, {n_det} ({100*n_det/max(len(df),1):.1f}%) deteriorated")

    combined = pd.concat(frames, ignore_index=True)

    # Fill missing static features with defaults (not available in real data)
    for col, default in [
        ("ejection_fraction", np.nan),
        ("ef_category", "Unknown"),
        ("nyha_class", np.nan),
        ("n_comorbidities", np.nan),
        ("n_medications", np.nan),
    ]:
        if col not in combined.columns:
            combined[col] = default

    n_det = combined["deteriorated"].sum()
    print(f"  Combined outcomes: {len(combined)} stays, {n_det} ({100*n_det/max(len(combined),1):.1f}%) deteriorated")
    return combined


def add_uci_outcomes(outcomes: pd.DataFrame, uci_path: str) -> pd.DataFrame:
    """
    Add UCI Heart Failure dataset as outcome-only data.

    UCI has static features (age, creatinine, EF, etc.) but no time-series.
    We add these patients to the outcome pool — the feature pipeline will
    skip them (no vitals data) but they provide additional validation data.
    """
    uci = pd.read_csv(uci_path)
    uci_records = []

    for i, row in uci.iterrows():
        uci_records.append({
            "patient_id": f"UCI-{i}",
            "deteriorated": int(row["DEATH_EVENT"]),
            "deterioration_day": max(1, int(row["time"]) - 14) if row["DEATH_EVENT"] else None,
            "age": int(row["age"]),
            "sex": "M" if row["sex"] == 1 else "F",
            "los_days": row["time"],
            "ejection_fraction": row["ejection_fraction"],
            "ef_category": (
                "HFrEF" if row["ejection_fraction"] < 40
                else "HFmrEF" if row["ejection_fraction"] < 50
                else "HFpEF"
            ),
            "nyha_class": np.nan,
            "n_comorbidities": sum([row["anaemia"], row["diabetes"], row["high_blood_pressure"]]),
            "n_medications": np.nan,
            "source": "uci",
        })

    uci_df = pd.DataFrame(uci_records)
    n_det = uci_df["deteriorated"].sum()
    print(f"  UCI: {len(uci_df)} patients, {n_det} ({100*n_det/max(len(uci_df),1):.1f}%) died")

    # UCI patients won't have vitals/labs, so they won't produce features.
    # But we keep them in outcomes for reference.
    combined = pd.concat([outcomes, uci_df], ignore_index=True)
    return combined


def combine_all(base_dir: str, output_dir: str):
    """Main combination pipeline."""
    base = Path(base_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Find available data directories
    data_dirs = []
    for name in ["mimic", "eicu"]:
        d = base / name
        if d.exists():
            data_dirs.append(d)
            print(f"Found {name} data at {d}")

    if not data_dirs:
        raise ValueError("No extracted data found!")

    print("\nCombining vitals...")
    vitals = combine_vitals(data_dirs)

    print("\nCombining labs...")
    labs = combine_labs(data_dirs)

    print("\nCombining symptoms...")
    symptoms = combine_symptoms(data_dirs)

    print("\nCombining outcomes...")
    outcomes = combine_outcomes(data_dirs)

    # Add UCI (static-only) data
    uci_path = base / "external" / "uci-hf" / "heart_failure_clinical_records.csv"
    if uci_path.exists():
        print("\nAdding UCI HF dataset...")
        outcomes = add_uci_outcomes(outcomes, str(uci_path))

    # Save
    vitals.to_parquet(out / "vitals.parquet", index=False)
    labs.to_parquet(out / "labs.parquet", index=False)
    symptoms.to_parquet(out / "symptoms.parquet", index=False)
    outcomes.to_parquet(out / "outcomes.parquet", index=False)

    print(f"\nSaved combined dataset to {out}/")
    print(f"  Vitals: {len(vitals):,} records ({vitals['patient_id'].nunique()} patients)")
    print(f"  Labs: {len(labs):,} records ({labs['patient_id'].nunique()} patients)")
    print(f"  Symptoms: {len(symptoms):,} records")
    print(f"  Outcomes: {len(outcomes)} total ({outcomes[outcomes['source']!='uci']['patient_id'].nunique()} with time-series)")

    return {"vitals": vitals, "labs": labs, "symptoms": symptoms, "outcomes": outcomes}


if __name__ == "__main__":
    import os

    BASE_DIR = str(Path(os.environ.get(
        "HFTA_DATA_DIR", Path(__file__).resolve().parents[2] / "data")))
    OUTPUT_DIR = str(Path(BASE_DIR) / "combined")

    combine_all(BASE_DIR, OUTPUT_DIR)
