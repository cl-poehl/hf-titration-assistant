"""
Heart Failure Deterioration Prediction Model Training

Trains XGBoost and LightGBM models with proper clinical evaluation:
- Patient-level train/test split (no data leakage between time windows of same patient)
- Stratified by outcome
- Calibration assessment (critical for clinical use)
- SHAP explanations for CDS transparency
- Threshold optimization for clinical alert tiers
"""

import json
import os
import pickle
import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
import xgboost as xgb

warnings.filterwarnings("ignore", category=UserWarning)

# Paths default to locations relative to the project root so the pipeline runs
# on any clone. Override with the HFTA_DATA_DIR / HFTA_MODEL_DIR
# environment variables (e.g. to point at an HPC scratch workspace).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("HFTA_DATA_DIR", PROJECT_ROOT / "data"))
MODEL_DIR = str(Path(os.environ.get("HFTA_MODEL_DIR", PROJECT_ROOT / "models")))
RESULTS_DIR = str(Path(MODEL_DIR) / "results")

DATA_PATH_SYNTHETIC = str(DATA_DIR / "processed" / "features.parquet")
DATA_PATH_REAL = str(DATA_DIR / "processed" / "features_real.parquet")

DATA_PATH = DATA_PATH_SYNTHETIC  # default, overridden by --real flag

EXCLUDE_COLS = ["patient_id", "prediction_day", "label"]


def load_and_split(data_path: str = None, test_size: float = 0.2, seed: int = 42):
    """
    Load features and split by PATIENT (not by row).
    Critical: all time windows of a patient go to the same split.
    """
    df = pd.read_parquet(data_path or DATA_PATH)

    # Patient-level outcome for stratification
    patient_outcomes = df.groupby("patient_id")["label"].max().reset_index()
    patient_outcomes.columns = ["patient_id", "ever_deteriorated"]

    rng = np.random.default_rng(seed)
    patients = patient_outcomes.copy()
    patients["fold"] = -1

    # Stratified split by deterioration outcome
    pos_patients = patients[patients["ever_deteriorated"] == 1]["patient_id"].values
    neg_patients = patients[patients["ever_deteriorated"] == 0]["patient_id"].values

    rng.shuffle(pos_patients)
    rng.shuffle(neg_patients)

    n_pos_test = int(len(pos_patients) * test_size)
    n_neg_test = int(len(neg_patients) * test_size)

    test_patients = set(
        list(pos_patients[:n_pos_test]) + list(neg_patients[:n_neg_test])
    )

    train_df = df[~df["patient_id"].isin(test_patients)].copy()
    test_df = df[df["patient_id"].isin(test_patients)].copy()

    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]

    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["label"].values

    print(f"Train: {len(X_train)} samples, {y_train.sum()} positive ({100*y_train.mean():.1f}%)")
    print(f"Test:  {len(X_test)} samples, {y_test.sum()} positive ({100*y_test.mean():.1f}%)")
    print(f"Train patients: {train_df['patient_id'].nunique()}")
    print(f"Test patients:  {test_df['patient_id'].nunique()}")
    print(f"Features: {len(feature_cols)}")

    return X_train, y_train, X_test, y_test, feature_cols, train_df, test_df


def train_xgboost(X_train, y_train, X_test, y_test, small_dataset=False):
    """Train XGBoost with parameters tuned for imbalanced clinical data."""
    scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    if small_dataset:
        # Conservative hyperparameters for small, highly imbalanced datasets
        params = {
            "objective": "binary:logistic",
            "eval_metric": ["auc", "aucpr"],
            "scale_pos_weight": scale_pos,
            "max_depth": 3,
            "learning_rate": 0.02,
            "n_estimators": 300,
            "subsample": 0.7,
            "colsample_bytree": 0.5,
            "min_child_weight": 3,
            "reg_alpha": 1.0,
            "reg_lambda": 5.0,
            "gamma": 1.0,
            "random_state": 42,
            "tree_method": "hist",
            "verbosity": 0,
        }
    else:
        params = {
            "objective": "binary:logistic",
            "eval_metric": ["auc", "aucpr"],
            "scale_pos_weight": scale_pos,
            "max_depth": 6,
            "learning_rate": 0.05,
            "n_estimators": 500,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "tree_method": "hist",
            "verbosity": 0,
        }

    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    return model


def train_lightgbm(X_train, y_train, X_test, y_test, small_dataset=False):
    """Train LightGBM with parameters tuned for imbalanced clinical data."""
    scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    if small_dataset:
        params = {
            "objective": "binary",
            "metric": ["auc", "average_precision"],
            "scale_pos_weight": scale_pos,
            "max_depth": 3,
            "learning_rate": 0.02,
            "n_estimators": 300,
            "subsample": 0.7,
            "colsample_bytree": 0.5,
            "min_child_samples": 5,
            "reg_alpha": 1.0,
            "reg_lambda": 5.0,
            "random_state": 42,
            "verbose": -1,
        }
    else:
        params = {
            "objective": "binary",
            "metric": ["auc", "average_precision"],
            "scale_pos_weight": scale_pos,
            "max_depth": 6,
            "learning_rate": 0.05,
            "n_estimators": 500,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 20,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "verbose": -1,
        }

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.log_evaluation(50)],
    )

    return model


def evaluate_model(model, X_test, y_test, feature_cols, model_name, results_dir,
                   test_groups=None):
    """Comprehensive clinical evaluation.

    If test_groups (patient id per test row) is given, the AUROC confidence
    interval is bootstrapped at the patient level.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {}

    # ROC
    auroc = roc_auc_score(y_test, y_prob)
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob)
    metrics["auroc"] = auroc

    # Precision-Recall (more informative for imbalanced data)
    auprc = average_precision_score(y_test, y_prob)
    precision, recall, pr_thresholds = precision_recall_curve(y_test, y_prob)
    metrics["auprc"] = auprc

    # 95% bootstrap CI for AUROC. Resample at the PATIENT level when groups are
    # available, so the interval respects that each patient contributes several
    # correlated windows (row-level resampling would understate uncertainty).
    rng = np.random.default_rng(42)
    n_boot = 1000
    boot = []
    if test_groups is not None:
        groups = np.asarray(test_groups)
        patients = np.unique(groups)
        idx_by_pt = {p: np.where(groups == p)[0] for p in patients}
        for _ in range(n_boot):
            sampled = rng.choice(patients, size=len(patients), replace=True)
            idx = np.concatenate([idx_by_pt[p] for p in sampled])
            if len(np.unique(y_test[idx])) < 2:
                continue
            boot.append(roc_auc_score(y_test[idx], y_prob[idx]))
    else:
        n = len(y_test)
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            if len(np.unique(y_test[idx])) < 2:
                continue
            boot.append(roc_auc_score(y_test[idx], y_prob[idx]))
    if boot:
        ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
        metrics["auroc_ci95_low"] = float(ci_low)
        metrics["auroc_ci95_high"] = float(ci_high)

    print(f"\n{'='*50}")
    print(f"{model_name} Results")
    print(f"{'='*50}")
    print(f"AUROC:  {auroc:.4f}", end="")
    if boot:
        print(f"  (95% CI {ci_low:.3f}–{ci_high:.3f}, patient-level bootstrap)")
    else:
        print()
    print(f"AUPRC:  {auprc:.4f}")

    # Find optimal threshold (maximize F1)
    f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = pr_thresholds[best_idx]
    metrics["optimal_threshold"] = float(best_threshold)
    metrics["optimal_f1"] = float(f1_scores[best_idx])

    # Clinical alert tiers — thresholds at which a patient reaches each tier,
    # matching RISK_TIER_THRESHOLDS in src/api/predictor.py and the documented
    # tiers (Medium >=15%, High >=60%, Critical >=80%). Each row reports metrics
    # for "alert at this tier or above".
    tiers = {
        "medium_risk": 0.15,
        "high_risk": 0.60,
        "critical_risk": 0.80,
    }

    print(f"\nClinical Alert Tiers:")
    for tier, threshold in tiers.items():
        y_pred = (y_prob >= threshold).astype(int)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        print(f"  {tier} (>={threshold:.2f}): sens={sens:.3f}, spec={spec:.3f}, ppv={ppv:.3f}, alerts={tp+fp}")
        metrics[f"{tier}_sensitivity"] = sens
        metrics[f"{tier}_specificity"] = spec
        metrics[f"{tier}_ppv"] = ppv

    # Classification report at optimal threshold
    y_pred_opt = (y_prob >= best_threshold).astype(int)
    print(f"\nAt optimal threshold ({best_threshold:.3f}):")
    print(classification_report(y_test, y_pred_opt, target_names=["Stable", "Deteriorating"]))

    # --- Plots ---

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # ROC curve
    axes[0, 0].plot(fpr, tpr, "b-", linewidth=2, label=f"AUROC = {auroc:.3f}")
    axes[0, 0].plot([0, 1], [0, 1], "k--", alpha=0.3)
    axes[0, 0].set_xlabel("False Positive Rate")
    axes[0, 0].set_ylabel("True Positive Rate")
    axes[0, 0].set_title("ROC Curve")
    axes[0, 0].legend()

    # PR curve
    axes[0, 1].plot(recall, precision, "r-", linewidth=2, label=f"AUPRC = {auprc:.3f}")
    baseline = y_test.mean()
    axes[0, 1].axhline(y=baseline, color="k", linestyle="--", alpha=0.3, label=f"Baseline = {baseline:.3f}")
    axes[0, 1].set_xlabel("Recall")
    axes[0, 1].set_ylabel("Precision")
    axes[0, 1].set_title("Precision-Recall Curve")
    axes[0, 1].legend()

    # Calibration plot
    n_bins = min(10, max(3, y_test.sum()))  # adaptive bins for small datasets
    try:
        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=n_bins)
        axes[1, 0].plot(prob_pred, prob_true, "go-", linewidth=2, label="Model")
    except ValueError:
        axes[1, 0].text(0.5, 0.5, "Insufficient data\nfor calibration", ha="center", va="center")
    axes[1, 0].plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect calibration")
    axes[1, 0].set_xlabel("Mean predicted probability")
    axes[1, 0].set_ylabel("Fraction of positives")
    axes[1, 0].set_title("Calibration Curve")
    axes[1, 0].legend()

    # Score distribution
    axes[1, 1].hist(y_prob[y_test == 0], bins=50, alpha=0.6, label="Stable", density=True)
    axes[1, 1].hist(y_prob[y_test == 1], bins=50, alpha=0.6, label="Deteriorating", density=True)
    for tier, threshold in tiers.items():
        axes[1, 1].axvline(x=threshold, color="gray", linestyle="--", alpha=0.5)
    axes[1, 1].set_xlabel("Predicted Risk Score")
    axes[1, 1].set_ylabel("Density")
    axes[1, 1].set_title("Risk Score Distribution")
    axes[1, 1].legend()

    plt.suptitle(f"{model_name} — Heart Failure Deterioration Prediction", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / f"{model_name}_evaluation.png", dpi=150)
    plt.close()

    # SHAP explanations
    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # For XGBoost binary, shap_values is already 1D
    # For LightGBM, it may return a list [neg_class, pos_class]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Feature importance from SHAP
    shap_importance = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "shap_importance": shap_importance,
    }).sort_values("shap_importance", ascending=False)

    print(f"\nTop 15 Features ({model_name}):")
    for _, row in importance_df.head(15).iterrows():
        print(f"  {row['feature']:40s} {row['shap_importance']:.4f}")

    importance_df.to_csv(results_dir / f"{model_name}_feature_importance.csv", index=False)

    # SHAP summary plot
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_cols, show=False, max_display=20)
    plt.title(f"{model_name} — SHAP Feature Importance")
    plt.tight_layout()
    plt.savefig(results_dir / f"{model_name}_shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Save metrics
    with open(results_dir / f"{model_name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main():
    import sys
    use_real = "--real" in sys.argv
    use_tolerance = "--tolerance" in sys.argv
    data_label = "Real Clinical (MIMIC+eICU)" if use_real else "Synthetic"
    prediction_target = "tolerance" if use_tolerance else "deterioration"

    print("=" * 60)
    print(f"Heart Failure {prediction_target.title()} Prediction — Model Training")
    print(f"Data: {data_label}")
    if use_tolerance:
        print("Target: Tolerance (will patient tolerate GDMT uptitration?)")
    print("=" * 60)

    data_path = DATA_PATH_REAL if use_real else DATA_PATH_SYNTHETIC
    suffix_parts = []
    if use_real:
        suffix_parts.append("real")
    if use_tolerance:
        suffix_parts.append("tolerance")
    model_suffix = ("_" + "_".join(suffix_parts)) if suffix_parts else ""
    results_dir = RESULTS_DIR + (f"_{'-'.join(suffix_parts)}" if suffix_parts else "")

    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    # Load and split data
    print("\nLoading data...")
    X_train, y_train, X_test, y_test, feature_cols, train_df, test_df = load_and_split(data_path)

    small = len(X_train) < 2000

    # Train XGBoost
    print("\n--- Training XGBoost ---")
    xgb_model = train_xgboost(X_train, y_train, X_test, y_test, small_dataset=small)
    test_groups = test_df["patient_id"].values
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test, feature_cols, "xgboost", results_dir, test_groups)

    # Train LightGBM
    print("\n--- Training LightGBM ---")
    lgbm_model = train_lightgbm(X_train, y_train, X_test, y_test, small_dataset=small)
    lgbm_metrics = evaluate_model(lgbm_model, X_test, y_test, feature_cols, "lightgbm", results_dir, test_groups)

    # Pick the best model
    best_name = "xgboost" if xgb_metrics["auroc"] >= lgbm_metrics["auroc"] else "lightgbm"
    best_model = xgb_model if best_name == "xgboost" else lgbm_model
    best_metrics = xgb_metrics if best_name == "xgboost" else lgbm_metrics

    print(f"\n{'='*50}")
    print(f"Best model: {best_name} (AUROC={best_metrics['auroc']:.4f})")
    print(f"{'='*50}")

    # Clinical tiers — canonical boundaries, kept in sync with
    # RISK_TIER_THRESHOLDS in src/api/predictor.py (Low <15%, Medium 15–59%,
    # High 60–79%, Critical >=80%). For tolerance the interpretation is inverted
    # by HFPredictor.predict_tolerance.
    clinical_tiers = {
        "medium_risk": 0.15,
        "high_risk": 0.60,
        "critical_risk": 0.80,
    }

    # Tolerance-specific tier thresholds for direct tolerance scoring
    tolerance_tiers = {
        "high_tolerance": 0.85,
        "moderate_tolerance": 0.65,
        "low_tolerance": 0.35,
        "very_low_tolerance": 0.0,
    }

    # Save best model
    model_path = Path(MODEL_DIR) / f"best_model{model_suffix}.pkl"
    bundle = {
        "model": best_model,
        "model_name": best_name,
        "feature_cols": feature_cols,
        "metrics": best_metrics,
        "data_source": data_label,
        "prediction_target": prediction_target,
        "clinical_tiers": clinical_tiers,
    }
    if use_tolerance:
        bundle["tolerance_tiers"] = tolerance_tiers
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"Saved best model to {model_path}")

    # Save feature columns for API
    with open(Path(MODEL_DIR) / f"feature_cols{model_suffix}.json", "w") as f:
        json.dump(feature_cols, f)

    print("\nDone.")


if __name__ == "__main__":
    main()
