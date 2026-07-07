"""Run the validation protocol from ``VALIDATION.md`` on a real feature matrix.

Unlike ``src/models/train.py`` (which watches the test set during training and
selects best-of-two on it), this harness enforces a **three-way, patient-level
split** and touches the held-out test set exactly once, after all model and
threshold selection is done on train+validation. It reports discrimination,
calibration (with post-hoc isotonic recalibration), and baseline comparisons with
patient-level bootstrap confidence intervals.

The functions are data-source-agnostic — they operate on a features parquet with a
``patient_id`` group column and a binary ``label`` column, the schema emitted by
``src/features/build_features.py``. ``main()`` is guarded to the real validation
cohort (``$HFTA_DATA_DIR/processed/features_mimic_full.parquet``) so it is never
confused with the demo pipeline. See ``VALIDATION.md`` for the full protocol.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

GROUP_COL = "patient_id"
LABEL_COL = "label"
NON_FEATURE_COLS = {GROUP_COL, LABEL_COL, "prediction_day"}
RNG_SEED = 42


def patient_level_split(df: pd.DataFrame, seed: int = RNG_SEED):
    """Grouped train / validation / test split (0.70 / 0.15 / 0.15 by patient)."""
    groups = df[GROUP_COL].values
    idx = np.arange(len(df))

    outer = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    trainval_idx, test_idx = next(outer.split(idx, groups=groups))

    tv = df.iloc[trainval_idx]
    inner = GroupShuffleSplit(n_splits=1, test_size=0.1765, random_state=seed)
    tr_rel, val_rel = next(inner.split(np.arange(len(tv)), groups=tv[GROUP_COL].values))

    train = tv.iloc[tr_rel]
    val = tv.iloc[val_rel]
    test = df.iloc[test_idx]
    assert not (set(train[GROUP_COL]) & set(test[GROUP_COL]))  # no patient overlap
    assert not (set(val[GROUP_COL]) & set(test[GROUP_COL]))
    return train, val, test


def _xy(df: pd.DataFrame, feature_cols):
    return df[feature_cols].values, df[LABEL_COL].astype(int).values


def bootstrap_auroc_ci(y, p, groups, n_boot: int = 1000, seed: int = RNG_SEED):
    """Patient-level bootstrap 95% CI for AUROC (resample patients, not rows)."""
    rng = np.random.default_rng(seed)
    y, p, groups = np.asarray(y), np.asarray(p), np.asarray(groups)
    uniq = np.unique(groups)
    by_group = {g: np.where(groups == g)[0] for g in uniq}
    scores = []
    for _ in range(n_boot):
        sampled = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([by_group[g] for g in sampled])
        if len(np.unique(y[rows])) < 2:
            continue
        scores.append(roc_auc_score(y[rows], p[rows]))
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return float(lo), float(hi)


def calibration_slope_intercept(y, p):
    """Logistic recalibration slope/intercept (1.0 / 0.0 == perfectly calibrated)."""
    eps = 1e-6
    logit = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    lr = LogisticRegression(penalty=None, solver="lbfgs")
    lr.fit(logit.reshape(-1, 1), y)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def evaluate(y, p, groups, label: str) -> dict:
    return {
        "cohort": label,
        "n": int(len(y)),
        "prevalence": float(np.mean(y)),
        "auroc": float(roc_auc_score(y, p)),
        "auroc_ci95": bootstrap_auroc_ci(y, p, groups),
        "auprc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "calibration_slope_intercept": calibration_slope_intercept(y, p),
    }


def fit_candidate(Xtr, ytr):
    """Gradient-boosted trees (the model under test)."""
    import xgboost as xgb

    pos = max(1, int(ytr.sum()))
    spw = float((len(ytr) - pos) / pos)
    clf = xgb.XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
        eval_metric="aucpr", random_state=RNG_SEED,
    )
    clf.fit(Xtr, ytr)
    return clf


def fit_baselines(Xtr, ytr):
    """Baseline B: penalised logistic regression on standardised features."""
    scaler = StandardScaler().fit(Xtr)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(scaler.transform(Xtr), ytr)
    return ("logistic_regression", lambda X: lr.predict_proba(scaler.transform(X))[:, 1])


def run(features_path: Path) -> dict:
    df = pd.read_parquet(features_path)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    train, val, test = patient_level_split(df)

    Xtr, ytr = _xy(train, feature_cols)
    Xval, yval = _xy(val, feature_cols)
    Xte, yte = _xy(test, feature_cols)

    candidate = fit_candidate(Xtr, ytr)
    base_name, base_predict = fit_baselines(Xtr, ytr)

    # Post-hoc isotonic recalibration, fit on validation, applied to test.
    p_val = candidate.predict_proba(Xval)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_val, yval)
    p_test_raw = candidate.predict_proba(Xte)[:, 1]
    p_test_cal = iso.predict(p_test_raw)

    groups_te = test[GROUP_COL].values
    report = {
        "features_path": str(features_path),
        "n_patients": int(df[GROUP_COL].nunique()),
        "split": {k: int(v[GROUP_COL].nunique()) for k, v in
                  {"train": train, "val": val, "test": test}.items()},
        "candidate_gbt_raw": evaluate(yte, p_test_raw, groups_te, "gbt (uncalibrated)"),
        "candidate_gbt_calibrated": evaluate(yte, p_test_cal, groups_te, "gbt (isotonic)"),
        "baseline_" + base_name: evaluate(
            yte, base_predict(Xte), groups_te, base_name),
    }
    return report


def main() -> None:
    base = Path(os.environ.get("HFTA_DATA_DIR", "data"))
    features_path = base / "processed" / "features_mimic_full.parquet"
    if not features_path.exists():
        raise RuntimeError(
            f"{features_path} not found. Build the real cohort first "
            "(src/validation/build_cohort.py, then src/features/build_features.py), "
            "using credentialed MIMIC-IV. See VALIDATION.md — no real data ships here."
        )
    report = run(features_path)
    out = base.parent / "models" / "results_validation" / "report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
