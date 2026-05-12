#!/usr/bin/env python3
"""
Automated evaluation pipeline for SKAB anomaly detection.

Runs IsolationForest, XGBoost, and RandomForest against all labeled SKAB
CSVs, logs metrics to MLflow, computes bootstrapped F1 confidence intervals,
and exits with code 1 if the best F1 drops below the regression threshold.

Usage:
    python scripts/eval_pipeline.py
    python scripts/eval_pipeline.py --threshold 0.80 --data-dir data/valve1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import bootstrap
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("WARNING: mlflow not installed. Install with: pip install mlflow")

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("WARNING: xgboost not installed. Skipping XGBoost model.")


FEATURE_COLS = [
    "Accelerometer1RMS", "Accelerometer2RMS", "Current",
    "Pressure", "Temperature", "Thermocouple",
    "Voltage", "Volume Flow RateRMS",
]
REGRESSION_RESULTS_PATH = Path("eval_results.json")
MLFLOW_EXPERIMENT = "skab-anomaly-detection"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_skab_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def add_rolling_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    df = df.copy()
    for col in FEATURE_COLS:
        df[f"{col}_roll_mean"] = df[col].rolling(window, min_periods=1).mean()
        df[f"{col}_roll_std"] = df[col].rolling(window, min_periods=1).std().fillna(0)
    return df


def load_all_data(data_dir: Path) -> pd.DataFrame:
    """Load and concatenate all labeled CSVs from data_dir, sorted by datetime."""
    dfs = []
    for p in sorted(data_dir.glob("*.csv")):
        if "unlabelled" in p.name:
            continue
        df = load_skab_csv(p)
        if "anomaly" not in df.columns:
            continue
        df["source_file"] = p.name
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"No labeled CSVs found in {data_dir}")
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values("datetime").reset_index(drop=True)
    return combined


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    df = add_rolling_features(df)
    feature_cols = FEATURE_COLS + [
        f"{c}_roll_mean" for c in FEATURE_COLS
    ] + [f"{c}_roll_std" for c in FEATURE_COLS]
    X = df[feature_cols].fillna(0).values
    y = df["anomaly"].values.astype(int)
    return X, y


def time_series_split(X: np.ndarray, y: np.ndarray, test_ratio: float = 0.2):
    """Chronological split — no shuffling to avoid data leakage."""
    n = len(X)
    split = int(n * (1 - test_ratio))
    return X[:split], X[split:], y[:split], y[split:]


# ---------------------------------------------------------------------------
# Bootstrap confidence interval for F1
# ---------------------------------------------------------------------------

def bootstrap_f1_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_resamples: int = 1000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Return (lower, upper) bootstrap CI for F1 score."""

    def f1_statistic(yt, yp):
        return np.array([f1_score(yt, yp, zero_division=0)])

    result = bootstrap(
        (y_true, y_pred),
        statistic=f1_statistic,
        n_resamples=n_resamples,
        confidence_level=confidence,
        paired=True,
        method="percentile",
        random_state=42,
    )
    return float(result.confidence_interval.low), float(result.confidence_interval.high)


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def get_models() -> dict:
    models = {
        "IsolationForest": None,  # unsupervised — handled separately
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1
        ),
    }
    if XGB_AVAILABLE:
        models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            scale_pos_weight=10,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )
    return models


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_isolation_forest(
    X_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
) -> dict:
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1)
    iso.fit(X_train_s)

    raw_scores = iso.decision_function(X_test_s)
    # Invert: lower score = more anomalous → higher "anomaly probability"
    scores = -raw_scores
    scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

    threshold = np.percentile(scores_norm, 95)
    preds = (scores_norm >= threshold).astype(int)

    ci_low, ci_high = bootstrap_f1_ci(y_test, preds)
    return {
        "f1": f1_score(y_test, preds, zero_division=0),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "auc_roc": roc_auc_score(y_test, scores_norm),
        "f1_ci_low": ci_low,
        "f1_ci_high": ci_high,
        "probas": scores_norm,
        "preds": preds,
    }


def evaluate_supervised(
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)
    probas = model.predict_proba(X_test_s)[:, 1]

    ci_low, ci_high = bootstrap_f1_ci(y_test, preds)
    return {
        "f1": f1_score(y_test, preds, zero_division=0),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "auc_roc": roc_auc_score(y_test, probas),
        "f1_ci_low": ci_low,
        "f1_ci_high": ci_high,
        "probas": probas,
        "preds": preds,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(data_dir: Path, threshold: float, test_ratio: float) -> dict:
    print(f"\n{'='*60}")
    print("SKAB Anomaly Detection — Evaluation Pipeline")
    print(f"{'='*60}")
    print(f"Data dir  : {data_dir}")
    print(f"F1 threshold (regression gate): {threshold}")
    print(f"Test ratio: {test_ratio:.0%}")

    df = load_all_data(data_dir)
    print(f"\nLoaded {len(df):,} rows | anomaly rate: {df['anomaly'].mean():.2%}")

    X, y = build_features(df)
    X_train, X_test, y_train, y_test = time_series_split(X, y, test_ratio)
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    models = get_models()
    results: dict[str, dict] = {}

    if MLFLOW_AVAILABLE:
        mlflow.set_experiment(MLFLOW_EXPERIMENT)

    for name, model in models.items():
        print(f"\n--- {name} ---")
        with (mlflow.start_run(run_name=name) if MLFLOW_AVAILABLE else _null_context()):
            if name == "IsolationForest":
                metrics = evaluate_isolation_forest(X_train, X_test, y_test)
            else:
                metrics = evaluate_supervised(model, X_train, X_test, y_train, y_test)

            loggable = {k: v for k, v in metrics.items() if k not in ("probas", "preds")}
            print(
                f"  F1={metrics['f1']:.4f} [{metrics['f1_ci_low']:.4f}–{metrics['f1_ci_high']:.4f} 95% CI]"
                f"  AUC={metrics['auc_roc']:.4f}"
                f"  P={metrics['precision']:.4f}  R={metrics['recall']:.4f}"
            )

            if MLFLOW_AVAILABLE:
                mlflow.log_params({"model": name, "test_ratio": test_ratio, "data_dir": str(data_dir)})
                mlflow.log_metrics(loggable)

            results[name] = loggable

    best_model = max(results, key=lambda k: results[k]["f1"])
    best_f1 = results[best_model]["f1"]

    print(f"\n{'='*60}")
    print(f"Best model: {best_model}  F1={best_f1:.4f}")

    REGRESSION_RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"Results saved to {REGRESSION_RESULTS_PATH}")

    if MLFLOW_AVAILABLE:
        print("MLflow UI: run `mlflow ui` and open http://localhost:5000")

    passed = best_f1 >= threshold
    status = "PASSED" if passed else "FAILED"
    print(f"\nRegression gate [{threshold}]: {status}")
    print(f"{'='*60}\n")

    return {"results": results, "best_model": best_model, "best_f1": best_f1, "passed": passed}


class _null_context:
    def __enter__(self): return self
    def __exit__(self, *_): pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SKAB anomaly detection eval pipeline")
    parser.add_argument("--data-dir", default="data/valve1", help="Path to labeled CSV directory")
    parser.add_argument("--threshold", type=float, default=0.80, help="Min F1 to pass regression gate")
    parser.add_argument("--test-ratio", type=float, default=0.20, help="Fraction of data for test set")
    args = parser.parse_args()

    outcome = run_pipeline(
        data_dir=Path(args.data_dir),
        threshold=args.threshold,
        test_ratio=args.test_ratio,
    )
    sys.exit(0 if outcome["passed"] else 1)


if __name__ == "__main__":
    main()
