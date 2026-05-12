#!/usr/bin/env python3
"""
Runtime uncertainty router for anomaly detection predictions.

Implements a confidence thresholding layer that routes predictions into
three buckets based on model probability output:

  P < low_threshold   → NORMAL   (confident non-anomaly)
  low <= P <= high    → REVIEW   (uncertain — routed to human review queue)
  P > high_threshold  → ANOMALY  (confident anomaly)

This is a production-grade safety mechanism: uncertain predictions are never
auto-flagged as anomalies. They are queued for human review with full context.

Usage:
    from scripts.uncertainty_router import UncertaintyRouter

    router = UncertaintyRouter(low_threshold=0.40, high_threshold=0.60)
    decisions = router.route(model, X_test, timestamps=df["datetime"])
    router.save_review_queue("human_review_queue.json")
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


DECISION_NORMAL = "NORMAL"
DECISION_REVIEW = "HUMAN_REVIEW"
DECISION_ANOMALY = "ANOMALY"


@dataclass
class PredictionRecord:
    index: int
    timestamp: str
    probability: float
    decision: str
    features: list[float] = field(default_factory=list)


@dataclass
class RoutingReport:
    total: int
    normal_count: int
    review_count: int
    anomaly_count: int
    review_rate: float
    anomaly_rate: float
    low_threshold: float
    high_threshold: float
    routed_at: str

    def print_summary(self) -> None:
        print(f"\n{'='*50}")
        print("Uncertainty Routing Report")
        print(f"{'='*50}")
        print(f"Total predictions : {self.total:,}")
        print(f"  NORMAL          : {self.normal_count:,} ({self.normal_count/self.total:.1%})")
        print(f"  HUMAN_REVIEW    : {self.review_count:,} ({self.review_rate:.1%})")
        print(f"  ANOMALY         : {self.anomaly_count:,} ({self.anomaly_rate:.1%})")
        print(f"Uncertainty zone  : [{self.low_threshold}, {self.high_threshold}]")
        print(f"{'='*50}\n")


class UncertaintyRouter:
    """Routes model predictions based on confidence thresholds."""

    def __init__(self, low_threshold: float = 0.40, high_threshold: float = 0.60):
        if not (0 < low_threshold < high_threshold < 1):
            raise ValueError("Need 0 < low_threshold < high_threshold < 1")
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self._review_queue: list[PredictionRecord] = []
        self._all_records: list[PredictionRecord] = []

    def route(
        self,
        probabilities: np.ndarray,
        timestamps: Any = None,
        features: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Route predictions given anomaly probabilities.

        Returns array of decisions: "NORMAL", "HUMAN_REVIEW", or "ANOMALY".
        Uncertain predictions are also appended to the internal review queue.
        """
        probabilities = np.asarray(probabilities, dtype=float)
        n = len(probabilities)
        decisions = np.empty(n, dtype=object)

        for i, prob in enumerate(probabilities):
            ts = str(timestamps.iloc[i]) if hasattr(timestamps, "iloc") else (
                timestamps[i] if timestamps is not None else datetime.now(timezone.utc).isoformat()
            )

            if prob < self.low_threshold:
                decision = DECISION_NORMAL
            elif prob > self.high_threshold:
                decision = DECISION_ANOMALY
            else:
                decision = DECISION_REVIEW

            decisions[i] = decision
            rec = PredictionRecord(
                index=i,
                timestamp=str(ts),
                probability=float(prob),
                decision=decision,
                features=features[i].tolist() if features is not None else [],
            )
            self._all_records.append(rec)
            if decision == DECISION_REVIEW:
                self._review_queue.append(rec)

        return decisions

    def route_from_model(
        self,
        model,
        X: np.ndarray,
        timestamps: Any = None,
    ) -> np.ndarray:
        """Convenience method: extract probabilities from a sklearn model then route."""
        if not hasattr(model, "predict_proba"):
            raise TypeError(f"{type(model).__name__} does not support predict_proba")
        probas = model.predict_proba(X)[:, 1]
        return self.route(probas, timestamps=timestamps, features=X)

    def get_report(self) -> RoutingReport:
        total = len(self._all_records)
        if total == 0:
            raise RuntimeError("No predictions routed yet — call route() first")
        review_count = sum(1 for r in self._all_records if r.decision == DECISION_REVIEW)
        anomaly_count = sum(1 for r in self._all_records if r.decision == DECISION_ANOMALY)
        normal_count = total - review_count - anomaly_count
        return RoutingReport(
            total=total,
            normal_count=normal_count,
            review_count=review_count,
            anomaly_count=anomaly_count,
            review_rate=review_count / total,
            anomaly_rate=anomaly_count / total,
            low_threshold=self.low_threshold,
            high_threshold=self.high_threshold,
            routed_at=datetime.now(timezone.utc).isoformat(),
        )

    def save_review_queue(self, path: str | Path = "human_review_queue.json") -> None:
        """Persist the uncertain-prediction queue to JSON for downstream human review."""
        path = Path(path)
        payload = {
            "metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "low_threshold": self.low_threshold,
                "high_threshold": self.high_threshold,
                "queue_size": len(self._review_queue),
            },
            "items": [asdict(r) for r in self._review_queue],
        }
        path.write_text(json.dumps(payload, indent=2))
        print(f"Review queue ({len(self._review_queue)} items) saved to {path}")

    def clear(self) -> None:
        self._review_queue.clear()
        self._all_records.clear()


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from pathlib import Path as P
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from scripts.eval_pipeline import (
        FEATURE_COLS,
        add_rolling_features,
        load_all_data,
        time_series_split,
    )

    data_dir = P("data/valve1")
    df = load_all_data(data_dir)
    df = add_rolling_features(df)
    feature_cols = FEATURE_COLS + [f"{c}_roll_mean" for c in FEATURE_COLS] + [f"{c}_roll_std" for c in FEATURE_COLS]
    X = df[feature_cols].fillna(0).values
    y = df["anomaly"].values.astype(int)

    X_train, X_test, y_train, y_test = time_series_split(X, y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train_s, y_train)

    router = UncertaintyRouter(low_threshold=0.40, high_threshold=0.60)
    decisions = router.route_from_model(clf, X_test_s)

    report = router.get_report()
    report.print_summary()
    router.save_review_queue("human_review_queue.json")
