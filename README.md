# Anomaly Detection MLOps Pipeline

End-to-end MLOps pipeline for real-time anomaly detection on industrial sensor data — includes automated evaluation, CI/CD regression testing, and runtime uncertainty controls.

**Dataset**: [SKAB (Skoltech Anomaly Benchmark)](https://github.com/waico/SKAB) — water pump sensor readings with labeled anomalies and changepoints.

---

## What this project demonstrates

| MLOps Capability | Implementation |
|---|---|
| Automated eval pipeline | `scripts/eval_pipeline.py` — reruns all models, logs to MLflow |
| CI/CD regression gate | `.github/workflows/eval.yml` — fails build if F1 < 0.80 |
| Runtime safety guardrails | `scripts/uncertainty_router.py` — routes uncertain predictions to human review |
| Statistical rigor | Bootstrapped 95% CI on F1, time-series-aware train/test split |

---

## Architecture

```
data/valve1/          ← SKAB labeled sensor CSVs
    ├── 1.csv ... 15.csv
scripts/
    ├── eval_pipeline.py       ← Automated eval + MLflow logging
    └── uncertainty_router.py  ← Confidence thresholding layer
.github/workflows/
    └── eval.yml               ← CI/CD regression gate
eval_results.json              ← Latest benchmark results
human_review_queue.json        ← Uncertain predictions for human review
```

---

## Quickstart

```bash
pip install -r requirements.txt

# Run the full eval pipeline
python scripts/eval_pipeline.py

# View results in MLflow UI
mlflow ui   # then open http://localhost:5000

# Run uncertainty routing demo
python scripts/uncertainty_router.py
```

---

## 1. Automated Evaluation Pipeline

`scripts/eval_pipeline.py` benchmarks three models against all labeled SKAB sensor files:

- **IsolationForest** (unsupervised baseline)
- **RandomForest** (supervised, class-balanced)
- **XGBoost** (supervised, gradient boosted)

### Time-series-aware split

Data is split **chronologically** — the last 20% of timestamped rows form the test set. No shuffling, no leakage.

```
[──────────── TRAIN (80%) ─────────────][── TEST (20%) ──]
     sorted by datetime →
```

### Bootstrapped confidence intervals

F1 scores are reported with **95% bootstrap confidence intervals** (1000 resamples, percentile method). This quantifies uncertainty in the metric itself — a single F1 number without a CI is not statistically meaningful on small anomaly-rate datasets.

### MLflow experiment tracking

Every run logs to a local MLflow experiment (`skab-anomaly-detection`):

```
mlflow ui   →   http://localhost:5000
```

Logged per run: `f1`, `precision`, `recall`, `auc_roc`, `f1_ci_low`, `f1_ci_high`, `model`, `test_ratio`.

### Sample output

```
==============================
SKAB Anomaly Detection — Evaluation Pipeline
==============================
Loaded 28,472 rows | anomaly rate: 11.43%
Train: 22,777 | Test: 5,695

--- IsolationForest ---
  F1=0.6821 [0.6441–0.7198 95% CI]  AUC=0.8134  P=0.5917  R=0.8049

--- RandomForest ---
  F1=0.8763 [0.8512–0.9011 95% CI]  AUC=0.9641  P=0.8801  R=0.8726

--- XGBoost ---
  F1=0.8834 [0.8591–0.9074 95% CI]  AUC=0.9712  P=0.8967  R=0.8705

Best model: XGBoost  F1=0.8834
Regression gate [0.80]: PASSED
```

---

## 2. CI/CD Regression Gate

`.github/workflows/eval.yml` runs automatically on every push and pull request to `main`.

**The gate**: if the best model F1 drops below **0.80**, the workflow exits with code 1 — the build fails and the commit is blocked from merge.

```
push to main
    └── GitHub Actions
            └── python scripts/eval_pipeline.py --threshold 0.80
                    ├── F1 ≥ 0.80 → ✅ green check
                    └── F1 < 0.80 → ❌ build failed
```

Eval results are uploaded as a workflow artifact (`eval_results.json`) on every run, including failures — so regressions are always traceable.

---

## 3. Runtime Uncertainty Controls

`scripts/uncertainty_router.py` implements a **confidence thresholding layer** that wraps any sklearn model's `predict_proba` output:

```
Anomaly probability
       │
       ├── P < 0.40  →  ✅  NORMAL    (auto-clear)
       │
       ├── 0.40 ≤ P ≤ 0.60  →  🔍  HUMAN_REVIEW  (queued)
       │
       └── P > 0.60  →  🚨  ANOMALY   (auto-flag)
```

Predictions in the uncertain zone (0.40–0.60) are **never auto-flagged**. They are written to `human_review_queue.json` with full context (timestamp, probability, feature values) for operator review.

This prevents the model from confidently mis-classifying borderline cases — a critical requirement for any safety-adjacent production system.

```python
from scripts.uncertainty_router import UncertaintyRouter

router = UncertaintyRouter(low_threshold=0.40, high_threshold=0.60)
decisions = router.route_from_model(model, X_test)
router.save_review_queue("human_review_queue.json")
```

---

## 4. Statistical Design Choices

| Decision | Rationale |
|---|---|
| Chronological train/test split | Time-series data has temporal dependencies — random splitting leaks future signal into training |
| `class_weight="balanced"` on RF | Anomaly rate ~11% — without balancing, models optimise for majority class and miss anomalies |
| Bootstrap CI over single F1 | Small test sets + class imbalance make point estimates misleading; CI exposes this |
| IsolationForest as baseline | Provides unsupervised reference — useful when labels are unavailable on new sensor installations |
| 95th percentile threshold on IF scores | Matches expected ~5% contamination; avoids tuning on test data |

---

## Results

| Model | F1 | 95% CI | AUC-ROC |
|---|---|---|---|
| IsolationForest | ~0.68 | ±0.04 | ~0.81 |
| RandomForest | ~0.88 | ±0.03 | ~0.96 |
| **XGBoost** | **~0.88** | **±0.02** | **~0.97** |

*Exact numbers vary by run — see `eval_results.json` for the latest.*

---

## Dataset

SKAB contains multivariate time-series sensor data from a water pump test bench:

- **Features**: Accelerometer (×2), Current, Pressure, Temperature, Thermocouple, Voltage, Volume Flow Rate
- **Labels**: `anomaly` (point anomaly), `changepoint` (regime change)
- **Source**: [waico/SKAB on GitHub](https://github.com/waico/SKAB)

---

## License

MIT
