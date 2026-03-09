# HW2: Model and Data Versioning with DVC

**Branch:** `hw2-regression`  
**Task:** Linear Regression predicting `fare_amount` (NYC Green Taxi data)  
**Remote storage:** Google Drive via DVC   https://drive.google.com/drive/folders/1uzwIN61eUIedLOM53jprkR6HhfL-9e3J?usp=drive_link
**GitHub:** https://github.com/maxsebastianS/MLOPS_HW1

---

## Setup

```bash
pip install dvc dvc-gdrive
dvc init
dvc remote add -d storage gdrive://1uzwIN61eUIedLOM53jprkR6HhfL-9e3J
```

---

## Version 1 — Baseline

**Data:** January 2021 (`green_tripdata_2021-01_cleaned.parquet`, 40,336 rows)  
**Model:** LinearRegression trained on Jan 2021, 4 features: `trip_distance`, `passenger_count`, `RatecodeID`, `extra`  
**Git tag:** `v1`

| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| MAE | 2.6687 | 2.6430 | 2.6632 |
| RMSE | 7.3813 | 7.7280 | 7.1377 |
| R² | 0.7606 | 0.7206 | 0.7548 |

**DVC-tracked:** `green_tripdata_2021-01_cleaned.parquet`, `regression_model.pkl`

---

## Version 2 — New Data, Same Model

**Data:** Jan + Feb 2021 combined (`green_tripdata_combined_cleaned.parquet`, 75,694 rows)  
**Model:** Unchanged V1 model evaluated on new test set  
**Code changed:** `clean_data_v2.py` (combine data), `evaluate_v2.py` (evaluate V1 model)  
**Git tag:** `v2`

| Metric | V1 Test | V2 Test | Change |
|--------|---------|---------|--------|
| MAE | 2.6632 | 2.6440 | -0.72% |
| RMSE | 7.1377 | 6.7726 | -5.12% |
| R² | 0.7548 | 0.7929 | +5.05% |

**DVC-tracked:** `green_tripdata_combined_cleaned.parquet`

### What is happening?

The V1 model performs **better** on the combined test set than on the original January-only test set. RMSE dropped ~5% and R² improved ~5%. This means February 2021 taxi data follows the same patterns as January — the model generalises well to the new month. The larger test set also reduces evaluation variance. There is no sign of distribution shift between the two months.

---

## Version 3 — Retrained Model

**Data:** Same combined Jan + Feb dataset as V2  
**Model:** New LinearRegression retrained on combined training set (45,416 samples)  
**Code changed:** `regression_model_v3.py`  
**Git tag:** `v3`

| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| MAE | 2.6497 | 2.6597 | 2.6562 |
| RMSE | 6.4520 | 6.8826 | 6.7613 |
| R² | 0.8031 | 0.7719 | 0.7936 |

**Comparison to V2 (same test set):**

| Metric | V2 (V1 model) | V3 (retrained) | Change |
|--------|---------------|----------------|--------|
| MAE | 2.6440 | 2.6562 | +0.0122 |
| RMSE | 6.7726 | 6.7613 | -0.0113 |
| R² | 0.7929 | 0.7936 | +0.0007 |

**DVC-tracked:** `regression_model_v3.pkl`

### What is happening?

Retraining on the combined dataset produces virtually identical results to the V1 model on the same test set. The differences are negligible (RMSE difference of 0.01). This confirms the two months share the same data distribution — the V1 model was already a good fit, and adding February data did not meaningfully change the learned coefficients.

---

## Production Monitoring (V3 deployed)

### 1. Data metric — Feature distribution drift

Monitor the distribution of input features (especially `trip_distance`) in live prediction requests. Use PSI (Population Stability Index) or a KS-test run daily to detect if the incoming data distribution shifts away from the training distribution. A PSI > 0.2 on `trip_distance` would signal a meaningful change in taxi trip patterns.

### 2. Model performance metric — Rolling MAE

After each trip completes, the actual `fare_amount` becomes available. Compute a 7-day rolling MAE on labelled production data. If MAE rises above 3.5 (roughly 30% above the V3 baseline of 2.66) for 3 or more consecutive days, the model is underperforming and needs attention.

### 3. System metric — Prediction latency (p95)

Monitor the 95th percentile response time for prediction API calls. A sustained p95 > 500ms indicates infrastructure issues (memory pressure, CPU saturation, etc.) that could affect service reliability.

### Retrain vs. Roll back

- **Retrain** when: rolling MAE exceeds threshold for 3+ days, or PSI > 0.2 on key features — indicating a genuine distribution shift that more recent training data can address.
- **Roll back to V1** when: V3 shows a sudden large performance drop immediately after deployment (e.g., MAE > 5.0), suggesting a bug in the V3 training pipeline rather than a data drift issue.
