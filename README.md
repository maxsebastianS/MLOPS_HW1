# ML Model Training Project

## Overview
This project trains and compares two machine learning models on NYC Green Taxi data:
1. **Regression Model** - Predicting fare amounts
2. **Classification Model** - Predicting payment type (Credit Card vs Cash)

Each model is trained in two versions:
- **V1**: Using 4 features (trip_distance, passenger_count, RatecodeID, extra)
- **V2**: Using 5 features (V1 features + hour of pickup)

---

## Regression Model Comparison

**Target**: fare_amount  
**Model Type**: Linear Regression

### V1 (4 Features)
| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| MAE | 2.6687 | 2.6430 | 2.6632 |
| RMSE | 7.3813 | 7.7280 | 7.1377 |
| R² Score | 0.7606 | 0.7206 | 0.7548 |

### V2 (5 Features with Hour)
| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| MAE | 2.6514 | 2.6381 | 2.6587 |
| RMSE | 7.1947 | 7.5895 | 7.0652 |
| R² Score | 0.7657 | 0.7282 | 0.7604 |

### Improvement Analysis
**Test Set Changes:**
- MAE: 2.6632 → 2.6587 (-0.0045, -0.17%)
- RMSE: 7.1377 → 7.0652 (-0.0725, -1.02%)
- R² Score: 0.7548 → 0.7604 (+0.0056, +0.74%)

**Conclusion**: Small positive improvement. Hour feature provides modest gain in RMSE (~1% reduction). R² improvement is small (+0.74%).

---

## Classification Model Comparison

**Target**: payment_type (1=Credit Card, 2=Cash)  
**Model Type**: Logistic Regression  
**Note**: Binary classification using only payment types 1 and 2

### V1 (4 Features)
| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| Accuracy | 0.6098 | 0.6099 | 0.6080 |
| Precision | 0.6103 | 0.6101 | 0.6093 |
| Recall | 0.9950 | 0.9963 | 0.9939 |
| F1-Score | 0.7565 | 0.7568 | 0.7554 |
| ROC-AUC | 0.5877 | 0.5875 | 0.5914 |

**Confusion Matrix (Test)**:
```
True Negatives:  21
False Positives: 3115
False Negatives: 30
True Positives:  4857
```

### V2 (5 Features with Hour)
| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| Accuracy | 0.6319 | 0.6292 | 0.6213 |
| Precision | 0.6325 | 0.6295 | 0.6218 |
| Recall | 0.9951 | 0.9955 | 0.9938 |
| F1-Score | 0.7706 | 0.7691 | 0.7638 |
| ROC-AUC | 0.6442 | 0.6411 | 0.6248 |

**Confusion Matrix (Test)**:
```
True Negatives:  176
False Positives: 2960
False Negatives: 30
True Positives:  4857
```

### Improvement Analysis
**Test Set Changes:**
- Accuracy: 0.6080 → 0.6213 (+0.0133, +2.19%)
- Precision: 0.6093 → 0.6218 (+0.0125, +2.05%)
- Recall: 0.9939 → 0.9938 (-0.0001, -0.01%)
- F1-Score: 0.7554 → 0.7638 (+0.0084, +1.11%)
- ROC-AUC: 0.5914 → 0.6248 (+0.0334, +5.65%)

**Conclusion**: Meaningful improvement in classification metrics. Hour feature provides substantial gains:
- ROC-AUC improvement of 5.65% (largest gain)
- Accuracy improvement of 2.19%
- Precision improvement of 2.05%
- Better discrimination (True Negatives increased 8x: 21 → 176)

---

## Is the Improvement Real or Due to Randomness?

### Statistical Significance

**Regression**:
- RMSE improvement of 1.02% is modest and could be within natural variance
- R² improvement of 0.74% is small
- Consistent improvement across train/val/test suggests some real effect, but magnitude is minor

**Classification**:
- ROC-AUC improvement of 5.65% is substantial and indicates real effect
- Accuracy improvement of 2.19% is meaningful  
- Consistent improvement across train/val/test splits supports significance
- True Negatives increased dramatically (8x), showing better class separation

**Verdict**: 
- **Regression**: Marginal improvement, likely within noise
- **Classification**: Moderate improvement, suggests hour is informative

---

## Data Leakage Analysis

### Does Hour Feature Leak Information?

**For Regression (fare_amount)**:
- Hour derived from lpep_pickup_datetime (captured when meter engaged)
- Fare calculated DURING the trip
- Hour is determined at START, before fare is finalized
- **NO LEAKAGE**: Hour known before fare calculated
- Note: Hour legitimately influences pricing (peak hour surcharges)

**For Classification (payment_type)**:
- Hour derived from lpep_pickup_datetime (at pickup)
- Payment type recorded AFTER trip ends
- Hour cannot influence payment method choice
- **NO LEAKAGE**: Hour independent of payment choice
- Note: Hour may correlate with passenger behavior patterns

---

## Production Risks for Hour Feature

### Risk 1: Time Zone Handling
- Model trained using specific timezone
- Different regions may use different time zones
- **Risk**: Model fails when applied to different timezone data
- **Mitigation**: Document timezone assumption, standardize before production

### Risk 2: Daylight Saving Time (DST)
- Dataset may include DST transitions
- Same "hour" value means different times of day in different months
- **Risk**: DST transitions cause inconsistent hour encoding
- **Mitigation**: Use UTC time before extracting hour

### Risk 3: Missing Datetime Data
- Hour feature requires lpep_pickup_datetime to be available
- If datetime field missing or corrupted, hour cannot be computed
- **Risk**: Missing data causes NaN values and model failures
- **Mitigation**: Validate datetime presence, set default behavior

### Risk 4: Temporal Distribution Shift
- Model trained on January 2021 data with specific rush hour patterns
- Production data may have different temporal patterns
- **Risk**: Performance degrades if rush hours change (work-from-home, emergencies, etc.)
- **Mitigation**: Monitor hour distribution, retrain periodically

### Risk 5: Feature Dependency
- Hour depends on lpep_pickup_datetime being reliably captured
- Data collection failures break this feature
- **Risk**: Upstream data quality issues propagate
- **Mitigation**: Monitor source data quality, validate timestamps

---

## GitHub Repository
https://github.com/maxsebastianS/MLOPS_HW1
