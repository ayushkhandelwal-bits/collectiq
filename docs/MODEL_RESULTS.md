# Model and Optimisation Results

Generated from the evaluation reports of pipeline run `PR-dfd4d7907f34` (profile `full`, dataset version `b96544ef2277e4e7`). All numbers are computed on **synthetic** data on a chronologically held-out test period that was not used for model or threshold selection.

## Model A — roll-forward risk

Selected: **XGBoost**. Splits — train 17,719 (2023-02-28 to 2025-03-31), validation 10,152 (2025-04-30 to 2025-09-30), test 11,158 (2025-10-31 to 2026-03-31).

| Model | Valid ROC-AUC | Test ROC-AUC | Test PR-AUC | Precision | Recall | F1 | Brier | Threshold |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.8502 | 0.8448 | 0.6748 | 0.4848 | 0.8566 | 0.6192 | 0.1356 | 0.164 |
| XGBoost (selected) | 0.8617 | 0.8546 | 0.6794 | 0.4707 | 0.8916 | 0.6161 | 0.1325 | 0.138 |
| Random Forest | 0.8557 | 0.8497 | 0.6802 | 0.4648 | 0.8841 | 0.6093 | 0.1328 | 0.159 |

Test prevalence: 27.4%.

## Model B — payment propensity

Selected: **Logistic Regression**. Splits — train 33,198 (2023-03-01 to 2025-06-01), validation 21,126 (2025-07-01 to 2025-12-01), test 27,677 (2026-01-01 to 2026-06-01).

| Model | Valid ROC-AUC | Test ROC-AUC | Test PR-AUC | Precision | Recall | F1 | Brier | Threshold |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression (selected) | 0.8173 | 0.8310 | 0.6459 | 0.5507 | 0.8757 | 0.6761 | 0.1508 | 0.340 |
| XGBoost | 0.8125 | 0.8271 | 0.6382 | 0.5436 | 0.8847 | 0.6734 | 0.1522 | 0.318 |

Test prevalence: 31.5%.

## Model C — payment amount given payment (hurdle)

Selected: **XGBoost Regressor** (on validation MAE of expected cash).

| Model | Test MAE given payment (₹) | Test RMSE given payment (₹) | Test R² (log) | Test MAE expected cash (₹) | Σ predicted / Σ actual |
|---|---|---|---|---|---|
| Ridge Regression | 11,564 | 22,294 | 0.706 | 8,093 | 1.041 |
| XGBoost Regressor (selected) | 11,761 | 24,139 | 0.697 | 8,050 | 1.013 |

## Default optimisation run

Status **OPTIMAL** (Optimal within relative MIP gap 0.05%.). Budget utilisation 82.2%, capacity utilisation 100.0%.

| Strategy | Accounts | Est. incremental gross (₹) | Cost (₹) | Est. net (₹) | Optimised advantage (₹) |
|---|---|---|---|---|---|
| Optimised plan (MILP) | 2,812 | 4,249,628 | 328,875 | 3,920,753 | — |
| Balance-first | 3,133 | 3,413,067 | 347,755 | 3,065,312 | 855,441 |
| Risk-first | 3,142 | 3,296,225 | 347,935 | 2,948,290 | 972,463 |
| Oldest-delinquency-first | 3,150 | 2,573,064 | 348,095 | 2,224,969 | 1,695,784 |
| ENRV ranking (greedy) | 1,497 | 3,674,084 | 362,155 | 3,311,929 | 608,824 |

These are model-based estimates on synthetic data. They do not show that the optimised plan would improve real-world recoveries.
