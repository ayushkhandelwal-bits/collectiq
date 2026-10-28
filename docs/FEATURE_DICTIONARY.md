# Feature Dictionary

All features are built in `src/features/build_features.py`. Every feature uses only
information available at the prediction timestamp.

**Prediction timestamps.**
* Model A — month-end snapshot date *t* for an account at 1-89 DPD. Target window *t+1 … t+3*.
* Models B / C — decision at the start of month *t*; features as of month-end *t−1*;
  collection history counts only events whose 30-day outcome windows had closed before
  the decision date. Outcome: cash received during month *t*.

Missing values: numeric features are median-imputed **inside** the model pipeline
(fitted on training data only) with a missing-indicator column; categoricals use the
training mode. Only `employment_length_years` (≈4%) and `prior_ptp_fulfilled_rate`
(undefined until a promise has been made) are missing in practice.

## Origination features (available at every later date)
| Feature | Definition / calculation | Source fields | Business interpretation |
|---|---|---|---|
| `bureau_score` | Synthetic bureau-style score at origination | loan_accounts | Higher = better credit history |
| `risk_grade` | A–E band of the score (one-hot) | loan_accounts | Origination risk tier |
| `loan_product` | Product (one-hot) | loan_accounts | Product-specific behaviour |
| `interest_rate` | % p.a. | loan_accounts | Price of risk |
| `loan_term_months` | Contract term | loan_accounts | Longer terms, longer exposure |
| `debt_to_income_ratio` | (other debt + this EMI) ÷ monthly income | loan_accounts | Repayment burden |
| `payment_to_income_ratio` | EMI × 12 ÷ annual income | loan_accounts | Affordability of this loan |
| `log_annual_income` | ln(annual income) | loan_accounts | Capacity (log for skew) |
| `employment_length_years` | Years employed | loan_accounts | Stability |
| `credit_history_length_months` | Age of oldest line | loan_accounts | Depth of history |
| `credit_utilization_ratio` | Revolving utilisation | loan_accounts | Credit stress |
| `prior_delinquency_count` | Bureau delinquencies before origination | loan_accounts | Past behaviour |
| `open_credit_accounts` | Open lines | loan_accounts | Credit appetite |
| `log_loan_amount` | ln(loan amount) | loan_accounts | Size of obligation |

## Behavioural features (Model A; observation window shown)
| Feature | Definition | Window | Interpretation |
|---|---|---|---|
| `months_on_book` | Months since origination | to *t* | Seasoning |
| `days_past_due` | DPD at *t* | *t* | Current severity |
| `bucket_ordinal` | 0 Current … 4 90+ | *t* | Stage |
| `missed_last_3m`, `missed_last_6m` | Count of months with an unpaid instalment | 3 / 6 months to *t* | Recent payment failures |
| `max_dpd_6m` | Maximum DPD | 6 months to *t* | Worst recent severity |
| `times_30plus_12m` | Months at 30+ DPD | 12 months to *t* | Chronic delinquency |
| `months_since_last_missed` | Months since last missed instalment (99 if never) | to *t* | Recency of stress |
| `payment_ratio_3m` | Paid ÷ scheduled (capped at 3) | 3 months to *t* | Payment consistency |
| `outstanding_ratio` | Outstanding ÷ original principal | *t* | Amortisation progress |
| `remaining_term_ratio` | Remaining ÷ total term | *t* | Time left on the loan |

## Collection features (Models B, C)
| Feature | Definition | Window | Interpretation |
|---|---|---|---|
| `stage` | Bucket at *t−1* (one-hot) | *t−1* | Collections stage |
| `action_type` | Decision: one of five actions or **No action** (one-hot) | decision | Lets the model estimate cash with and without an action |
| `months_since_last_payment` | Months since any cash received | to *t−1* | Engagement |
| `log_collectible_balance` | ln(1 + outstanding / remaining charged-off balance) | *t−1* | What can be collected |
| `log_amount_due` | ln(1 + arrears) ; for charged-off accounts the balance | *t−1* | Size of the ask |
| `log_emi` | ln(EMI) | origination | Instalment size |
| `months_since_charge_off` | Months since charge-off (−1 if not charged off) | *t−1* | Age of recovery |
| `prior_events_90d` | Collection actions in previous 90 days | 90 days before decision | Contact intensity |
| `prior_ptp_count` | Promises to pay in closed outcome windows | before decision | Engagement history |
| `prior_ptp_fulfilled_rate` | Fulfilled ÷ promises (closed windows only) | before decision | Reliability of promises |

## Excluded fields (leakage or policy)
`loan_status`, `current_outstanding_principal`, `charge_off_date`,
`charged_off_principal`, `final_recovery_amount`, `final_recovery_date`,
`contact_outcome`, `promise_to_pay_*` and `promise_fulfilled_indicator` of the event
itself, `subsequent_payment_amount`, all recovery-outcome fields, and `region`
(excluded by fairness policy; monitored only). Enforced by
`assert_no_leakage` and a data-quality check on every run.
