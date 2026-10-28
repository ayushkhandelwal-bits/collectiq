# KPI Dictionary

Implemented in `src/analytics/kpis.py`. "As-of" is the reporting date of the synthetic
book (30 Jun 2026). Money is INR (synthetic).

## Portfolio KPIs
| KPI | Definition | Numerator / denominator | Fields | Period | Interpretation | Limitations |
|---|---|---|---|---|---|---|
| Total outstanding exposure | Principal owed on open accounts | Σ outstanding principal, open accounts | loan_status, current_outstanding_principal | As-of | Size of the live book | Excludes charged-off balances (shown separately) |
| Delinquency rate (30+ DPD) | Share of open accounts 30+ DPD | open accounts 30+ DPD ÷ open accounts | loan_status / delinquency_bucket | Month-end | Severity of stress | Point-in-time; sensitive to book growth |
| Delinquency rate (1+ DPD) | Share of open accounts past due | open 1+ DPD ÷ open accounts | same | Month-end | Early stress | Includes short technical delays |
| Default rate | Share of loans charged off | charged-off loans ÷ loans originated | loan_status | Cumulative to as-of | Loss incidence | Younger cohorts have had less time to default |
| Portfolio recovery rate | Cash recovered on written-off principal | Σ post-charge-off recoveries ÷ Σ charged-off principal | final_recovery_amount, charged_off_principal | Cumulative to as-of | Recovery effectiveness | Recent charge-offs are still in recovery |
| Recovery amount | Cash recovered after charge-off | Σ final_recovery_amount | — | Cumulative / monthly | Recovery cash | — |
| Risk-segment concentration | Exposure share by segment | segment exposure ÷ total exposure | — | As-of | Concentration risk | — |
| Vintage performance | Cumulative share of a cohort ever 30+ DPD / charged off by month on book | loans reaching event by MOB m ÷ cohort size | snapshots | Comparable ages only | Cohort quality | Cells beyond the youngest loan's age are blank |
| Roll rate | Share of accounts moving bucket i → j next month | transitions i→j ÷ accounts in i | consecutive snapshots | Selected months | Delinquency dynamics | Only true consecutive month-ends |

## Collections KPIs (simulated events)
| KPI | Formula | Interpretation |
|---|---|---|
| Contact success rate | phone actions with right-party contact ÷ phone actions | Reachability |
| Promise-to-pay rate | promises ÷ right-party contacts | Negotiation outcome |
| Promise fulfilment rate | promises honoured ÷ promises (closed windows) | Promise quality |
| Recovery per collection action | Σ cash in 30-day windows ÷ actions | Productivity (association) |
| Average collection cost | Σ action cost ÷ actions | Unit cost |
| Net recovery value | Σ (window cash − cost) | Observed net cash (not incremental) |
| Actions per account | actions ÷ accounts worked | Contact intensity |
| Agent capacity utilisation | agent hours planned ÷ hours available | Resource use (optimisation) |

## Model KPIs
ROC-AUC, PR-AUC (average precision), precision, recall, F1 at the validation-chosen
threshold, Brier score, calibration curve, monthly ROC-AUC (stability), segment ROC-AUC
and calibration-in-the-large, PSI drift (train vs test). See MODEL_CARD.md.

## Optimisation KPIs
| KPI | Definition |
|---|---|
| Estimated (incremental) recovery value | Σ (expected cash with action − expected cash with no action) for selected pairs |
| Estimated net recovery value | estimated recovery value − Σ action cost |
| Budget utilisation | Σ cost ÷ budget |
| Capacity utilisation | Σ agent hours ÷ hours available |
| Selected accounts | distinct accounts with an action |
| Incremental value vs baseline | optimised net − baseline net, same constraints and eligibility |

All optimisation KPIs are model estimates, not observed outcomes.
