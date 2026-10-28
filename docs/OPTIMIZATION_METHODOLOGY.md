# Optimisation and Business Methodology

## Three different questions
| Quantity | Question | Source | Used for |
|---|---|---|---|
| **Roll-forward risk** | How likely is this delinquent account to reach 90+ DPD or charge-off in 3 months? | Model A | Severity, early warning, context in the rationale |
| **Expected cash** | How much cash is expected next month if we take action *a* — and if we do nothing? | Models B × C | Valuing actions |
| **Priority / net value** | Where does an action add the most value after its cost? | Expected cash (action − none) − cost | Ranking and optimisation |

High risk does not imply high value. A 60-89 DPD account with roll-forward risk 80%
may have little that any action can recover; a large 1-29 DPD account that will likely
cure on its own has high expected cash but little *incremental* value. CollectIQ ranks
by net value and shows risk alongside it.

## Valuation
For each eligible account *i* and permitted action *a*:

* `cash(i,a) = P(payment | i, a) × E[amount | payment, i, a]` (Models B and C; capped at
  what is collectible)
* `cash(i,∅)` — the same with action = "No action"
* `gross(i,a) = cash(i,a) − cash(i,∅)` — estimated incremental recovery
* `net(i,a) = gross(i,a) − cost(a)`

Action costs and agent hours (config/settings.yaml): Reminder ₹15 / 0 h, Outbound call
₹120 / 0.25 h, Follow-up call ₹100 / 0.2 h, Payment-plan review ₹450 / 0.75 h,
Specialist review ₹1,200 / 1.5 h. Permitted actions depend on stage.

## Eligibility (configurable demonstration rules — not legal guidance)
Excluded: no collectible balance; restricted states (*Dispute under review*, *Hardship
assessment hold*); contacted within the cooling-off period (7 days); contact cap reached
(3 in 30 days); charged-off accounts if excluded by policy. *Do-not-call request* allows
automated reminders only. Recommendations never include harassment, deception or contact
outside these rules.

## Formulation (MILP)
Decision `x(i,a) ∈ {0,1}` for eligible pairs.

```
maximise    Σ net(i,a) · x(i,a)
subject to  Σ cost(a) · x(i,a)        ≤ B          budget (INR)
            Σ hours(a) · x(i,a)       ≤ H          agent hours
            Σ_i x(i,a)                ≤ K_a        per action type
            Σ_a x(i,a)                ≤ M          per account (default 1)
            Σ_{i∈stage S} Σ_a x(i,a)  ≥ q_S        optional minimum coverage
```

Solved with HiGHS through `scipy.optimize.milp` (relative gap 0.05%, 60 s limit). Pairs with
non-positive net value are dropped unless a coverage constraint is active. Status is
reported as OPTIMAL, FEASIBLE_TIME_LIMIT, INFEASIBLE (with a diagnostic, e.g. "covering 25
accounts needs at least ₹X but the budget is ₹Y"), EMPTY or FAILED. If the solver is missing
or errors, an explicitly labelled greedy heuristic runs and is reported as HEURISTIC,
never as optimal.

## Baselines (identical eligibility and constraints)
1. **Balance-first** — highest outstanding balance first, stage default action.
2. **Risk-first** — highest roll-forward risk first (90+ DPD / charged off treated as
   maximum), stage default action.
3. **Oldest-delinquency-first** — highest DPD first, stage default action.
4. **ENRV ranking (greedy)** — best-action net value ranking filled greedily: the queue
   without optimisation.

Each baseline fills greedily in its order while budget, hours, action limits and the
per-account limit hold. The table reports accounts, cost, hours, gross, net and the
optimised plan's advantage. Results for the default run are in MODEL_RESULTS.md.

## Scenario analysis
Scenario Lab adjusts a copy of the candidate table: payment-probability and amount
multipliers apply to both the with-action and no-action cash (keeping the incremental
definition), delinquency volume and product composition are changed by resampling
eligible accounts as hypothetical replicas, and budget / hours are varied. Sensitivity
curves re-optimise at 25%–200% of budget and of capacity. A regression test checks that an
unchanged scenario reproduces the optimiser exactly.

## Interpretation
All values are model estimates on synthetic data. The optimiser shows how a plan would
change *if the estimates were right*; it is not evidence that realised recoveries would
improve. The incremental reading holds here only because the simulated history assigned
actions at random within stage.
