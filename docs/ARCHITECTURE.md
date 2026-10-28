# Architecture

```mermaid
flowchart LR
    subgraph Source
        K[Kaggle LendingClub<br/>probe only] -. unavailable .-> G
        G[Synthetic generator<br/>seed 42] --> R[(data/synthetic<br/>raw parquet, immutable)]
    end
    R --> S[Staging<br/>standardise types]
    S --> D[Derived tables<br/>recovery_outcomes,<br/>portfolio_snapshots]
    D --> Q{Data-quality gate<br/>42 checks}
    Q -- critical failure --> X[Run BLOCKED<br/>recorded in pipeline_runs]
    Q -- pass --> DB[(SQLite<br/>10 tables)]
    DB --> F[Features<br/>leakage-guarded]
    F --> A[Model A<br/>roll-forward risk]
    F --> B[Model B<br/>payment propensity]
    F --> C[Model C<br/>amount if paid]
    A & B & C --> REG[(Artifact registry<br/>SHA-256)]
    REG --> P[Prioritisation<br/>eligibility + net value]
    P --> O[MILP optimiser<br/>HiGHS + baselines]
    O --> DB
    DB --> UI[Streamlit app<br/>8 pages]
    P --> UI
    REG --> UI
```

## Layers and modules
| Layer | Module | Responsibility |
|---|---|---|
| Configuration | `config/settings.yaml`, `src/config.py` | Paths, profiles, costs, policy, hyper-parameters |
| Source | `src/ingestion/source_check.py` | Probe the public source; record the decision |
| Generation | `src/data_generation/generate_data.py` | Deterministic simulation |
| Staging / DQ | `src/preprocessing/` | Standardise, derive tables, 42 checks, source-to-target mapping |
| Storage | `src/database/` | Schema with constraints and indexes, transactional idempotent loads, parameterised queries |
| Features | `src/features/` | Point-in-time features, leakage guard |
| Models | `src/models/` | Training, evaluation, explanation, verified registry |
| Decisions | `src/prioritization/`, `src/optimization/` | Eligibility, valuation, MILP, baselines, persistence |
| Analytics | `src/analytics/`, `src/forecasting/` | KPIs, segments, vintages, roll rates, scenarios, forecasts |
| Orchestration | `src/pipeline/run_pipeline.py` | End-to-end run, manifest, generated docs |
| Presentation | `app.py`, `ui/` | Pages, shared components, cached data access |

UI code never trains models or writes SQL directly: it reads processed outputs through
`ui/data.py` (cached) and calls the same engine functions the pipeline uses.

## Idempotency and reproducibility
* Raw source folders are keyed by dataset, generator version, size and seed, and are
  never overwritten.
* Core tables are replaced in a single transaction (delete children → insert), so re-runs
  never duplicate rows; audit tables (`pipeline_runs`, `data_quality_results`,
  `model_runs`, `optimization_runs`) append.
* Seeds are fixed for generation, sampling and model training; package versions are
  recorded in `data/metadata/environment.json`.
