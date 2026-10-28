"""Generate documentation that must match executed results: the data dictionary
(from the live schema metadata) and model / optimisation results (from reports)."""

from __future__ import annotations

import json
from typing import Any

from src.config import PROJECT_ROOT, path
from src.database import db


def _md_escape(x: Any) -> str:
    return str(x).replace("|", "\\|").replace("\n", " ")


def data_dictionary_md() -> str:
    dd = db.data_dictionary()
    lines = [
        "# Data Dictionary", "",
        "Generated from the SQLAlchemy schema metadata in `src/database/schema.py` (the same "
        "metadata that creates the database), so it cannot drift from the implementation. The "
        "Data & Governance page verifies it against the live database.", "",
        "**Conventions.** Dates: ISO `YYYY-MM-DD`; timestamps `YYYY-MM-DDTHH:MM:SS` (UTC for "
        "audit tables). Money: synthetic Indian Rupees (INR) stored as REAL rupees. Ratios "
        "(DTI, utilisation, payment-to-income) are fractions (0.35 = 35%). `interest_rate` is "
        "percent per annum (14.5 = 14.5% p.a.). `data_origin` is PUBLIC or SYNTHETIC on every "
        "data row.", "",
    ]
    for t, g in dd.groupby("table_name", sort=False):
        lines += [f"## `{t}`", "", "| Column | Type | Null | PK | FK | Definition | Allowed values "
                  "| Provenance |", "|---|---|---|---|---|---|---|---|"]
        for r in g.itertuples(index=False):
            lines.append(f"| `{r.column_name}` | {r.data_type} | {'yes' if r.nullable else 'no'} "
                         f"| {'✓' if r.primary_key else ''} | {_md_escape(r.foreign_key)} | "
                         f"{_md_escape(r.definition)} | {_md_escape(r.allowed_values)} | "
                         f"{_md_escape(r.provenance)} |")
        lines.append("")
    return "\n".join(lines)


def _f(x: Any, d: int = 4) -> str:
    return "—" if x is None else f"{x:.{d}f}"


def model_results_md() -> str:
    rep = path("reports_dir")
    a = json.loads((rep / "risk_model_report.json").read_text())
    b = json.loads((rep / "propensity_model_report.json").read_text())
    c = json.loads((rep / "amount_model_report.json").read_text())
    s = json.loads((rep / "pipeline_summary.json").read_text())
    lines = ["# Model and Optimisation Results", "",
             f"Generated from the evaluation reports of pipeline run `{s['pipeline_run_id']}` "
             f"(profile `{s['profile']}`, dataset version `{s.get('dataset_version')}`). All "
             "numbers are computed on **synthetic** data on a chronologically held-out test "
             "period that was not used for model or threshold selection.", ""]
    for title, r in [("Model A — roll-forward risk", a), ("Model B — payment propensity", b)]:
        sd = r["split_dates"]
        lines += [f"## {title}", "",
                  f"Selected: **{r['selected']}**. Splits — train {r['split_sizes']['train']:,} "
                  f"({sd['train'][0]} to {sd['train'][1]}), validation "
                  f"{r['split_sizes']['valid']:,} ({sd['valid'][0]} to {sd['valid'][1]}), test "
                  f"{r['split_sizes']['test']:,} ({sd['test'][0]} to {sd['test'][1]}).", "",
                  "| Model | Valid ROC-AUC | Test ROC-AUC | Test PR-AUC | Precision | Recall | "
                  "F1 | Brier | Threshold |", "|---|---|---|---|---|---|---|---|---|"]
        for m, x in r["results"].items():
            t = x["test"]
            lines.append(f"| {m}{' (selected)' if m == r['selected'] else ''} | "
                         f"{_f(x['valid']['roc_auc'])} | {_f(t['roc_auc'])} | {_f(t['pr_auc'])} "
                         f"| {_f(t['precision'])} | {_f(t['recall'])} | {_f(t['f1'])} | "
                         f"{_f(t['brier'])} | {_f(x['threshold'], 3)} |")
        prev = r["results"][r["selected"]]["test"]["prevalence"]
        lines.append(f"\nTest prevalence: {prev:.1%}.\n")
    lines += ["## Model C — payment amount given payment (hurdle)", "",
              f"Selected: **{c['selected']}** (on validation MAE of expected cash).", "",
              "| Model | Test MAE given payment (₹) | Test RMSE given payment (₹) | "
              "Test R² (log) | Test MAE expected cash (₹) | Σ predicted / Σ actual |",
              "|---|---|---|---|---|---|"]
    for m, x in c["results"].items():
        ev = x["test_expected_value"]
        lines.append(f"| {m}{' (selected)' if m == c['selected'] else ''} | "
                     f"{x['test_positive']['mae']:,.0f} | {x['test_positive']['rmse']:,.0f} | "
                     f"{_f(x['test_positive_log']['r2'], 3)} | {ev['mae']:,.0f} | "
                     f"{ev['sum_predicted'] / ev['sum_actual']:.3f} |")
    o = s["optimization"]
    lines += ["", "## Default optimisation run", "",
              f"Status **{o['status']}** ({o['message']}). Budget utilisation "
              f"{o['budget_utilization']:.1%}, capacity utilisation "
              f"{o['capacity_utilization']:.1%}.", "",
              "| Strategy | Accounts | Est. incremental gross (₹) | Cost (₹) | Est. net (₹) | "
              "Optimised advantage (₹) |", "|---|---|---|---|---|---|",
              f"| Optimised plan (MILP) | {o['selected_accounts']:,} | {o['gross']:,.0f} | "
              f"{o['cost']:,.0f} | {o['net']:,.0f} | — |"]
    for name, bl in o["baselines"].items():
        lines.append(f"| {name} | {bl['selected_accounts']:,} | {bl['gross']:,.0f} | "
                     f"{bl['cost']:,.0f} | {bl['net']:,.0f} | "
                     f"{bl['incremental_net_vs_baseline']:,.0f} |")
    lines += ["", "These are model-based estimates on synthetic data. They do not show that the "
              "optimised plan would improve real-world recoveries.", ""]
    return "\n".join(lines)


def docs_dir():
    """Project docs folder, or <COLLECTIQ_HOME>/docs for isolated (test) runs."""
    import os
    from pathlib import Path

    base = Path(os.environ.get("COLLECTIQ_HOME", PROJECT_ROOT))
    d = base / "docs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_generated_docs() -> None:
    docs = docs_dir()
    (docs / "DATA_DICTIONARY.md").write_text(data_dictionary_md(), encoding="utf-8")
    (docs / "MODEL_RESULTS.md").write_text(model_results_md(), encoding="utf-8")
