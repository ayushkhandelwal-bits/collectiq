"""Cached data access for the Streamlit app. All SQL is parameterised via src.database."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st

from src.config import load_settings, path
from src.database import db
from src.models.registry import load_artifact, load_registry
from src.prioritization.engine import Policy, apply_policy

REQUIRED_FILES = ["priority_queue.parquet", "collections_population.parquet",
                  "action_candidates.parquet", "monthly_trends.parquet"]


def artifacts_ready() -> tuple[bool, list[str]]:
    """Whether the database, processed outputs and model artifacts exist."""
    missing = []
    try:
        if not db.db_exists():
            missing.append("database (data/processed/collectiq.db)")
    except Exception as exc:  # corrupt or locked database
        missing.append(f"database error: {exc}")
    for f in REQUIRED_FILES:
        if not (path("processed_dir") / f).exists():
            missing.append(f"data/processed/{f}")
    reg = load_registry()
    for m in ("risk_model", "propensity_model", "amount_model"):
        if m not in reg:
            missing.append(f"model artifact '{m}'")
    return (not missing), missing


@st.cache_data(show_spinner=False)
def parquet(name: str) -> pd.DataFrame:
    p = path("processed_dir") / name
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def report(name: str) -> dict[str, Any]:
    p = path("reports_dir") / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


@st.cache_data(show_spinner=False)
def manifest() -> dict[str, Any]:
    p = path("metadata_dir") / "source_manifest.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


@st.cache_data(show_spinner=False)
def loans() -> pd.DataFrame:
    return db.query("SELECT * FROM loan_accounts")


@st.cache_data(show_spinner=False)
def portfolio_snapshots() -> pd.DataFrame:
    return db.query("SELECT * FROM portfolio_snapshots")


@st.cache_data(show_spinner=False)
def table(name: str, order_by: str | None = None, limit: int = 500) -> pd.DataFrame:
    allowed = {"pipeline_runs", "data_quality_results", "model_runs", "optimization_runs"}
    if name not in allowed:
        raise ValueError("table not allowed")
    order = f" ORDER BY {order_by} DESC" if order_by else ""
    return db.query(f"SELECT * FROM {name}{order} LIMIT :n", {"n": limit})


def latest_quality() -> pd.DataFrame:
    return db.query(
        "SELECT * FROM data_quality_results WHERE pipeline_run_id = ("
        "SELECT pipeline_run_id FROM pipeline_runs WHERE pipeline_status IN "
        "('SUCCESS','BLOCKED') ORDER BY started_at DESC LIMIT 1)")


def search_loans(term: str, limit: int = 50) -> pd.DataFrame:
    return db.query("SELECT loan_id, loan_product, risk_grade, loan_status, "
                    "current_outstanding_principal FROM loan_accounts WHERE loan_id LIKE :t "
                    "ORDER BY loan_id LIMIT :n", {"t": f"%{term.strip().upper()}%", "n": limit})


def account_history(loan_id: str) -> pd.DataFrame:
    return db.query("SELECT * FROM repayment_snapshots WHERE loan_id = :l ORDER BY snapshot_date",
                    {"l": loan_id})


def account_events(loan_id: str) -> pd.DataFrame:
    return db.query("SELECT * FROM collection_events WHERE loan_id = :l ORDER BY event_date",
                    {"l": loan_id})


def optimization_runs(limit: int = 20) -> pd.DataFrame:
    return db.query("SELECT * FROM optimization_runs ORDER BY run_timestamp DESC LIMIT :n",
                    {"n": limit})


@st.cache_resource(show_spinner=False)
def model(name: str):
    return load_artifact(name)


@st.cache_data(show_spinner=False)
def registry() -> dict[str, Any]:
    return load_registry()


@st.cache_data(show_spinner="Applying contact policy…")
def queue_for_policy(cooling_off_days: int, max_contacts_30d: int, include_charged_off: bool,
                     min_days_past_due: int, excluded_actions: tuple[str, ...]) -> pd.DataFrame:
    """Re-derive the prioritisation queue for a policy without re-scoring models."""
    pop = parquet("collections_population.parquet")
    cand = parquet("action_candidates.parquet")
    base = Policy.from_settings()
    pol = Policy(cooling_off_days=cooling_off_days, max_contacts_30d=max_contacts_30d,
                 include_charged_off=include_charged_off, min_days_past_due=min_days_past_due,
                 restricted_states=base.restricted_states,
                 digital_only_states=base.digital_only_states,
                 excluded_actions=list(excluded_actions))
    reg = registry()
    mv = " | ".join(reg[k]["model_version"] for k in ("risk_model", "propensity_model",
                                                       "amount_model") if k in reg)
    q = apply_policy(pop, cand, pol, model_version=mv,
                     data_timestamp=load_settings()["generation"]["as_of_date"])
    money = ["outstanding_principal", "amount_due", "amount_if_paid", "expected_cash_with_action",
             "expected_cash_no_action", "expected_gross_recovery", "action_cost",
             "expected_net_recovery", "priority_score"]
    for c in money:
        if c in q.columns:
            q[c] = q[c].round(0)
    return q


def default_queue() -> pd.DataFrame:
    c = load_settings()["collections"]
    return queue_for_policy(c["cooling_off_days"], c["max_contacts_30d"], True, 1, ())


def clear_caches() -> None:
    st.cache_data.clear()
    st.cache_resource.clear()
