"""CollectIQ - Collections Intelligence & Recovery Optimization (Streamlit entry point)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from src.config import load_settings  # noqa: E402
from ui import data, filters  # noqa: E402
from ui.components.core import fmt_date, footer, inject_css  # noqa: E402

SETTINGS = load_settings()
APP = SETTINGS["app"]

st.set_page_config(page_title="CollectIQ", page_icon=str(ROOT / "ui/styles/logo.svg"),
                   layout="wide", initial_sidebar_state="expanded")
inject_css()
st.logo(str(ROOT / "ui/styles/logo.svg"), size="large")


def bootstrap() -> bool:
    """Build data and models on first start (e.g. a fresh cloud deployment)."""
    ready, missing = data.artifacts_ready()
    if ready:
        return True
    st.title("CollectIQ")
    st.markdown("The analytical database and models have not been built in this environment "
                "yet. Building them runs the full pipeline: synthetic data generation, "
                "quality checks, database load, model training, prioritisation and a default "
                "optimisation run.")
    st.caption("Missing: " + "; ".join(missing))
    profile = os.environ.get("COLLECTIQ_PROFILE", "demo")
    n = SETTINGS["profiles"][profile]["n_loans"]
    if st.button(f"Build data and models ({n:,} synthetic loans)", type="primary"):
        from src.pipeline.run_pipeline import run

        with st.spinner("Running pipeline — typically under two minutes…"):
            try:
                summary = run(profile=profile)
            except Exception as exc:  # surfaced to the user, also logged
                st.error(f"Pipeline failed: {type(exc).__name__}: {exc}")
                return False
        if summary.get("status") != "SUCCESS":
            st.error(f"Pipeline ended with status {summary.get('status')}: "
                     f"{summary.get('error', '')}")
            return False
        data.clear_caches()
        st.rerun()
    return False


def sidebar_status() -> None:
    reg = data.registry()
    m = data.manifest()
    st.sidebar.markdown(f"<div style='color:#51607A;font-size:0.82rem;margin-top:-0.4rem'>"
                        f"{APP['tagline']}</div>", unsafe_allow_html=True)
    filters.sidebar_filters()
    st.sidebar.divider()
    mode = m.get("data_mode", "SYNTHETIC").title()
    st.sidebar.markdown(f"**Data mode:** {mode}  \n"
                        f"**As-of date:** {fmt_date(SETTINGS['generation']['as_of_date'])}  \n"
                        f"**Dataset:** `{m.get('dataset_version', 'n/a')}`")
    trained = reg.get("risk_model", {}).get("trained_at")
    n_models = sum(k in reg for k in ("risk_model", "propensity_model", "amount_model"))
    st.sidebar.markdown(f"**Models available:** {n_models} of 3  \n"
                        f"**Last trained:** {fmt_date(trained) if trained else 'n/a'}")
    runs = data.table("pipeline_runs", "started_at", 1)
    if not runs.empty:
        r = runs.iloc[0]
        st.sidebar.markdown(f"**Last pipeline run:** {r['pipeline_status'].title()} "
                            f"({fmt_date(r['started_at'])})")
    if st.sidebar.button("Reload data", help="Clear cached data and models and re-read them"):
        data.clear_caches()
        st.rerun()
    st.sidebar.caption(f"Version {APP['version']}")


if bootstrap():
    from ui.pages import (
        account,
        executive,
        governance,
        models,
        optimization,
        portfolio,
        scenario,
        workbench,
    )

    pages = [
        st.Page(executive.render, title="Executive Overview", url_path="overview", default=True),
        st.Page(portfolio.render, title="Portfolio Explorer", url_path="portfolio"),
        st.Page(account.render, title="Account Intelligence", url_path="account"),
        st.Page(workbench.render, title="Collections Workbench", url_path="workbench"),
        st.Page(optimization.render, title="Recovery Optimization", url_path="optimization"),
        st.Page(models.render, title="Model & Monitoring", url_path="models"),
        st.Page(governance.render, title="Data & Governance", url_path="governance"),
        st.Page(scenario.render, title="Scenario Lab", url_path="scenario"),
    ]
    nav = st.navigation(pages)
    sidebar_status()
    try:
        nav.run()
    except Exception as exc:  # visible, understandable error instead of a blank page
        st.error(f"This page could not be rendered: {type(exc).__name__}: {exc}")
        st.caption("Try 'Reload data' in the sidebar. If the problem persists, re-run the "
                   "pipeline from Data & Governance.")
    m = data.manifest()
    footer(APP["version"], f"Data: {m.get('dataset_name', 'n/a')} "
           f"({m.get('data_mode', 'SYNTHETIC').lower()}), version "
           f"{m.get('dataset_version', 'n/a')}. Currency: INR (synthetic).")
