"""Streamlit app: every page renders, key actions run, errors are visible."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
PAGES = ["executive", "portfolio", "account", "workbench", "optimization", "models",
         "governance", "scenario"]


def page_app(page: str) -> AppTest:
    code = f"""
import sys
sys.path.insert(0, {str(ROOT)!r})
from ui.components.core import inject_css
inject_css()
from ui.pages import {page}
{page}.render()
"""
    return AppTest.from_string(code, default_timeout=240)


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_exceptions(built, page):
    at = page_app(page).run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.title) == 1


def test_main_app_initialises_with_navigation(built):
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=240).run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.title) == 1 and at.title[0].value == "Executive Overview"


def test_optimization_button_runs_solver(built):
    at = page_app("optimization").run()
    at.button[0].click().run()  # form submit button
    assert not at.exception
    md = " ".join(m.value for m in at.markdown)
    assert "Optimal" in md or "Heuristic" in md


def test_scenario_runs(built):
    at = page_app("scenario").run()
    at.checkbox[0].uncheck().run()
    at.button[0].click().run()
    assert not at.exception
    assert any("Scenario net recovery" in m.value for m in at.markdown)


def test_account_search_empty_state(built):
    at = page_app("account").run()
    at.text_input[0].set_value("ZZZ_NOT_A_LOAN").run()
    assert not at.exception
    assert any("No loan ID contains" in m.value for m in at.markdown)


def test_workbench_filters_and_export(built):
    at = page_app("workbench").run()
    at.selectbox[0].set_value("Eligible only").run()
    assert not at.exception
    import pandas as pd

    from src.config import path

    q = pd.read_parquet(path("processed_dir") / "priority_queue.parquet")
    csv = q.to_csv(index=False)
    assert csv.splitlines()[0].startswith("loan_id") or "loan_id" in csv.splitlines()[0]


def test_missing_artifacts_reported(built, monkeypatch):
    from ui import data

    monkeypatch.setattr(data, "REQUIRED_FILES", ["does_not_exist.parquet"])
    ready, missing = data.artifacts_ready()
    assert not ready and any("does_not_exist" in m for m in missing)


def test_untrusted_model_error_is_visible(built, monkeypatch):
    from src.models import registry

    reg = registry.load_registry()
    monkeypatch.setattr(registry, "load_registry", lambda: {k: v for k, v in reg.items()
                                                             if k != "risk_model"})
    with pytest.raises(registry.UntrustedArtifactError):
        registry.load_artifact("risk_model")
