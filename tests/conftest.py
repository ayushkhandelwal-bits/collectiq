"""Shared fixtures: an isolated, small end-to-end pipeline build used by most tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def built(tmp_path_factory):
    """Run the full pipeline once with the 'test' profile in a temporary data root."""
    home = tmp_path_factory.mktemp("collectiq_home")
    os.environ["COLLECTIQ_HOME"] = str(home)
    os.environ["COLLECTIQ_PROFILE"] = "test"
    from src.database import db
    from src.pipeline.run_pipeline import run

    db.dispose_engines()
    summary = run(profile="test")
    assert summary["status"] == "SUCCESS", summary
    yield {"home": home, "summary": summary}
    db.dispose_engines()


@pytest.fixture(scope="session")
def small_generation():
    from src.data_generation.generate_data import generate

    return generate(n_loans=2000, seed=7)
