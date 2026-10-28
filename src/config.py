"""Configuration loading and project paths for CollectIQ."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"


@lru_cache(maxsize=1)
def load_settings() -> dict[str, Any]:
    """Load ``config/settings.yaml`` once per process."""
    with open(CONFIG_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def get_profile() -> str:
    """Return the active data profile (full, demo or test)."""
    return os.environ.get("COLLECTIQ_PROFILE", "full")


def path(key: str) -> Path:
    """Resolve a configured path relative to the project root.

    The database location can be overridden with ``COLLECTIQ_DB``; the whole data
    root can be redirected with ``COLLECTIQ_HOME`` (used by tests for isolation).
    """
    if key == "database" and os.environ.get("COLLECTIQ_DB"):
        return Path(os.environ["COLLECTIQ_DB"])
    base = Path(os.environ.get("COLLECTIQ_HOME", PROJECT_ROOT))
    return base / load_settings()["paths"][key]


def ensure_dirs() -> None:
    """Create every configured directory if it does not exist."""
    for key in [
        "raw_dir",
        "staging_dir",
        "processed_dir",
        "synthetic_dir",
        "metadata_dir",
        "artifacts_dir",
        "reports_dir",
        "figures_dir",
    ]:
        path(key).mkdir(parents=True, exist_ok=True)
    path("database").parent.mkdir(parents=True, exist_ok=True)
