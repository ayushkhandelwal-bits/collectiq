"""Generate the synthetic source only and print a summary.

Usage: python scripts/generate_data.py [--profile full|demo|test]
"""
import argparse
import json

import _bootstrap  # noqa: F401

from src.config import ensure_dirs, load_settings
from src.pipeline.run_pipeline import acquire

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="full", choices=["full", "demo", "test"])
    a = p.parse_args()
    ensure_dirs()
    s = load_settings()
    raw, params, d = acquire(s["profiles"][a.profile]["n_loans"], s["generation"]["seed"])
    print(json.dumps({k: len(v) for k, v in raw.items()}, indent=2), "->", d)
