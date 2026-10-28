"""Run the full CollectIQ pipeline. Usage: python scripts/run_pipeline.py [--profile demo]"""
import _bootstrap  # noqa: F401

from src.pipeline.run_pipeline import main

if __name__ == "__main__":
    raise SystemExit(main())
