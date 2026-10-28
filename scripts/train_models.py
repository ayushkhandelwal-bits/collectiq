"""Retrain models from the processed feature tables (run the pipeline first)."""
import _bootstrap  # noqa: F401
import pandas as pd

from src.config import path
from src.database import db
from src.models import train as tr

if __name__ == "__main__":
    proc = path("processed_dir")
    panel = pd.read_parquet(proc / "risk_model_panel.parquet")
    opp = pd.read_parquet(proc / "opportunity_features.parquet")
    version = db.scalar("SELECT source_checksum FROM pipeline_runs WHERE pipeline_status="
                        "'SUCCESS' ORDER BY started_at DESC LIMIT 1") or "manual"
    a = tr.train_risk_model(panel, version)
    b = tr.train_propensity_model(opp, version)
    c = tr.train_amount_model(opp, b["pipeline"], version)
    for name, out in [("risk_model", a), ("propensity_model", b), ("amount_model", c)]:
        tr.save_reports(name, out)
    db.insert_rows("model_runs", a["runs"] + b["runs"] + c["runs"])
    print("Selected:", a["selected"], "|", b["selected"], "|", c["selected"])
    print("Re-run the pipeline with --skip-train to re-score the queue with these models.")
