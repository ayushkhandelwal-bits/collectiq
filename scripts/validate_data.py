"""Run the data-quality checks on the staged tables and print failures."""
import _bootstrap  # noqa: F401
import pandas as pd

from src.config import load_settings, path
from src.database.schema import CORE_TABLES
from src.preprocessing.quality import run_quality_checks

if __name__ == "__main__":
    frames = {n: pd.read_parquet(path("staging_dir") / f"{n}.parquet") for n in CORE_TABLES}
    rep = run_quality_checks(frames, load_settings()["generation"]["as_of_date"])
    df = rep.frame
    print(df["check_status"].value_counts().to_string())
    bad = df[df["check_status"].isin(["FAIL", "WARN"])]
    print("No failures or warnings." if bad.empty else bad.to_string())
    raise SystemExit(0 if rep.passed else 1)
