"""Database schema, constraints, idempotent loading and the generated dictionary."""

from __future__ import annotations

import pytest
from sqlalchemy.exc import IntegrityError

from src.database import db, schema


def test_all_tables_and_columns_exist(built):
    live = db.actual_schema()
    dd = db.data_dictionary()
    assert set(schema.metadata.tables) <= set(live["table_name"])
    merged = dd.merge(live, on=["table_name", "column_name"], how="left", indicator=True)
    assert (merged["_merge"] == "both").all()
    pk = merged[merged["primary_key_x"]]
    assert pk["primary_key_y"].all()


def test_required_addendum_tables_present(built):
    for t in ["loan_accounts", "repayment_snapshots", "collection_events", "recovery_outcomes",
              "portfolio_snapshots", "data_quality_results", "model_runs", "optimization_runs",
              "optimization_assignments", "pipeline_runs"]:
        assert t in schema.metadata.tables


def test_core_tables_populated(built):
    counts = db.table_counts().set_index("table_name")["record_count"]
    for t in ["loan_accounts", "repayment_snapshots", "collection_events", "recovery_outcomes",
              "portfolio_snapshots", "data_quality_results", "model_runs", "optimization_runs"]:
        assert counts[t] > 0


def test_foreign_keys_enforced(built):
    with pytest.raises(IntegrityError):
        db.execute("INSERT INTO collection_events (event_id, loan_id, event_date, data_origin) "
                   "VALUES ('EVX', 'NO_SUCH_LOAN', '2026-01-01', 'SYNTHETIC')")


def test_unique_snapshot_constraint(built):
    row = db.query("SELECT * FROM repayment_snapshots LIMIT 1").iloc[0]
    with pytest.raises(IntegrityError):
        db.execute("INSERT INTO repayment_snapshots (snapshot_id, loan_id, snapshot_date, "
                   "days_past_due, payment_received, outstanding_principal, data_origin) "
                   "VALUES ('NEWID', :l, :d, 0, 0, 0, 'SYNTHETIC')",
                   {"l": row["loan_id"], "d": row["snapshot_date"]})


def test_check_constraint_blocks_negative_amounts(built):
    with pytest.raises(IntegrityError):
        db.execute("UPDATE loan_accounts SET loan_amount = -1 WHERE loan_id = "
                   "(SELECT loan_id FROM loan_accounts LIMIT 1)")


def test_parameterised_query_safe(built):
    out = db.query("SELECT COUNT(*) AS n FROM loan_accounts WHERE loan_id = :x",
                   {"x": "x' OR '1'='1"})
    assert out["n"].iloc[0] == 0


def test_reload_is_idempotent(built):
    import pandas as pd

    from src.config import path

    before = db.table_counts().set_index("table_name")["record_count"]
    frames = {n: pd.read_parquet(path("staging_dir") / f"{n}.parquet")
              for n in schema.CORE_TABLES}
    db.replace_core_tables(frames)
    db.replace_core_tables(frames)
    after = db.table_counts().set_index("table_name")["record_count"]
    for t in schema.CORE_TABLES:
        assert before[t] == after[t]


def test_public_and_synthetic_distinguishable(built):
    origins = db.query("SELECT DISTINCT data_origin FROM loan_accounts")["data_origin"]
    assert set(origins) <= {"PUBLIC", "SYNTHETIC"}
