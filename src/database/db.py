"""Database access layer: engine creation, schema initialisation, idempotent loads
and parameterised queries. All SQL in the project goes through this module."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import Engine, create_engine, event, inspect, text

from src.config import path
from src.database import schema
from src.utils.helpers import get_logger

log = get_logger(__name__)

# Delete order respects foreign keys (children first).
_LOAD_ORDER = ["loan_accounts", "repayment_snapshots", "collection_events",
               "recovery_outcomes", "portfolio_snapshots"]

_ENGINES: dict[str, Engine] = {}


def get_engine(db_path: str | Path | None = None) -> Engine:
    """Return a cached SQLite engine with foreign-key enforcement enabled."""
    db_file = Path(db_path) if db_path else path("database")
    key = str(db_file.resolve())
    if key not in _ENGINES:
        db_file.parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(f"sqlite:///{db_file}", future=True)

        @event.listens_for(engine, "connect")
        def _pragmas(dbapi_conn, _record):  # pragma: no cover - driver hook
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA synchronous=NORMAL")
            cur.close()

        _ENGINES[key] = engine
    return _ENGINES[key]


def dispose_engines() -> None:
    """Dispose all cached engines (used by tests)."""
    for eng in _ENGINES.values():
        eng.dispose()
    _ENGINES.clear()


def init_db(engine: Engine | None = None) -> None:
    """Create all tables, constraints and indexes if they do not exist (schema v1)."""
    engine = engine or get_engine()
    schema.metadata.create_all(engine)


def db_exists(engine: Engine | None = None) -> bool:
    """True if the database contains loaded loan records."""
    engine = engine or get_engine()
    if "loan_accounts" not in inspect(engine).get_table_names():
        return False
    with engine.connect() as conn:
        return bool(conn.execute(text("SELECT COUNT(*) FROM loan_accounts")).scalar())


def replace_core_tables(frames: dict[str, pd.DataFrame], engine: Engine | None = None,
                        chunksize: int = 20_000) -> dict[str, int]:
    """Idempotently replace core tables in one transaction.

    Existing rows are deleted (children first) and the new frames inserted, so
    re-running the pipeline never duplicates records. Either every table is
    replaced or - on any error - none is.
    """
    engine = engine or get_engine()
    init_db(engine)
    counts: dict[str, int] = {}
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM optimization_assignments"))
        for name in reversed(_LOAD_ORDER):
            conn.execute(text(f"DELETE FROM {name}"))
        for name in _LOAD_ORDER:
            if name not in frames:
                continue
            df = frames[name]
            cols = [c.name for c in schema.metadata.tables[name].columns]
            missing = [c for c in cols if c not in df.columns]
            if missing:
                raise ValueError(f"{name}: missing columns {missing}")
            df[cols].to_sql(name, conn, if_exists="append", index=False, chunksize=chunksize)
            counts[name] = len(df)
            log.info("Loaded %s rows into %s", f"{len(df):,}", name)
    return counts


def insert_rows(table: str, rows: list[dict[str, Any]] | pd.DataFrame,
                engine: Engine | None = None) -> None:
    """Append rows to a table inside a transaction."""
    engine = engine or get_engine()
    df = pd.DataFrame(rows) if not isinstance(rows, pd.DataFrame) else rows
    if df.empty:
        return
    with engine.begin() as conn:
        df.to_sql(table, conn, if_exists="append", index=False)


def execute(sql: str, params: dict[str, Any] | None = None,
            engine: Engine | None = None) -> None:
    """Execute a parameterised statement in a transaction."""
    engine = engine or get_engine()
    with engine.begin() as conn:
        conn.execute(text(sql), params or {})


def query(sql: str, params: dict[str, Any] | None = None,
          engine: Engine | None = None) -> pd.DataFrame:
    """Run a parameterised SELECT and return a DataFrame."""
    engine = engine or get_engine()
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})


def scalar(sql: str, params: dict[str, Any] | None = None,
           engine: Engine | None = None) -> Any:
    """Run a parameterised query returning a single value."""
    engine = engine or get_engine()
    with engine.connect() as conn:
        return conn.execute(text(sql), params or {}).scalar()


def table_counts(engine: Engine | None = None) -> pd.DataFrame:
    """Row counts for every schema table."""
    engine = engine or get_engine()
    rows = []
    existing = set(inspect(engine).get_table_names())
    for name in schema.metadata.tables:
        n = scalar(f"SELECT COUNT(*) FROM {name}", engine=engine) if name in existing else None
        rows.append({"table_name": name, "record_count": n})
    return pd.DataFrame(rows)


def data_dictionary() -> pd.DataFrame:
    """Column-level dictionary generated from the SQLAlchemy schema metadata."""
    rows = []
    for tname, table in schema.metadata.tables.items():
        for c in table.columns:
            fks = ", ".join(f"{fk.column.table.name}.{fk.column.name}" for fk in c.foreign_keys)
            rows.append({
                "table_name": tname,
                "column_name": c.name,
                "data_type": str(c.type),
                "nullable": bool(c.nullable),
                "primary_key": bool(c.primary_key),
                "foreign_key": fks or "",
                "definition": c.info.get("definition", ""),
                "allowed_values": c.info.get("allowed", ""),
                "provenance": c.info.get("provenance", ""),
            })
    return pd.DataFrame(rows)


def actual_schema(engine: Engine | None = None) -> pd.DataFrame:
    """Introspect the live database schema (used to verify the dictionary)."""
    engine = engine or get_engine()
    insp = inspect(engine)
    rows = []
    for tname in insp.get_table_names():
        pk = set(insp.get_pk_constraint(tname).get("constrained_columns", []))
        for c in insp.get_columns(tname):
            rows.append({"table_name": tname, "column_name": c["name"],
                         "data_type": str(c["type"]), "nullable": bool(c["nullable"]),
                         "primary_key": c["name"] in pk})
    return pd.DataFrame(rows)
