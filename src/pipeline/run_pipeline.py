"""End-to-end CollectIQ pipeline.

raw source -> staging -> data-quality gate -> SQLite load -> features -> models ->
scoring & prioritisation -> optimisation -> analytics / reports -> metadata.

Usage::

    python -m src.pipeline.run_pipeline            # full profile (50k loans)
    COLLECTIQ_PROFILE=demo python -m src.pipeline.run_pipeline
    python -m src.pipeline.run_pipeline --skip-train   # reuse registered models
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
import uuid
from datetime import timedelta
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import pandas as pd

from src.analytics import kpis
from src.config import ensure_dirs, get_profile, load_settings, path
from src.data_generation.generate_data import generate, summarise
from src.database import db
from src.features.build_features import (
    PROHIBITED_FEATURES,
    RECOVERY_CATEGORICAL,
    RECOVERY_NUMERIC,
    RISK_CATEGORICAL,
    RISK_NUMERIC,
    latest_account_state,
    opportunity_features,
    risk_panel,
    snapshot_features,
)
from src.forecasting.forecast import forecast_series
from src.ingestion.source_check import probe_primary_source
from src.models import train as tr
from src.models.registry import load_artifact, load_registry
from src.optimization.service import default_constraints, optimize_and_compare, persist_run
from src.preprocessing.mapping import mapping_frame, mapping_markdown
from src.preprocessing.quality import leakage_check, run_quality_checks
from src.preprocessing.standardize import (
    build_portfolio_snapshots,
    build_recovery_outcomes,
    standardize,
)
from src.prioritization.engine import Policy, apply_policy, score_population
from src.utils.docgen import docs_dir, write_generated_docs
from src.utils.helpers import file_sha256, frame_fingerprint, get_logger, utc_now, write_json

log = get_logger("collectiq.pipeline")

PACKAGES = ["pandas", "numpy", "scipy", "scikit-learn", "xgboost", "shap", "statsmodels",
            "sqlalchemy", "streamlit", "plotly", "pyarrow"]


class PipelineBlocked(RuntimeError):
    """Raised when critical data-quality failures block downstream steps."""


def _package_versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for p in PACKAGES:
        try:
            out[p] = importlib_metadata.version(p)
        except importlib_metadata.PackageNotFoundError:
            out[p] = "not installed"
    return out


def _raw_dir(params_key: str) -> Path:
    return path("synthetic_dir") / params_key


def acquire(n_loans: int, seed: int) -> tuple[dict[str, pd.DataFrame], dict[str, Any], Path]:
    """Load the raw synthetic source, generating it only if absent (never overwritten)."""
    gcfg = load_settings()["generation"]
    key = f"{gcfg['dataset_name']}_{gcfg['version']}_n{n_loans}_seed{seed}"
    d = _raw_dir(key)
    files = {n: d / f"{n}.parquet" for n in ("loans", "snapshots", "events")}
    if all(f.exists() for f in files.values()) and (d / "generation_parameters.json").exists():
        log.info("Reusing existing raw synthetic source at %s", d)
        from src.utils.helpers import read_json

        params = read_json(d / "generation_parameters.json")
        raw = {n: pd.read_parquet(f) for n, f in files.items()}
    else:
        log.info("Generating synthetic source (%s loans, seed %s)", f"{n_loans:,}", seed)
        res = generate(n_loans=n_loans, seed=seed)
        d.mkdir(parents=True, exist_ok=True)
        res.loans.to_parquet(files["loans"], index=False)
        res.snapshots.to_parquet(files["snapshots"], index=False)
        res.events.to_parquet(files["events"], index=False)
        params = res.parameters
        write_json(params, d / "generation_parameters.json")
        write_json(summarise(res), d / "dataset_summary.json")
        raw = {"loans": res.loans, "snapshots": res.snapshots, "events": res.events}
    return raw, params, d


def run(profile: str | None = None, skip_train: bool = False, seed: int | None = None
        ) -> dict[str, Any]:
    """Execute the full pipeline and return a summary dict."""
    t_start = time.time()
    ensure_dirs()
    settings = load_settings()
    profile = profile or get_profile()
    n_loans = settings["profiles"][profile]["n_loans"]
    seed = settings["generation"]["seed"] if seed is None else seed
    as_of = settings["generation"]["as_of_date"]
    planning_start = (pd.Timestamp(as_of) + timedelta(days=1)).strftime("%Y-%m-%d")
    planning_end = (pd.Timestamp(planning_start)
                    + timedelta(days=settings["collections"]["planning_period_days"] - 1)
                    ).strftime("%Y-%m-%d")
    engine = db.get_engine()
    db.init_db(engine)
    run_id = f"PR-{uuid.uuid4().hex[:12]}"
    db.insert_rows("pipeline_runs", [{
        "pipeline_run_id": run_id, "started_at": utc_now(), "completed_at": None,
        "pipeline_status": "RUNNING", "source_dataset": settings["generation"]["dataset_name"],
        "source_checksum": None, "input_record_count": None, "output_record_count": None,
        "error_summary": None}])
    summary: dict[str, Any] = {"pipeline_run_id": run_id, "profile": profile}
    try:
        # 1-3. Source probe and acquisition --------------------------------- #
        source = probe_primary_source()
        summary["source_probe"] = source.to_dict()
        raw, params, raw_dir = acquire(n_loans, seed)
        checksums = {f.name: file_sha256(f) for f in sorted(raw_dir.glob("*.parquet"))}
        combined_checksum = file_sha256_combined(checksums)

        # 4-5. Staging and derived tables ------------------------------------ #
        frames = standardize(raw["loans"], raw["snapshots"], raw["events"])
        frames["recovery_outcomes"] = build_recovery_outcomes(
            frames["collection_events"], frames["repayment_snapshots"], frames["loan_accounts"])
        frames["portfolio_snapshots"] = build_portfolio_snapshots(
            frames["repayment_snapshots"], frames["loan_accounts"])
        stg = path("staging_dir")
        for name, df in frames.items():
            df.to_parquet(stg / f"{name}.parquet", index=False)

        # 6. Data-quality gate ------------------------------------------------ #
        report = run_quality_checks(frames, as_of)
        lk = leakage_check({"risk_model": RISK_NUMERIC + RISK_CATEGORICAL,
                            "recovery_models": RECOVERY_NUMERIC + RECOVERY_CATEGORICAL},
                           PROHIBITED_FEATURES)
        dq = pd.concat([report.frame, lk.frame], ignore_index=True)
        dq.insert(0, "pipeline_run_id", run_id)
        dq.insert(0, "quality_result_id", [f"{run_id}-Q{i:03d}" for i in range(len(dq))])
        dq["executed_at"] = utc_now()
        db.insert_rows("data_quality_results", dq)
        dq.to_csv(path("reports_dir") / "data_quality_report.csv", index=False)
        crit = dq[(dq["check_status"] == "FAIL") & (dq["severity"] == "CRITICAL")]
        summary["quality"] = {"checks": len(dq), "failed_critical": len(crit),
                              "warnings": int((dq["check_status"] == "WARN").sum())}
        if not crit.empty:
            raise PipelineBlocked("Critical data-quality failures: "
                                  + "; ".join(crit["table_name"] + "." + crit["check_name"]))

        # 7. Load database (single transaction, idempotent) ------------------ #
        counts = db.replace_core_tables(frames, engine)
        summary["loaded"] = counts
        dataset_version = frame_fingerprint(frames["loan_accounts"][["loan_id", "loan_status"]])
        summary["dataset_version"] = dataset_version

        # 8. Features ---------------------------------------------------------- #
        loans = frames["loan_accounts"]
        events = frames["collection_events"]
        sf = snapshot_features(frames["repayment_snapshots"], loans)
        rc = settings["modeling"]["risk_model"]
        panel = risk_panel(sf, rc["horizon_months"], rc["test_end"],
                           rc["snapshot_sample_fraction"], settings["modeling"]["random_seed"])
        evf = opportunity_features(events, sf)
        proc = path("processed_dir")
        panel.to_parquet(proc / "risk_model_panel.parquet", index=False)
        evf.to_parquet(proc / "opportunity_features.parquet", index=False)
        state = latest_account_state(sf, events, as_of, planning_start)
        state.to_parquet(proc / "account_state.parquet", index=False)

        # 9. Models ------------------------------------------------------------ #
        have_models = all(k in load_registry() for k in
                          ("risk_model", "propensity_model", "amount_model"))
        if skip_train and have_models:
            log.info("Skipping training; using registered artifacts")
        else:
            a = tr.train_risk_model(panel, dataset_version)
            b = tr.train_propensity_model(evf, dataset_version)
            c = tr.train_amount_model(evf, b["pipeline"], dataset_version)
            tr.save_reports("risk_model", a)
            tr.save_reports("propensity_model", b)
            tr.save_reports("amount_model", c)
            db.insert_rows("model_runs", a["runs"] + b["runs"] + c["runs"])
            summary["models"] = {
                "risk": {"selected": a["selected"],
                         "test": {m: r["test"] for m, r in a["results"].items()}},
                "propensity": {"selected": b["selected"],
                               "test": {m: r["test"] for m, r in b["results"].items()}},
                "amount": {"selected": c["selected"],
                           "test": {m: {"positive": r["test_positive"],
                                        "expected_value": r["test_expected_value"]}
                                    for m, r in c["results"].items()}},
            }

        # 10. Scoring and prioritisation -------------------------------------- #
        reg = load_registry()
        pop, cand = score_population(state, loans, load_artifact("risk_model"),
                                     reg["risk_model"], load_artifact("propensity_model"),
                                     load_artifact("amount_model"), reg["amount_model"])
        mv = " | ".join(reg[k]["model_version"] for k in
                        ("risk_model", "propensity_model", "amount_model"))
        queue = apply_policy(pop, cand, Policy.from_settings(), model_version=mv,
                             data_timestamp=as_of)
        pop.to_parquet(proc / "collections_population.parquet", index=False)
        cand.to_parquet(proc / "action_candidates.parquet", index=False)
        queue.drop(columns=["permitted_actions"]).to_parquet(proc / "priority_queue.parquet",
                                                             index=False)
        summary["prioritisation"] = {
            "population": len(pop), "candidates": len(cand),
            "eligible": int(queue["eligible"].sum()),
            "positive_enrv_accounts": int(queue["priority_score"].notna().sum()),
            "priority_counts": queue["priority_category"].value_counts().to_dict()}

        # 11. Default optimisation run ------------------------------------------ #
        from src.prioritization.engine import eligible_candidates

        ec = eligible_candidates(pop, cand, Policy.from_settings())
        cons = default_constraints()
        res, baselines = optimize_and_compare(ec, queue, cons)
        opt_id = persist_run(res, baselines, cons, queue, planning_start, planning_end,
                             {"source": "pipeline default run"})
        summary["optimization"] = {"run_id": opt_id, "status": res.status,
                                   "message": res.message, **res.totals(cons),
                                   "baselines": baselines}

        # 12. Analytics, forecasting --------------------------------------------- #
        trends = kpis.monthly_trends(frames["portfolio_snapshots"])
        trends.to_parquet(proc / "monthly_trends.parquet", index=False)
        vb = kpis.vintage_base(frames["repayment_snapshots"], loans, as_of)
        vb.to_parquet(proc / "vintage_base.parquet", index=False)
        rr = kpis.roll_rate_counts(frames["repayment_snapshots"], loans)
        rr.to_parquet(proc / "roll_rate_counts.parquet", index=False)
        kpis.action_performance(events).to_parquet(proc / "action_performance.parquet",
                                                   index=False)
        series = trends.set_index("snapshot_date")["delinquency_rate_30plus"]
        fc = forecast_series(series)
        write_json(fc, path("reports_dir") / "delinquency_forecast.json")
        write_json({"portfolio": kpis.portfolio_kpis(loans),
                    "collections_all": kpis.collections_kpis(events),
                    "collections_last_3m": kpis.collections_kpis(
                        events, (pd.Timestamp(as_of) - pd.DateOffset(months=3)
                                 + timedelta(days=1)).strftime("%Y-%m-%d"))},
                   path("reports_dir") / "kpi_summary.json")

        # 13. Metadata / manifest ----------------------------------------------- #
        mp = mapping_frame(frames)
        mp.to_csv(path("metadata_dir") / "source_to_target_mapping.csv", index=False)
        (docs_dir() / "SOURCE_TO_TARGET_MAPPING.md").write_text(
            mapping_markdown(mp), encoding="utf-8")
        manifest = build_manifest(source.to_dict(), params, raw_dir, checksums, frames,
                                  combined_checksum, dataset_version, profile)
        write_json(manifest, path("metadata_dir") / "source_manifest.json")
        write_json(params, path("metadata_dir") / "generation_parameters.json")
        write_json(_package_versions(), path("metadata_dir") / "environment.json")
        (docs_dir() / "DATA_SOURCES.md").write_text(
            data_sources_markdown(manifest), encoding="utf-8")

        total_in = sum(len(v) for v in raw.values())
        total_out = sum(counts.values())
        db.execute("UPDATE pipeline_runs SET completed_at=:c, pipeline_status='SUCCESS', "
                   "source_checksum=:s, input_record_count=:i, output_record_count=:o "
                   "WHERE pipeline_run_id=:r",
                   {"c": utc_now(), "s": combined_checksum, "i": total_in, "o": total_out,
                    "r": run_id})
        summary["status"] = "SUCCESS"
    except PipelineBlocked as exc:
        db.execute("UPDATE pipeline_runs SET completed_at=:c, pipeline_status='BLOCKED', "
                   "error_summary=:e WHERE pipeline_run_id=:r",
                   {"c": utc_now(), "e": str(exc)[:2000], "r": run_id})
        summary["status"] = "BLOCKED"
        summary["error"] = str(exc)
        log.error("Pipeline blocked: %s", exc)
    except Exception as exc:
        db.execute("UPDATE pipeline_runs SET completed_at=:c, pipeline_status='FAILED', "
                   "error_summary=:e WHERE pipeline_run_id=:r",
                   {"c": utc_now(), "e": f"{type(exc).__name__}: {exc}"[:2000], "r": run_id})
        raise
    summary["runtime_seconds"] = round(time.time() - t_start, 1)
    write_json(summary, path("reports_dir") / "pipeline_summary.json")
    if summary.get("status") == "SUCCESS" and "models" in summary:
        write_generated_docs()
    log.info("Pipeline %s in %.1fs", summary["status"], summary["runtime_seconds"])
    return summary


def file_sha256_combined(checksums: dict[str, str]) -> str:
    """Checksum over the per-file checksums (order-independent)."""
    import hashlib

    h = hashlib.sha256()
    for k in sorted(checksums):
        h.update(f"{k}:{checksums[k]}".encode())
    return h.hexdigest()


def build_manifest(source: dict, params: dict, raw_dir: Path, checksums: dict[str, str],
                   frames: dict[str, pd.DataFrame], combined: str, dataset_version: str,
                   profile: str) -> dict[str, Any]:
    """Machine-readable dataset manifest."""
    rel = raw_dir.relative_to(path("synthetic_dir").parent.parent)
    return {
        "dataset_name": params["dataset_name"],
        "source_type": "synthetic",
        "data_mode": "SYNTHETIC",
        "primary_source": {"name": source["source_name"], "url": source["source_url"],
                           "probe": source,
                           "used": False,
                           "reason_not_used": source["decision"]},
        "acquisition_method": "Deterministic simulation (src/data_generation/generate_data.py)",
        "retrieval_or_generation_date": utc_now(),
        "file_location": str(rel),
        "files": [{"file": k, "sha256": v,
                   "size_bytes": (raw_dir / k).stat().st_size} for k, v in checksums.items()],
        "combined_checksum": combined,
        "dataset_version": dataset_version,
        "profile": profile,
        "record_counts": {k: int(len(v)) for k, v in frames.items()},
        "column_counts": {k: int(v.shape[1]) for k, v in frames.items()},
        "license_or_terms": "Synthetic data generated for this project; free to use and "
                            "redistribute. No real customer data.",
        "generation": {"seed": params["seed"], "version": params["generation_version"],
                       "n_loans": params["n_loans"], "as_of_date": params["as_of_date"],
                       "method": "Month-by-month simulation with latent risk index, "
                                 "logistic hazards, stage-conditional historical collection "
                                 "policy and capped recoveries. Parameters in "
                                 "generation_parameters.json."},
        "currency": "INR (synthetic)",
        "transformation_history": [
            "generate -> data/synthetic/<dataset>/ (raw, immutable)",
            "standardize names/types -> data/staging/",
            "derive recovery_outcomes (event windows) and portfolio_snapshots (aggregates)",
            "data-quality gate (critical failures block loading and training)",
            "load SQLite in one transaction (delete + insert, idempotent)",
            "feature engineering -> data/processed/",
        ],
        "known_limitations": [
            "All records are synthetic; no PUBLIC observations are present.",
            "Collection-event outcomes are simulated and are not evidence of real-world "
            "collection effectiveness.",
            "Historical collection actions were assigned at random within delinquency stage; "
            "real-world action assignment is usually non-random (confounded).",
            "Payment timing within a month is not simulated; each month has at most one "
            "collection event per account and the month's cash is attributed to it.",
            "Macro stress and seasonality are stylised assumptions.",
        ],
    }


def data_sources_markdown(m: dict[str, Any]) -> str:
    """Human-readable DATA_SOURCES.md generated from the manifest."""
    rc = m["record_counts"]
    probe = m["primary_source"]["probe"]
    lines = [
        "# Data Sources", "",
        "> **Data mode: SYNTHETIC.** Every record analysed by CollectIQ is synthetic. "
        "No public or real customer observations are included.", "",
        "## 1. Preferred public source (not used)", "",
        f"* Source: {m['primary_source']['name']}",
        f"* URL: {m['primary_source']['url']}",
        f"* Probe time (UTC): {probe['checked_at']}",
        f"* Network reachable: {probe['network_reachable']} ({probe['network_detail']})",
        f"* Authorised Kaggle credentials present: {probe['credentials_available']}",
        f"* Decision: `{probe['decision']}`", "",
        "Kaggle requires an authenticated account to download competition data. No "
        "authorised route existed in the build environment, and the project does not "
        "request, store or hard-code tokens, or scrape the site. The prescribed synthetic "
        "fallback was therefore used for **all** tables. To add the public data later, "
        "place the downloaded files in `data/raw/lendingclub/` and record them in the "
        "manifest with `data_origin = 'PUBLIC'`; the schema keeps public and synthetic rows "
        "distinguishable.", "",
        "## 2. Synthetic dataset", "",
        f"* Name: `{m['dataset_name']}`  |  version `{m['generation']['version']}`  |  "
        f"seed {m['generation']['seed']}  |  profile `{m['profile']}`",
        f"* As-of (reporting) date: {m['generation']['as_of_date']}",
        f"* Location: `{m['file_location']}` (raw parquet, never overwritten)",
        f"* Combined checksum: `{m['combined_checksum']}`",
        f"* Dataset version (fingerprint): `{m['dataset_version']}`",
        f"* Currency: {m['currency']}", "",
        "| Table | Records | Columns |", "|---|---:|---:|",
    ]
    for k, v in rc.items():
        lines.append(f"| {k} | {v:,} | {m['column_counts'][k]} |")
    lines += ["", "### Generation method", "", m["generation"]["method"], "",
              "### Files", "", "| File | Size (bytes) | SHA-256 |", "|---|---:|---|"]
    for f in m["files"]:
        lines.append(f"| {f['file']} | {f['size_bytes']:,} | `{f['sha256'][:16]}…` |")
    lines += ["", "### Transformation history", ""]
    lines += [f"{i + 1}. {t}" for i, t in enumerate(m["transformation_history"])]
    lines += ["", "### Known limitations", ""] + [f"* {x}" for x in m["known_limitations"]]
    lines += ["", f"_Generated by the pipeline at {m['retrieval_or_generation_date']}._", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the CollectIQ pipeline")
    parser.add_argument("--profile", choices=["full", "demo", "test"], default=None)
    parser.add_argument("--skip-train", action="store_true",
                        help="Reuse registered model artifacts if present")
    args = parser.parse_args(argv)
    summary = run(profile=args.profile, skip_train=args.skip_train)
    return 0 if summary.get("status") == "SUCCESS" else 1


if __name__ == "__main__":
    sys.exit(main())
