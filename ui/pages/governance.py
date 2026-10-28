"""Data & Governance: provenance, schema, quality, pipeline history, exports."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.database import db
from src.preprocessing.quality import describe_missing
from ui import data
from ui.components.core import (
    download_csv,
    empty_state,
    fmt_date,
    kpi_strip,
    note,
    num,
    page_header,
    provenance_band,
)


def render() -> None:
    page_header("Data & Governance",
                "Where the data comes from, how it is structured and validated, and what it "
                "cannot tell you.")
    m = data.manifest()
    if not m:
        empty_state("No source manifest found. Run the pipeline.")
        return
    provenance_band([("Dataset", m["dataset_name"]), ("Version", m["dataset_version"]),
                     ("Generated", fmt_date(m["retrieval_or_generation_date"]))])
    loans = data.loans()
    origin = loans["data_origin"].value_counts()
    kpi_strip([
        {"label": "Synthetic loan records", "value": num(origin.get("SYNTHETIC", 0))},
        {"label": "Public loan records", "value": num(origin.get("PUBLIC", 0))},
        {"label": "Monthly snapshots", "value": num(m["record_counts"]["repayment_snapshots"])},
        {"label": "Collection events", "value": num(m["record_counts"]["collection_events"])},
    ])
    note("<b>Synthetic data.</b> Every record is simulated. The preferred public source "
         "(LendingClub data on Kaggle) could not be accessed through an authorised route in the "
         f"build environment — decision: <code>{m['primary_source']['reason_not_used']}</code>.",
         adverse=True)

    t1, t2, t3, t4, t5 = st.tabs(["Provenance", "Schema & dictionary", "Data quality",
                                  "Pipeline & models", "Exports"])
    with t1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Source manifest**")
            st.json({k: m[k] for k in ("dataset_name", "source_type", "acquisition_method",
                                       "file_location", "combined_checksum", "currency",
                                       "license_or_terms", "generation")}, expanded=False)
            st.markdown("**Files**")
            st.dataframe(pd.DataFrame(m["files"]), hide_index=True, width="stretch")
        with c2:
            st.markdown("**Transformation history**")
            st.markdown("\n".join(f"{i + 1}. {t}" for i, t in
                                  enumerate(m["transformation_history"])))
            st.markdown("**Known limitations**")
            st.markdown("\n".join(f"* {x}" for x in m["known_limitations"]))
    with t2:
        counts = db.table_counts()
        st.markdown("**Database tables**")
        st.dataframe(counts, hide_index=True, width="stretch")
        dd = db.data_dictionary()
        live = db.actual_schema()
        merged = dd.merge(live[["table_name", "column_name"]], how="outer",
                          on=["table_name", "column_name"], indicator=True)
        mism = merged[merged["_merge"] != "both"]
        if mism.empty:
            st.success(f"Dictionary verified against the live database: {len(dd)} columns match.")
        else:
            st.error(f"{len(mism)} columns differ between dictionary and database.")
            st.dataframe(mism, hide_index=True)
        term = st.text_input("Search the data dictionary", placeholder="e.g. recovery, loan_id")
        view = dd
        if term.strip():
            t = term.strip().lower()
            view = dd[dd.apply(lambda r: t in " ".join(map(str, r.values)).lower(), axis=1)]
        st.dataframe(view, hide_index=True, width="stretch", height=380,
                     column_config={"definition": st.column_config.TextColumn(width="large")})
        st.markdown("**Missing values — loan_accounts**")
        miss = describe_missing(loans)
        miss = miss[miss["missing_count"] > 0]
        st.dataframe(miss, hide_index=True, width="stretch", column_config={
            "missing_rate": st.column_config.NumberColumn("Missing rate", format="percent")})
        st.caption("Outcome fields (charge-off, recovery) are null by design for loans that "
                   "have not charged off; contact_restriction is null when none applies.")
    with t3:
        dq = data.latest_quality()
        if dq.empty:
            empty_state("No data-quality results recorded.")
        else:
            fails = (dq["check_status"] == "FAIL").sum()
            warns = (dq["check_status"] == "WARN").sum()
            st.markdown(f"Latest run `{dq['pipeline_run_id'].iloc[0]}` — {len(dq)} checks, "
                        f"**{fails} failed**, {warns} warnings. Critical failures block "
                        "database loading and model training.")
            status = st.multiselect("Status", ["PASS", "WARN", "FAIL", "INFO"],
                                    placeholder="All statuses")
            v = dq[dq["check_status"].isin(status)] if status else dq
            st.dataframe(v[["table_name", "check_name", "check_status", "severity",
                            "affected_record_count", "details"]], hide_index=True,
                         width="stretch", height=420)
            download_csv(dq, "Download validation report (CSV)",
                         "collectiq_data_quality.csv", "dl_dq")
    with t4:
        st.markdown("**ETL execution history**")
        st.dataframe(data.table("pipeline_runs", "started_at", 20), hide_index=True,
                     width="stretch")
        reg = data.registry()
        st.markdown("**Model artifacts**")
        st.dataframe(pd.DataFrame([{"artifact": k, "model": v.get("model_name"),
                                    "version": v.get("model_version"),
                                    "dataset_version": v.get("dataset_version"),
                                    "sha256": v.get("sha256", "")[:16] + "…",
                                    "trained_at": v.get("trained_at")}
                                   for k, v in reg.items()]), hide_index=True,
                     width="stretch")
        st.markdown("**Re-run the pipeline**")
        st.caption("Regenerates staging data, re-validates, reloads the database and re-scores "
                   "the queue. Raw source files are reused, never overwritten.")
        retrain = st.checkbox("Retrain models", value=False)
        if st.button("Run pipeline now"):
            from src.pipeline.run_pipeline import run

            with st.spinner("Running pipeline…"):
                try:
                    summary = run(skip_train=not retrain)
                    data.clear_caches()
                    st.success(f"Pipeline {summary['status']} in {summary['runtime_seconds']}s.")
                except Exception as exc:
                    st.error(f"Pipeline failed: {type(exc).__name__}: {exc}")
    with t5:
        st.markdown("Download analytical outputs as CSV.")
        q = data.default_queue().drop(columns=["permitted_actions"], errors="ignore")
        download_csv(q, "Prioritisation queue (default policy)", "collectiq_queue.csv", "ex_q")
        download_csv(db.data_dictionary(), "Data dictionary", "collectiq_data_dictionary.csv",
                     "ex_dd")
        download_csv(data.latest_quality(), "Data-quality report", "collectiq_dq.csv", "ex_dqr")
        download_csv(data.parquet("monthly_trends.parquet"), "Monthly portfolio trends",
                     "collectiq_monthly_trends.csv", "ex_tr")
        download_csv(data.table("model_runs", "trained_at", 100), "Model runs",
                     "collectiq_model_runs.csv", "ex_mr")
