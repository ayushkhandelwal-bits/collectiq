"""Run an optimisation, compare baselines and persist results to the database."""

from __future__ import annotations

import uuid
from typing import Any

import pandas as pd

from src.config import load_settings
from src.database import db
from src.optimization.optimizer import (
    Constraints,
    OptimizationResult,
    allocation_totals,
    run_baselines,
    solve_milp,
)
from src.utils.helpers import to_json, utc_now


def default_constraints() -> Constraints:
    """Constraints from configuration."""
    o = load_settings()["optimization"]
    return Constraints(budget=float(o["default_budget"]),
                       capacity_hours=float(o["default_capacity_hours"]),
                       action_caps={k: float(v) for k, v in o["default_action_caps"].items()},
                       time_limit=float(o["time_limit_seconds"]),
                       mip_rel_gap=float(o["mip_rel_gap"]))


def optimize_and_compare(cand: pd.DataFrame, queue: pd.DataFrame, cons: Constraints
                         ) -> tuple[OptimizationResult, dict[str, Any]]:
    """Solve the MILP and evaluate baselines under identical constraints."""
    res = solve_milp(cand, cons)
    default_action = load_settings()["collections"]["default_action_by_stage"]
    baselines = run_baselines(cand, queue, cons, default_action)
    opt_net = res.totals(cons)["net"] if res.ok else None
    for b in baselines.values():
        b["incremental_net_vs_baseline"] = (opt_net - b["net"]) if opt_net is not None else None
    return res, baselines


def persist_run(res: OptimizationResult, baselines: dict[str, Any], cons: Constraints,
                queue: pd.DataFrame, planning_start: str, planning_end: str,
                extra_assumptions: dict[str, Any] | None = None) -> str:
    """Write ``optimization_runs`` and ``optimization_assignments`` in one transaction."""
    run_id = f"OR-{uuid.uuid4().hex[:12]}"
    tot = allocation_totals(res.selected, cons)
    run = {
        "optimization_run_id": run_id, "run_timestamp": utc_now(),
        "planning_period_start": planning_start, "planning_period_end": planning_end,
        "budget_limit": cons.budget, "available_capacity": cons.capacity_hours,
        "selected_account_count": tot["selected_accounts"],
        "estimated_gross_recovery": tot["gross"], "estimated_operating_cost": tot["cost"],
        "estimated_net_recovery": tot["net"], "solver_name": res.solver_name,
        "solver_status": res.status, "objective_value": res.objective_value,
        "baseline_comparison_json": to_json(baselines),
        "assumptions_json": to_json({**cons.as_dict(), "message": res.message,
                                     "runtime_seconds": res.runtime_seconds,
                                     "candidate_count": res.candidate_count,
                                     **(extra_assumptions or {})}),
    }
    assign = pd.DataFrame()
    if not res.selected.empty:
        q = queue.set_index("loan_id")
        s = res.selected
        assign = pd.DataFrame({
            "assignment_id": [f"{run_id}-{i:05d}" for i in range(len(s))],
            "optimization_run_id": run_id, "loan_id": s["loan_id"].to_numpy(),
            "action_type": s["action_type"].to_numpy(),
            "estimated_gross_recovery": s["expected_gross_recovery"].round(2).to_numpy(),
            "estimated_action_cost": s["action_cost"].to_numpy(),
            "estimated_net_recovery": s["expected_net_recovery"].round(2).to_numpy(),
            "required_capacity": s["agent_hours"].to_numpy(),
            "assigned_priority": s["assigned_priority"].to_numpy(),
            "eligibility_status": q.loc[s["loan_id"], "eligibility_status"].to_numpy(),
            "recommendation_reason": [
                f"Selected by {res.solver_name}: {a} estimated net ₹{v:,.0f} "
                f"(P(payment) {p:.0%}, cost ₹{c:,.0f})"
                for a, v, p, c in zip(s["action_type"], s["expected_net_recovery"],
                                      s["p_payment"], s["action_cost"])],
        })
    eng = db.get_engine()
    with eng.begin() as conn:
        pd.DataFrame([run]).to_sql("optimization_runs", conn, if_exists="append", index=False)
        if not assign.empty:
            assign.to_sql("optimization_assignments", conn, if_exists="append", index=False)
    return run_id
