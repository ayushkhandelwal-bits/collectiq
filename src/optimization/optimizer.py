"""Collection resource-allocation optimisation.

Mixed-integer linear programme (solved with HiGHS via ``scipy.optimize.milp``):

    maximise    sum_{i,a} v_ia x_ia                 (v = expected net recovery)
    subject to  sum c_ia x_ia      <= B             (operating budget, INR)
                sum h_ia x_ia      <= H             (agent hours)
                sum_i x_ia         <= K_a   for a   (action-specific capacity)
                sum_a x_ia         <= M     for i   (actions per account)
                sum_{i in S} sum_a x_ia >= q        (optional minimum coverage of a stage)
                x_ia in {0, 1}, only for eligible account-action pairs

Values are model estimates. Results are simulations, not evidence of realised
improvement. If HiGHS is unavailable or errors, an explicitly labelled greedy
heuristic is used and reported as ``HEURISTIC`` (never presented as optimal).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

STATUS_OPTIMAL = "OPTIMAL"
STATUS_TIME_LIMIT = "FEASIBLE_TIME_LIMIT"
STATUS_HEURISTIC = "HEURISTIC"
STATUS_INFEASIBLE = "INFEASIBLE"
STATUS_EMPTY = "EMPTY"
STATUS_FAILED = "FAILED"


@dataclass
class Constraints:
    """Resource and policy constraints for one planning period."""

    budget: float
    capacity_hours: float
    action_caps: dict[str, float] = field(default_factory=dict)
    max_actions_per_account: int = 1
    min_coverage: dict[str, int] = field(default_factory=dict)  # stage -> minimum accounts
    time_limit: float = 60.0
    mip_rel_gap: float = 5e-4

    def as_dict(self) -> dict[str, Any]:
        return {"budget": self.budget, "capacity_hours": self.capacity_hours,
                "action_caps": self.action_caps,
                "max_actions_per_account": self.max_actions_per_account,
                "min_coverage": self.min_coverage}


@dataclass
class OptimizationResult:
    """Solver outcome and the selected account-action assignments."""

    status: str
    solver_name: str
    message: str
    selected: pd.DataFrame
    objective_value: float | None
    runtime_seconds: float
    candidate_count: int

    @property
    def ok(self) -> bool:
        return self.status in (STATUS_OPTIMAL, STATUS_TIME_LIMIT, STATUS_HEURISTIC)

    def totals(self, cons: Constraints) -> dict[str, float]:
        return allocation_totals(self.selected, cons)


def allocation_totals(sel: pd.DataFrame, cons: Constraints) -> dict[str, float]:
    """Totals and utilisation for a set of selected assignments."""
    if sel is None or sel.empty:
        return {"selected_accounts": 0, "selected_actions": 0, "gross": 0.0, "cost": 0.0,
                "net": 0.0, "hours": 0.0, "budget_utilization": 0.0,
                "capacity_utilization": 0.0}
    cost, hours = float(sel["action_cost"].sum()), float(sel["agent_hours"].sum())
    return {
        "selected_accounts": int(sel["loan_id"].nunique()), "selected_actions": int(len(sel)),
        "gross": float(sel["expected_gross_recovery"].sum()), "cost": cost,
        "net": float(sel["expected_net_recovery"].sum()), "hours": hours,
        "budget_utilization": cost / cons.budget if cons.budget > 0 else 0.0,
        "capacity_utilization": hours / cons.capacity_hours if cons.capacity_hours > 0 else 0.0,
    }


def _prepare(cand: pd.DataFrame, cons: Constraints) -> pd.DataFrame:
    """Drop candidates that can never be optimal (non-positive value) unless a coverage
    constraint may require them."""
    c = cand.copy()
    required = ["loan_id", "action_type", "expected_net_recovery", "expected_gross_recovery",
                "action_cost", "agent_hours", "stage"]
    missing = [r for r in required if r not in c.columns]
    if missing:
        raise ValueError(f"Candidates missing columns: {missing}")
    if not cons.min_coverage:
        c = c[c["expected_net_recovery"] > 0]
    caps = cons.action_caps or {}
    c = c[c["action_type"].map(lambda a: caps.get(a, np.inf) > 0)]
    return c.reset_index(drop=True)


def _rank(sel: pd.DataFrame) -> pd.DataFrame:
    sel = sel.sort_values("expected_net_recovery", ascending=False).reset_index(drop=True)
    sel["assigned_priority"] = np.arange(1, len(sel) + 1)
    return sel


def solve_milp(cand: pd.DataFrame, cons: Constraints) -> OptimizationResult:
    """Solve the allocation MILP with HiGHS; falls back to a labelled heuristic on error."""
    t0 = time.time()
    if cons.budget < 0 or cons.capacity_hours < 0:
        return OptimizationResult(STATUS_INFEASIBLE, "validation", "Budget and capacity must "
                                  "be non-negative.", cand.iloc[0:0], None, 0.0, 0)
    c = _prepare(cand, cons)
    if c.empty:
        msg = ("No eligible account-action candidates with positive estimated net value. "
               "Relax eligibility rules or check that the queue is populated.")
        if cons.min_coverage and any(v > 0 for v in cons.min_coverage.values()):
            return OptimizationResult(STATUS_INFEASIBLE, "validation",
                                      "Minimum coverage requested but no candidates exist.",
                                      c, None, time.time() - t0, 0)
        return OptimizationResult(STATUS_EMPTY, "n/a", msg, c, 0.0, time.time() - t0, 0)
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
    except ImportError as exc:  # pragma: no cover - depends on SciPy build
        res = solve_greedy(cand, cons)
        res.message = f"HiGHS MILP unavailable ({exc}); greedy heuristic used. " + res.message
        return res

    n = len(c)
    rows, lo, hi = [], [], []
    rows.append(sparse.csr_matrix(c["action_cost"].to_numpy(float)[None, :]))
    lo.append(-np.inf)
    hi.append(cons.budget)
    rows.append(sparse.csr_matrix(c["agent_hours"].to_numpy(float)[None, :]))
    lo.append(-np.inf)
    hi.append(cons.capacity_hours)
    for a, cap in (cons.action_caps or {}).items():
        mask = (c["action_type"] == a).to_numpy(float)
        if mask.any() and np.isfinite(cap):
            rows.append(sparse.csr_matrix(mask[None, :]))
            lo.append(-np.inf)
            hi.append(float(cap))
    acc_codes, acc_index = pd.factorize(c["loan_id"])
    A_acc = sparse.csr_matrix((np.ones(n), (acc_codes, np.arange(n))),
                              shape=(len(acc_index), n))
    rows.append(A_acc)
    lo.extend([-np.inf] * len(acc_index))
    hi.extend([float(cons.max_actions_per_account)] * len(acc_index))
    for stage, k in (cons.min_coverage or {}).items():
        if k and k > 0:
            mask = (c["stage"] == stage).to_numpy(float)
            rows.append(sparse.csr_matrix(mask[None, :]))
            lo.append(float(k))
            hi.append(np.inf)
    A = sparse.vstack(rows).tocsr()
    try:
        res = milp(c=-c["expected_net_recovery"].to_numpy(float),
                   constraints=LinearConstraint(A, np.array(lo), np.array(hi)),
                   integrality=np.ones(n), bounds=Bounds(0, 1),
                   options={"time_limit": cons.time_limit, "mip_rel_gap": cons.mip_rel_gap,
                            "disp": False})
    except Exception as exc:  # solver crash -> labelled heuristic
        res_h = solve_greedy(cand, cons)
        res_h.message = f"HiGHS solver error ({exc}); greedy heuristic used. " + res_h.message
        return res_h
    elapsed = time.time() - t0
    if res.status == 2:
        return OptimizationResult(STATUS_INFEASIBLE, "HiGHS (scipy.optimize.milp)",
                                  _infeasibility_hint(c, cons), c.iloc[0:0], None, elapsed, n)
    if res.x is None:
        return OptimizationResult(STATUS_FAILED, "HiGHS (scipy.optimize.milp)",
                                  f"Solver returned no solution: {res.message}", c.iloc[0:0],
                                  None, elapsed, n)
    x = np.round(res.x).astype(bool)
    sel = _rank(c.loc[x].copy())
    status = STATUS_OPTIMAL if res.status == 0 else STATUS_TIME_LIMIT
    gap = getattr(res, "mip_gap", None)
    msg = ("Optimal within relative MIP gap "
           f"{cons.mip_rel_gap:.2%}." if status == STATUS_OPTIMAL else
           f"Time limit reached; best feasible solution returned (gap {gap}).")
    return OptimizationResult(status, "HiGHS (scipy.optimize.milp)", msg, sel, float(-res.fun),
                              elapsed, n)


def _infeasibility_hint(c: pd.DataFrame, cons: Constraints) -> str:
    parts = []
    for stage, k in (cons.min_coverage or {}).items():
        sub = c[c["stage"] == stage]
        n_acc = sub["loan_id"].nunique()
        if k > n_acc:
            parts.append(f"minimum coverage for {stage} ({k}) exceeds eligible accounts "
                         f"({n_acc})")
            continue
        cheapest = sub.groupby("loan_id")["action_cost"].min().nsmallest(int(k))
        hours = sub.groupby("loan_id")["agent_hours"].min().nsmallest(int(k))
        if cheapest.sum() > cons.budget:
            parts.append(f"covering {k} {stage} accounts needs at least "
                         f"₹{cheapest.sum():,.0f} but the budget is ₹{cons.budget:,.0f}")
        if hours.sum() > cons.capacity_hours:
            parts.append(f"covering {k} {stage} accounts needs at least {hours.sum():,.1f} "
                         f"agent hours but {cons.capacity_hours:,.1f} are available")
    hint = "; ".join(parts) or "constraints conflict (e.g. coverage vs. action caps)"
    return f"Problem is infeasible: {hint}. Increase budget/capacity or lower coverage."


def solve_greedy(cand: pd.DataFrame, cons: Constraints) -> OptimizationResult:
    """Greedy fallback: rank by net value per unit of scarcest resource and fill.

    Approximate - not guaranteed optimal; always reported as ``HEURISTIC``.
    """
    t0 = time.time()
    c = _prepare(cand, cons)
    if c.empty:
        return OptimizationResult(STATUS_EMPTY, "Greedy heuristic", "No candidates.", c, 0.0,
                                  0.0, 0)
    weight = (c["action_cost"] / max(cons.budget, 1e-9)
              + c["agent_hours"] / max(cons.capacity_hours, 1e-9)) + 1e-9
    c["_ratio"] = c["expected_net_recovery"] / weight
    sel = greedy_fill(c.sort_values("_ratio", ascending=False), cons)
    cov_ok = all((sel["stage"] == s).sum() >= k for s, k in (cons.min_coverage or {}).items())
    status = STATUS_HEURISTIC if cov_ok else STATUS_INFEASIBLE
    msg = ("Approximate greedy allocation (not guaranteed optimal)." if cov_ok else
           "Greedy heuristic could not satisfy minimum coverage.")
    return OptimizationResult(status, "Greedy heuristic", msg, _rank(sel.drop(columns="_ratio")),
                              float(sel["expected_net_recovery"].sum()), time.time() - t0,
                              len(c))


def greedy_fill(ordered: pd.DataFrame, cons: Constraints) -> pd.DataFrame:
    """Take candidates in the given order while every constraint still holds."""
    budget, hours = cons.budget, cons.capacity_hours
    caps = dict(cons.action_caps or {})
    per_acc: dict[str, int] = {}
    keep = []
    for idx, r in zip(ordered.index, ordered.itertuples(index=False)):
        if per_acc.get(r.loan_id, 0) >= cons.max_actions_per_account:
            continue
        if r.action_cost > budget + 1e-9 or r.agent_hours > hours + 1e-9:
            continue
        if caps.get(r.action_type, np.inf) < 1:
            continue
        keep.append(idx)
        budget -= r.action_cost
        hours -= r.agent_hours
        caps[r.action_type] = caps.get(r.action_type, np.inf) - 1
        per_acc[r.loan_id] = per_acc.get(r.loan_id, 0) + 1
    return ordered.loc[keep].copy()


BASELINES = {
    "Balance-first": "Accounts ranked by outstanding balance (descending); stage default action.",
    "Risk-first": "Accounts ranked by risk (roll-forward probability; 90+ DPD and charged-off "
                  "treated as observed maximum risk); stage default action.",
    "Oldest-delinquency-first": "Accounts ranked by days past due (descending); stage default "
                                "action.",
    "ENRV ranking (greedy)": "Accounts ranked by best-action expected net value and filled "
                             "greedily - the prioritisation queue without optimisation.",
}


def run_baselines(cand: pd.DataFrame, queue: pd.DataFrame, cons: Constraints,
                  default_action: dict[str, str]) -> dict[str, dict[str, Any]]:
    """Evaluate simple allocation rules under identical eligibility and constraints."""
    out: dict[str, dict[str, Any]] = {}
    if cand.empty:
        return {name: {"description": d, **allocation_totals(None, cons)}
                for name, d in BASELINES.items()}
    acc = queue[["loan_id", "outstanding_principal", "risk_probability", "days_past_due",
                 "delinquency_bucket"]].drop_duplicates("loan_id")
    risk = acc["risk_probability"].fillna(1.0)
    acc = acc.assign(_risk=risk)
    dflt = cand[cand.apply(lambda r: default_action.get(r["stage"]) == r["action_type"], axis=1)]
    # If the default action is not permitted for an account, use its cheapest permitted one.
    missing = set(cand["loan_id"]) - set(dflt["loan_id"])
    if missing:
        alt = (cand[cand["loan_id"].isin(missing)].sort_values("action_cost")
               .groupby("loan_id").head(1))
        dflt = pd.concat([dflt, alt])
    dflt = dflt.merge(acc, on="loan_id", how="left")
    orders = {
        "Balance-first": dflt.sort_values("outstanding_principal", ascending=False),
        "Risk-first": dflt.sort_values(["_risk", "outstanding_principal"], ascending=False),
        "Oldest-delinquency-first": dflt.sort_values(["days_past_due", "outstanding_principal"],
                                                     ascending=False),
    }
    best = cand.sort_values("expected_net_recovery", ascending=False).groupby("loan_id").head(1)
    orders["ENRV ranking (greedy)"] = best[best["expected_net_recovery"] > 0]
    for name, ordered in orders.items():
        sel = greedy_fill(ordered, cons)
        out[name] = {"description": BASELINES[name], **allocation_totals(sel, cons)}
    return out
