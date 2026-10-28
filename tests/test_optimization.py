"""Optimisation engine: constraints, edge cases, failure handling, baselines."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from src.optimization import optimizer as opt
from src.optimization.optimizer import Constraints, run_baselines, solve_greedy, solve_milp

ACTIONS = {"Reminder notification": (15, 0.0), "Outbound call": (120, 0.25),
           "Specialist review": (1200, 1.5)}


def make_candidates(n_accounts=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_accounts):
        stage = rng.choice(["1-29 DPD", "90+ DPD"])
        for a, (cost, hours) in ACTIONS.items():
            gross = float(rng.gamma(2, 600 if a != "Specialist review" else 900))
            rows.append({"loan_id": f"L{i:03d}", "stage": stage, "action_type": a,
                         "expected_gross_recovery": gross, "action_cost": float(cost),
                         "agent_hours": hours, "expected_net_recovery": gross - cost,
                         "p_payment": 0.5, "p_payment_no_action": 0.3})
    return pd.DataFrame(rows)


def cons(**kw):
    base = {"budget": 6000.0, "capacity_hours": 6.0,
            "action_caps": {"Reminder notification": 20, "Outbound call": 15,
                            "Specialist review": 3}}
    base.update(kw)
    return Constraints(**base)


def check_feasible(sel, c):
    assert sel["action_cost"].sum() <= c.budget + 1e-6
    assert sel["agent_hours"].sum() <= c.capacity_hours + 1e-6
    assert sel.groupby("loan_id").size().max() <= c.max_actions_per_account
    for a, cap in c.action_caps.items():
        assert (sel["action_type"] == a).sum() <= cap


def test_constraints_respected():
    c = cons()
    res = solve_milp(make_candidates(), c)
    assert res.status == opt.STATUS_OPTIMAL
    check_feasible(res.selected, c)
    assert (res.selected["expected_net_recovery"] > 0).all()
    assert list(res.selected["assigned_priority"]) == list(range(1, len(res.selected) + 1))


def test_two_actions_per_account_allowed_when_configured():
    c = cons(max_actions_per_account=2, budget=1e7, capacity_hours=1e4,
             action_caps={a: 1e6 for a in ACTIONS})
    res = solve_milp(make_candidates(10), c)
    check_feasible(res.selected, c)
    assert res.selected.groupby("loan_id").size().max() == 2


def test_matches_brute_force_on_small_instance():
    cand = make_candidates(4, seed=3)
    c = cons(budget=1400.0, capacity_hours=1.8, action_caps={a: 2 for a in ACTIONS})
    res = solve_milp(cand, c)
    pos = cand[cand["expected_net_recovery"] > 0].reset_index(drop=True)
    best = 0.0
    for mask in itertools.product([0, 1], repeat=len(pos)):
        sel = pos[np.array(mask, bool)]
        if sel.empty:
            continue
        if (sel["action_cost"].sum() <= c.budget and sel["agent_hours"].sum() <= c.capacity_hours
                and sel.groupby("loan_id").size().max() <= 1
                and all((sel["action_type"] == a).sum() <= 2 for a in ACTIONS)):
            best = max(best, sel["expected_net_recovery"].sum())
    assert abs(res.selected["expected_net_recovery"].sum() - best) < 1e-6


def test_optimal_at_least_greedy():
    cand, c = make_candidates(80, seed=5), cons()
    assert (solve_milp(cand, c).selected["expected_net_recovery"].sum()
            >= solve_greedy(cand, c).selected["expected_net_recovery"].sum() - 1e-6)


def test_empty_population():
    res = solve_milp(make_candidates().iloc[0:0], cons())
    assert res.status == opt.STATUS_EMPTY and res.selected.empty


def test_zero_budget_selects_nothing_costly():
    res = solve_milp(make_candidates(), cons(budget=0.0))
    assert res.ok
    assert res.selected["action_cost"].sum() == 0


def test_insufficient_capacity_uses_only_zero_hour_actions():
    res = solve_milp(make_candidates(), cons(capacity_hours=0.0))
    assert res.ok
    assert (res.selected["agent_hours"] == 0).all()


def test_all_actions_infeasible_by_caps():
    res = solve_milp(make_candidates(), cons(action_caps={a: 0 for a in ACTIONS}))
    assert res.status == opt.STATUS_EMPTY


def test_infeasible_coverage_reported():
    res = solve_milp(make_candidates(), cons(budget=100.0, min_coverage={"90+ DPD": 25}))
    assert res.status == opt.STATUS_INFEASIBLE
    assert "infeasible" in res.message.lower()


def test_solver_failure_falls_back_to_labelled_heuristic(monkeypatch):
    import scipy.optimize

    def boom(*a, **k):
        raise RuntimeError("solver crashed")

    monkeypatch.setattr(scipy.optimize, "milp", boom)
    c = cons()
    res = solve_milp(make_candidates(), c)
    assert res.status == opt.STATUS_HEURISTIC
    assert "heuristic" in res.message.lower()
    check_feasible(res.selected, c)


def test_reproducible():
    cand, c = make_candidates(50, seed=9), cons()
    a, b = solve_milp(cand, c), solve_milp(cand, c)
    assert a.selected[["loan_id", "action_type"]].equals(b.selected[["loan_id", "action_type"]])


def test_baselines_respect_constraints_and_output_schema():
    cand, c = make_candidates(), cons()
    queue = cand.groupby("loan_id").first().reset_index()[["loan_id", "stage"]]
    queue = queue.assign(outstanding_principal=np.arange(len(queue)) * 1000.0,
                         risk_probability=0.5, days_past_due=30,
                         delinquency_bucket=queue["stage"])
    out = run_baselines(cand, queue, c, {"1-29 DPD": "Reminder notification",
                                         "90+ DPD": "Outbound call"})
    assert set(out) == set(opt.BASELINES)
    for b in out.values():
        assert b["cost"] <= c.budget + 1e-6 and b["hours"] <= c.capacity_hours + 1e-6
        for k in ("selected_accounts", "gross", "cost", "net", "budget_utilization"):
            assert k in b
    best = solve_milp(cand, c).totals(c)["net"]
    assert all(best >= b["net"] - 1e-6 for b in out.values())


def test_invalid_negative_budget():
    res = solve_milp(make_candidates(), cons(budget=-1.0))
    assert res.status == opt.STATUS_INFEASIBLE


@pytest.mark.parametrize("missing", ["action_cost", "agent_hours"])
def test_missing_columns_raise(missing):
    with pytest.raises(ValueError):
        solve_milp(make_candidates().drop(columns=[missing]), cons())


def test_scenario_baseline_matches_optimizer(built):
    """Scenario Lab with unchanged assumptions reproduces the optimiser exactly."""
    from src.analytics.scenario import ScenarioAssumptions, run_scenario
    from src.config import path
    from src.optimization.service import default_constraints
    from src.prioritization.engine import Policy, eligible_candidates

    pop = pd.read_parquet(path("processed_dir") / "collections_population.parquet")
    cand = pd.read_parquet(path("processed_dir") / "action_candidates.parquet")
    ec = eligible_candidates(pop, cand, Policy.from_settings())
    c = default_constraints()
    direct = solve_milp(ec, c).totals(c)["net"]
    scen = run_scenario(ec, pop[["loan_id", "loan_product"]],
                        ScenarioAssumptions(c.budget, c.capacity_hours), c)
    assert abs(scen["net"] - direct) < 1.0
