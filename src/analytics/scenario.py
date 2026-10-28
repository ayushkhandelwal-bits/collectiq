"""Scenario analysis on top of model estimates.

Scenarios adjust *copies* of the scored candidate table - historical data and
trained models are never modified or retrained. Outputs are hypothetical.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.optimization.optimizer import Constraints, solve_milp


@dataclass
class ScenarioAssumptions:
    """Hypothetical changes relative to the model-based baseline."""

    budget: float
    capacity_hours: float
    delinquency_multiplier: float = 1.0      # scales the eligible population size
    recovery_prob_multiplier: float = 1.0    # scales P(payment)
    recovery_amount_multiplier: float = 1.0  # scales E[amount | payment]
    product_weights: dict[str, float] = field(default_factory=dict)  # composition change

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def adjust_candidates(cand: pd.DataFrame, accounts: pd.DataFrame, a: ScenarioAssumptions,
                      seed: int = 42) -> pd.DataFrame:
    """Apply scenario assumptions to a copy of the eligible candidates."""
    rng = np.random.default_rng(seed)
    c = cand.copy()
    prod = accounts.set_index("loan_id")["loan_product"]
    c["loan_product"] = c["loan_id"].map(prod)
    acc = c[["loan_id", "loan_product"]].drop_duplicates("loan_id")
    weights = acc["loan_product"].map(lambda p: a.product_weights.get(p, 1.0)).to_numpy()
    expected = weights * a.delinquency_multiplier
    whole = np.floor(expected).astype(int)
    extra = (rng.random(len(acc)) < (expected - whole)).astype(int)
    copies = whole + extra
    pieces = []
    for k in range(int(copies.max()) if len(copies) else 0):
        ids = acc.loc[copies > k, "loan_id"]
        part = c[c["loan_id"].isin(ids)].copy()
        if k > 0:
            part["loan_id"] = part["loan_id"] + f"~S{k}"  # hypothetical replica
        pieces.append(part)
    if not pieces:
        return c.iloc[0:0]
    c = pd.concat(pieces, ignore_index=True)
    return rescale_values(c, a.recovery_prob_multiplier, a.recovery_amount_multiplier)


def rescale_values(c: pd.DataFrame, prob_mult: float, amount_mult: float) -> pd.DataFrame:
    """Apply multipliers to both the with-action and the no-action expected cash, keeping
    the incremental value definition used by the prioritisation engine."""
    c = c.copy()
    p0 = c["p_payment_no_action"].clip(lower=1e-9)
    amount_no_action = c["expected_cash_no_action"] / p0
    c["p_payment"] = np.clip(c["p_payment"] * prob_mult, 0, 1)
    c["p_payment_no_action"] = np.clip(c["p_payment_no_action"] * prob_mult, 0, 1)
    c["amount_if_paid"] = c["amount_if_paid"] * amount_mult
    c["expected_cash_with_action"] = c["p_payment"] * c["amount_if_paid"]
    c["expected_cash_no_action"] = c["p_payment_no_action"] * amount_no_action * amount_mult
    c["expected_gross_recovery"] = c["expected_cash_with_action"] - c["expected_cash_no_action"]
    c["expected_net_recovery"] = c["expected_gross_recovery"] - c["action_cost"]
    return c


def run_scenario(cand: pd.DataFrame, accounts: pd.DataFrame, a: ScenarioAssumptions,
                 base_cons: Constraints) -> dict[str, Any]:
    """Optimise under scenario assumptions and return headline results."""
    c = adjust_candidates(cand, accounts, a)
    cons = Constraints(budget=a.budget, capacity_hours=a.capacity_hours,
                       action_caps=base_cons.action_caps,
                       max_actions_per_account=base_cons.max_actions_per_account,
                       time_limit=base_cons.time_limit, mip_rel_gap=base_cons.mip_rel_gap)
    res = solve_milp(c, cons)
    tot = res.totals(cons)
    return {"assumptions": a.as_dict(), "status": res.status, "message": res.message,
            "eligible_accounts": int(c["loan_id"].nunique()), **tot,
            "by_action": (res.selected.groupby("action_type").size().to_dict()
                          if not res.selected.empty else {})}


def sensitivity(cand: pd.DataFrame, accounts: pd.DataFrame, a: ScenarioAssumptions,
                base_cons: Constraints, variable: str,
                multipliers: list[float]) -> pd.DataFrame:
    """Re-optimise across a range of budget or capacity levels."""
    rows = []
    for m in multipliers:
        b = ScenarioAssumptions(**{**a.as_dict()})
        if variable == "budget":
            b.budget = a.budget * m
        else:
            b.capacity_hours = a.capacity_hours * m
        r = run_scenario(cand, accounts, b, base_cons)
        rows.append({"multiplier": m, "budget": b.budget, "capacity_hours": b.capacity_hours,
                     "net": r["net"], "gross": r["gross"], "cost": r["cost"],
                     "selected_accounts": r["selected_accounts"], "status": r["status"]})
    return pd.DataFrame(rows)
