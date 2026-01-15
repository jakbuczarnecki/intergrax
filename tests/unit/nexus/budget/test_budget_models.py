from __future__ import annotations

import pytest

from intergrax.runtime.nexus.budget.budget_models import (
    RunBudget,
    BudgetPolicy,
    BudgetEnforcementMode,
)


pytestmark = pytest.mark.unit


def test_run_budget_validation_allows_none() -> None:
    budget = RunBudget()
    budget.validate()


def test_run_budget_validation_rejects_negative_values() -> None:
    budget = RunBudget(max_total_tokens=-1)
    with pytest.raises(ValueError):
        budget.validate()


def test_budget_policy_enum_is_stable() -> None:
    policy = BudgetPolicy(
        enforcement_mode=BudgetEnforcementMode.ABORT,
        hitl_reason=None,
    )
    assert policy.enforcement_mode is BudgetEnforcementMode.ABORT
