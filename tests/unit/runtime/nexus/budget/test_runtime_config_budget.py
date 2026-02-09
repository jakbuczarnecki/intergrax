# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import pytest

from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.budget.budget_models import RunBudget, BudgetPolicy, BudgetEnforcementMode
from tests._support.builder import FakeLLMAdapter

pytestmark = pytest.mark.unit


def _base_config() -> RuntimeConfig:
    return RuntimeConfig(
        llm_adapter=FakeLLMAdapter(),
        enable_rag=False,
        enable_websearch=False,
    )


def test_config_allows_no_budget() -> None:
    cfg = _base_config()
    cfg.validate()  # should not raise


def test_config_rejects_budget_without_policy() -> None:
    cfg = _base_config()
    cfg.run_budget = RunBudget(max_llm_calls=1)

    with pytest.raises(ValueError, match="budget_policy"):
        cfg.validate()


def test_config_rejects_invalid_budget() -> None:
    cfg = _base_config()
    cfg.run_budget = RunBudget(max_llm_calls=-1)
    cfg.budget_policy = BudgetPolicy(enforcement_mode=BudgetEnforcementMode.ABORT)

    with pytest.raises(ValueError):
        cfg.validate()


def test_config_allows_budget_with_policy() -> None:
    cfg = _base_config()
    cfg.run_budget = RunBudget(max_llm_calls=1)
    cfg.budget_policy = BudgetPolicy(enforcement_mode=BudgetEnforcementMode.ABORT)

    cfg.validate()  # should not raise


def test_config_allows_policy_without_budget() -> None:
    cfg = _base_config()
    cfg.budget_policy = BudgetPolicy(enforcement_mode=BudgetEnforcementMode.ABORT)

    cfg.validate()  # should not raise
