# © Artur Czarnecki. All rights reserved.

"""Production run budget requirements (IDEAL-24.1)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from intergrax.runtime.nexus.budget.budget_models import RunBudget

if TYPE_CHECKING:
    from intergrax.runtime.nexus.config import RuntimeConfig


class ProductionBudgetPolicyError(ValueError):
    """Raised when production_mode runs lack mandatory budget."""


def default_production_run_budget() -> RunBudget:
    """Conservative harness default when host omits explicit limits."""
    return RunBudget(max_total_tokens=128_000, max_llm_calls=64, max_tool_calls=128)


def ensure_production_run_budget(config: RuntimeConfig) -> RuntimeConfig:
    """Require ``run_budget`` when ``production_mode`` is enabled."""
    if not config.production_mode:
        return config
    if config.run_budget is None:
        config.run_budget = default_production_run_budget()
    return config


def validate_production_run_budget(
    *,
    production_mode: bool,
    run_budget: RunBudget | None,
) -> None:
    if production_mode and run_budget is None:
        raise ProductionBudgetPolicyError(
            "run_budget is required when production_mode=True"
        )
