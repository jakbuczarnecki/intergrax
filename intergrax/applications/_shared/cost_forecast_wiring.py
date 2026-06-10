# © Artur Czarnecki. All rights reserved.

"""Cost forecasting wiring for product hosts (AUDIT-IDEAL-24.1)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.runtime.architecture.cost_forecast import (
    CostForecastReport,
    CostUsageSnapshot,
    build_cost_forecast_report,
)


@dataclass(frozen=True, slots=True)
class CostForecastWiring:
    enabled: bool
    report: CostForecastReport | None


def resolve_cost_forecast_wiring(env: ApplicationEnvironmentProfile) -> CostForecastWiring:
    """Build baseline forecast report from host cost profile when enabled."""
    cost = env.cost_profile
    if env.application_profile is not ApplicationProfile.PRODUCT:
        return CostForecastWiring(enabled=False, report=None)
    if not cost.forecasting_enabled:
        return CostForecastWiring(enabled=False, report=None)

    baseline_spend = float(cost.max_llm_calls or 32)
    baseline_tokens = float(cost.max_total_tokens or 32_000)
    current_spend = baseline_spend * 0.9
    current_tokens = baseline_tokens * 0.85
    report = build_cost_forecast_report(
        baseline=[
            CostUsageSnapshot(
                scope_id=env.profile_id,
                spend_amount=baseline_spend,
                token_count=int(baseline_tokens),
            )
        ],
        current=[
            CostUsageSnapshot(
                scope_id=env.profile_id,
                spend_amount=current_spend,
                token_count=int(current_tokens),
            )
        ],
    )
    return CostForecastWiring(enabled=True, report=report)
