# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.runtime.architecture.cost_budget import BudgetScope, evaluate_budget_envelopes
from intergrax.runtime.architecture.cost_forecast import CostUsageSnapshot, build_cost_forecast_report
from intergrax.runtime.architecture.cost_quota import (
    QuotaResourceType,
    QuotaUsageRequest,
    evaluate_quota_enforcement,
)
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from intergrax.tools.providers.cost.contracts import (
    CostAnomalyOutput,
    CostCheckQuotaInput,
    CostCheckQuotaOutput,
    CostForecastPointOutput,
    CostForecastSpendInput,
    CostForecastSpendOutput,
    CostGetRunBudgetInput,
    CostGetRunBudgetOutput,
)
from intergrax.tools.registry.wiring import ToolWiringContext

COST_GET_RUN_BUDGET_TOOL_ID = "cost.get_run_budget"
COST_CHECK_QUOTA_TOOL_ID = "cost.check_quota"
COST_FORECAST_SPEND_TOOL_ID = "cost.forecast_spend"


def cost_get_run_budget(ctx: ToolWiringContext, params: CostGetRunBudgetInput) -> CostGetRunBudgetOutput:
    run_budget = ctx.run_budget
    if run_budget is None:
        return CostGetRunBudgetOutput(configured=False, within_budget=True)

    if not isinstance(run_budget, RunBudget):
        raise RuntimeError("run_budget_invalid_type")

    within_budget = True
    remaining_amount: float | None = None
    tenant_id = params.tenant_id.strip()
    if tenant_id and ctx.cost_envelopes:
        report = evaluate_budget_envelopes(list(ctx.cost_envelopes))
        tenant_decisions = [
            item for item in report.decisions if item.scope == BudgetScope.TENANT and item.scope_id == tenant_id
        ]
        if tenant_decisions:
            within_budget = all(item.within_budget for item in tenant_decisions)
            remaining_amount = min(item.remaining_amount for item in tenant_decisions)

    return CostGetRunBudgetOutput(
        configured=True,
        max_input_tokens=run_budget.max_input_tokens,
        max_output_tokens=run_budget.max_output_tokens,
        max_total_tokens=run_budget.max_total_tokens,
        max_llm_calls=run_budget.max_llm_calls,
        max_tool_calls=run_budget.max_tool_calls,
        max_wall_time_seconds=run_budget.max_wall_time_seconds,
        within_budget=within_budget,
        remaining_amount=remaining_amount,
    )


def cost_check_quota(ctx: ToolWiringContext, params: CostCheckQuotaInput) -> CostCheckQuotaOutput:
    if not ctx.cost_quotas:
        raise RuntimeError("cost_quotas_not_configured")
    try:
        resource_type = QuotaResourceType(params.resource_type.strip().lower())
    except ValueError as exc:
        raise RuntimeError("quota_resource_type_invalid") from exc
    report = evaluate_quota_enforcement(
        quotas=list(ctx.cost_quotas),
        requests=[
            QuotaUsageRequest(
                resource_type=resource_type,
                scope_id=params.scope_id.strip(),
                requested_units=params.requested_units,
            )
        ],
    )
    decision = report.decisions[0]
    return CostCheckQuotaOutput(
        action=decision.action.value,
        allowed_units=decision.allowed_units,
        reasons=list(decision.reasons),
    )


def cost_forecast_spend(ctx: ToolWiringContext, params: CostForecastSpendInput) -> CostForecastSpendOutput:
    if not ctx.cost_envelopes:
        raise RuntimeError("cost_envelopes_not_configured")

    baseline: list[CostUsageSnapshot] = []
    current: list[CostUsageSnapshot] = []
    for envelope in ctx.cost_envelopes:
        scope_id = f"{envelope.scope.value}:{envelope.scope_id}"
        spent_amount = float(envelope.spent_amount)
        current.append(
            CostUsageSnapshot(
                scope_id=scope_id,
                spend_amount=spent_amount,
                token_count=0,
            )
        )
        baseline.append(
            CostUsageSnapshot(
                scope_id=scope_id,
                spend_amount=spent_amount * params.baseline_spend_ratio,
                token_count=0,
            )
        )

    report = build_cost_forecast_report(
        baseline=baseline,
        current=current,
        growth_multiplier=params.growth_multiplier,
    )
    return CostForecastSpendOutput(
        schema_version=report.schema_version,
        forecasts=[
            CostForecastPointOutput(
                scope_id=item.scope_id,
                projected_spend=item.projected_spend,
                projected_tokens=item.projected_tokens,
            )
            for item in report.forecasts
        ],
        anomalies=[
            CostAnomalyOutput(
                scope_id=item.scope_id,
                severity=item.severity.value,
                spend_delta_ratio=item.spend_delta_ratio,
                token_delta_ratio=item.token_delta_ratio,
                reasons=list(item.reasons),
            )
            for item in report.anomalies
        ],
    )
