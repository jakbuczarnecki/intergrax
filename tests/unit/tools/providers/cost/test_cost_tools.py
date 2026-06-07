# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.runtime.architecture.cost_budget import BudgetEnvelope, BudgetScope
from intergrax.runtime.architecture.cost_quota import QuotaResourceType, ResourceQuota
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from intergrax.tools.providers.cost.contracts import CostCheckQuotaInput, CostGetRunBudgetInput
from intergrax.tools.providers.cost.service import cost_check_quota, cost_get_run_budget
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = pytest.mark.unit


def test_cost_get_run_budget_returns_limits() -> None:
    ctx = ToolWiringContext(
        run_budget=RunBudget(max_total_tokens=1000, max_tool_calls=5),
        cost_envelopes=(
            BudgetEnvelope(scope=BudgetScope.TENANT, scope_id="t-1", limit_amount=10.0, spent_amount=2.0),
        ),
    )
    out = cost_get_run_budget(ctx, CostGetRunBudgetInput(tenant_id="t-1"))
    assert out.configured is True
    assert out.max_total_tokens == 1000
    assert out.within_budget is True


def test_cost_check_quota_allows_request() -> None:
    ctx = ToolWiringContext(
        cost_quotas=(
            ResourceQuota(resource_type=QuotaResourceType.TOKENS, scope_id="agent-a", limit=100, used=10),
        )
    )
    out = cost_check_quota(
        ctx,
        CostCheckQuotaInput(resource_type="tokens", scope_id="agent-a", requested_units=5),
    )
    assert out.action == "allow"
