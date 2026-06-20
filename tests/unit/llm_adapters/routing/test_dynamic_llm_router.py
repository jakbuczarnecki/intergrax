# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.agents.authoring.dynamic_llm_router import wrap_dynamic_llm_router
from intergrax.agents.authoring.llm_router import StepLLMRouter
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.registry.profile import LLMProfile
from intergrax.llm_adapters.routing import (
    BudgetExceededDegradeRule,
    LLMRoutingProfile,
    RoutingContext,
)


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_dynamic_router_applies_degrade_rule() -> None:
    router = StepLLMRouter(
        allowed_models=("premium", "economy"),
        default_model="premium",
    )
    routing = LLMRoutingProfile(
        default_profile=LLMProfile(provider=LLMProvider.OPENAI, model="premium"),
        rules=(BudgetExceededDegradeRule(),),
    )
    dynamic = wrap_dynamic_llm_router(
        router,
        routing_profile=routing,
        context_provider=lambda: RoutingContext(budget_degrade_active=True),
    )
    result = await dynamic.complete("hello", model_hint="premium")
    assert result.model_id == "economy"
