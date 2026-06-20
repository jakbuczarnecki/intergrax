# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.registry.profile import LLMProfile
from intergrax.llm_adapters.routing import (
    BudgetBelowRule,
    LLMRoutingEvaluator,
    LLMRoutingProfile,
    RoutingContext,
)
from intergrax.runtime.nexus.tracing.adapters.llm_routing_attempt import (
    LLMRoutingRuleDiagV1,
    emit_llm_routing_rule_diag,
    routing_evaluation_to_diag,
)


@pytest.mark.unit
@pytest.mark.gate
def test_routing_rule_diag_schema_and_emit() -> None:
    primary = LLMProfile(provider=LLMProvider.OPENAI, model="gpt-4o-mini")
    local = LLMProfile(provider=LLMProvider.VLLM, model="meta-llama/Llama-3.1-8B")
    evaluation = LLMRoutingEvaluator().evaluate(
        LLMRoutingProfile(
            default_profile=primary,
            allowed_profiles=(primary, local),
            rules=(BudgetBelowRule(threshold=0.2, profile=local),),
        ),
        RoutingContext(budget_remaining_ratio=0.1),
    )
    diag = routing_evaluation_to_diag(evaluation)
    assert diag.schema_id() == "intergrax.diag.engine.core_llm.routing_rule"
    assert diag.matched_rule_id == "builtin.budget_below"
    assert diag.profile_id == "vllm:meta-llama/Llama-3.1-8B"

    emitted: list[LLMRoutingRuleDiagV1] = []

    def _trace_event(**kwargs: object) -> None:
        payload = kwargs.get("payload")
        assert isinstance(payload, LLMRoutingRuleDiagV1)
        emitted.append(payload)

    emit_llm_routing_rule_diag(_trace_event, evaluation)
    assert len(emitted) == 1
    assert emitted[0].routing_reason.startswith("rule:")
