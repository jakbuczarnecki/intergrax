# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.registry.profile import LLMProfile
from intergrax.llm_adapters.routing import (
    AllowlistViolationError,
    BudgetBelowRule,
    BudgetExceededDegradeRule,
    LLMRoutingEvaluator,
    LLMRoutingProfile,
    LLMRoutingRuleBase,
    RoutingContext,
    RoutingHint,
    RoutingTarget,
    TaskClassRule,
    TokenThresholdRule,
)


def _profile(provider: LLMProvider, model: str) -> LLMProfile:
    return LLMProfile(provider=provider, model=model)


@pytest.mark.unit
@pytest.mark.gate
def test_evaluator_first_match_by_priority() -> None:
    cheap = _profile(LLMProvider.GROQ, "cheap")
    premium = _profile(LLMProvider.OPENAI, "gpt-4o")
    routing = LLMRoutingProfile(
        default_profile=premium,
        allowed_profiles=(premium, cheap),
        rules=(
            TaskClassRule(classes=("simple",), hint=RoutingHint.CHEAPEST, priority=5),
            BudgetBelowRule(threshold=0.2, profile=cheap, priority=10),
        ),
    )
    evaluation = LLMRoutingEvaluator().evaluate(
        routing,
        RoutingContext(task_class="simple", budget_remaining_ratio=0.1),
    )
    assert evaluation.matched_rule_id == "builtin.budget_below"
    assert evaluation.selected_profile.model == "cheap"


@pytest.mark.unit
@pytest.mark.gate
def test_evaluator_rejects_disallowed_profile() -> None:
    routing = LLMRoutingProfile(
        default_profile=_profile(LLMProvider.OPENAI, "gpt-4o"),
        allowed_profiles=(_profile(LLMProvider.OPENAI, "gpt-4o"),),
        rules=(
            BudgetBelowRule(
                threshold=0.5,
                profile=_profile(LLMProvider.GROQ, "llama"),
                priority=10,
            ),
        ),
    )
    with pytest.raises(AllowlistViolationError):
        LLMRoutingEvaluator().evaluate(
            routing,
            RoutingContext(budget_remaining_ratio=0.1),
        )


@pytest.mark.unit
@pytest.mark.gate
def test_budget_degrade_rule_sets_cheapest_hint() -> None:
    routing = LLMRoutingProfile(
        default_profile=_profile(LLMProvider.OPENAI, "gpt-4o"),
        rules=(BudgetExceededDegradeRule(),),
    )
    evaluation = LLMRoutingEvaluator().evaluate(
        routing,
        RoutingContext(budget_degrade_active=True),
    )
    assert evaluation.policy_route_hint == "cheapest"
    assert evaluation.matched_rule_id == "builtin.budget_degrade"


@pytest.mark.unit
@pytest.mark.gate
def test_token_threshold_rule() -> None:
    routing = LLMRoutingProfile(
        default_profile=_profile(LLMProvider.OPENAI, "gpt-4o"),
        rules=(TokenThresholdRule(threshold=1000, hint=RoutingHint.CHEAPEST),),
    )
    evaluation = LLMRoutingEvaluator().evaluate(
        routing,
        RoutingContext(tokens_used=1500),
    )
    assert evaluation.policy_route_hint == "cheapest"


class _CustomRule(LLMRoutingRuleBase):
    rule_id = "custom.app_rule"
    priority = 20

    def matches(self, context: RoutingContext) -> bool:
        return context.agent_id == "legal"

    def resolve(self, context: RoutingContext) -> RoutingTarget:
        return RoutingTarget(
            profile=_profile(LLMProvider.VLLM, "meta-llama/Llama-3.1-8B"),
            reason="legal_agent",
        )


@pytest.mark.unit
@pytest.mark.gate
def test_custom_rule_class() -> None:
    vllm = _profile(LLMProvider.VLLM, "meta-llama/Llama-3.1-8B")
    routing = LLMRoutingProfile(
        default_profile=_profile(LLMProvider.OPENAI, "gpt-4o"),
        allowed_profiles=(_profile(LLMProvider.OPENAI, "gpt-4o"), vllm),
        rules=(_CustomRule(),),
    )
    evaluation = LLMRoutingEvaluator().evaluate(
        routing,
        RoutingContext(agent_id="legal"),
    )
    assert evaluation.matched_rule_id == "custom.app_rule"
    assert evaluation.selected_profile.provider is LLMProvider.VLLM
