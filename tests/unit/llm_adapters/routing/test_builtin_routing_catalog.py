# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.registry.profile import LLMProfile
from intergrax.llm_adapters.routing import (
    AgentIdInRule,
    AlwaysRule,
    BUILTIN_ROUTING_RULE_TYPES,
    BudgetAboveRule,
    BudgetBelowRule,
    BudgetExceededDegradeRule,
    CompositeAllRule,
    CompositeAnyRule,
    LLMRoutingEvaluator,
    LLMRoutingProfile,
    ModelHintPresentRule,
    PolicyHintRule,
    RoutingContext,
    RoutingHint,
    StepIndexAtLeastRule,
    StepIndexBelowRule,
    TaskClassInRule,
    TaskClassNotInRule,
    TokenUsedAboveRule,
    TokenUsedBelowRule,
)


def _profile(model: str = "gpt-4o-mini") -> LLMProfile:
    return LLMProfile(provider=LLMProvider.OPENAI, model=model)


def _local() -> LLMProfile:
    return LLMProfile(provider=LLMProvider.VLLM, model="meta-llama/Llama-3.1-8B")


@pytest.mark.unit
@pytest.mark.gate
def test_builtin_catalog_exports_minimum_classes() -> None:
    assert len(BUILTIN_ROUTING_RULE_TYPES) >= 12
    assert BudgetBelowRule in BUILTIN_ROUTING_RULE_TYPES
    assert TaskClassInRule in BUILTIN_ROUTING_RULE_TYPES
    assert TokenUsedAboveRule in BUILTIN_ROUTING_RULE_TYPES


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.parametrize(
    ("rule", "context", "expected_rule_id"),
    [
        (BudgetBelowRule(threshold=0.5, profile=_local()), RoutingContext(budget_remaining_ratio=0.2), "builtin.budget_below"),
        (BudgetAboveRule(threshold=0.5, profile=_local()), RoutingContext(budget_remaining_ratio=0.8), "builtin.budget_above"),
        (TaskClassInRule(classes=("legal",), profile=_local()), RoutingContext(task_class="legal"), "builtin.task_class"),
        (TaskClassNotInRule(classes=("legal",), profile=_local()), RoutingContext(task_class="lab"), "builtin.task_class_not_in"),
        (TokenUsedAboveRule(threshold=100, hint=RoutingHint.CHEAPEST), RoutingContext(tokens_used=200), "builtin.token_threshold"),
        (TokenUsedBelowRule(threshold=100, profile=_local()), RoutingContext(tokens_used=50), "builtin.token_used_below"),
        (StepIndexAtLeastRule(min_step=3, profile=_local()), RoutingContext(step_index=4), "builtin.step_index_at_least"),
        (StepIndexBelowRule(max_step=3, profile=_local()), RoutingContext(step_index=1), "builtin.step_index_below"),
        (AgentIdInRule(agent_ids=("legal",), profile=_local()), RoutingContext(agent_id="legal"), "builtin.agent_id_in"),
        (ModelHintPresentRule(profile=_local()), RoutingContext(model_hint="economy"), "builtin.model_hint_present"),
        (PolicyHintRule(hint=RoutingHint.QUALITY), RoutingContext(), "builtin.policy_hint"),
        (AlwaysRule(profile=_local()), RoutingContext(), "builtin.always"),
        (BudgetExceededDegradeRule(), RoutingContext(budget_degrade_active=True), "builtin.budget_degrade"),
    ],
)
def test_predefined_rules_match_and_resolve(
    rule: object,
    context: RoutingContext,
    expected_rule_id: str,
) -> None:
    routing = LLMRoutingProfile(
        default_profile=_profile(),
        allowed_profiles=(_profile(), _local()),
        rules=(rule,),
    )
    evaluation = LLMRoutingEvaluator().evaluate(routing, context)
    assert evaluation.matched_rule_id == expected_rule_id


@pytest.mark.unit
@pytest.mark.gate
def test_composite_all_requires_every_nested_rule() -> None:
    routing = LLMRoutingProfile(
        default_profile=_profile(),
        allowed_profiles=(_profile(), _local()),
        rules=(
            CompositeAllRule(
                rules=(
                    TaskClassInRule(classes=("legal",)),
                    BudgetBelowRule(threshold=0.5, profile=_local()),
                ),
                profile=_local(),
                priority=20,
            ),
        ),
    )
    matched = LLMRoutingEvaluator().evaluate(
        routing,
        RoutingContext(task_class="legal", budget_remaining_ratio=0.1),
    )
    assert matched.matched_rule_id == "builtin.composite_all"
    assert matched.selected_profile.model == "meta-llama/Llama-3.1-8B"
    default = LLMRoutingEvaluator().evaluate(
        routing,
        RoutingContext(task_class="legal", budget_remaining_ratio=0.9),
    )
    assert default.matched_rule_id is None


@pytest.mark.unit
@pytest.mark.gate
def test_composite_any_matches_single_nested_rule() -> None:
    routing = LLMRoutingProfile(
        default_profile=_profile(),
        rules=(
            CompositeAnyRule(
                rules=(
                    TaskClassInRule(classes=("missing",)),
                    BudgetBelowRule(threshold=0.5, profile=_local()),
                ),
                hint=RoutingHint.CHEAPEST,
                priority=20,
            ),
        ),
    )
    evaluation = LLMRoutingEvaluator().evaluate(
        routing,
        RoutingContext(budget_remaining_ratio=0.1),
    )
    assert evaluation.matched_rule_id == "builtin.composite_any"
    assert evaluation.policy_route_hint == "cheapest"
