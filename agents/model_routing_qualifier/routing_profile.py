# © Artur Czarnecki. All rights reserved.

"""Production LLMRoutingProfile for Q4 qualification (reusable routing rules)."""

from __future__ import annotations

from intergrax.llm_adapters.registry.profile import LLMProfile
from intergrax.llm_adapters.routing import (
    BudgetBelowRule,
    LLMRoutingProfile,
    TaskClassRule,
)
from model_routing_qualifier.model_routing import (
    Q4_INVOKE_FAIL_TASK_CLASS,
    Q4_PRIMARY_TASK_CLASS,
    build_invoke_fail_profile,
    build_profile_a,
    build_profile_b,
)


def build_q4_qualification_routing_profile(
    *,
    profile_a: LLMProfile,
    profile_b: LLMProfile,
    invoke_fail_profile: LLMProfile,
) -> LLMRoutingProfile:
    return LLMRoutingProfile(
        default_profile=profile_a,
        allowed_profiles=(profile_a, profile_b, invoke_fail_profile),
        rules=(
            TaskClassRule(
                classes=(Q4_INVOKE_FAIL_TASK_CLASS,),
                profile=invoke_fail_profile,
                priority=15,
                rule_id="q4.task_class_invoke_fail",
            ),
            BudgetBelowRule(
                threshold=0.25,
                profile=profile_b,
                priority=10,
                rule_id="q4.budget_below_cheap",
            ),
            TaskClassRule(
                classes=(Q4_PRIMARY_TASK_CLASS,),
                profile=profile_a,
                priority=5,
                rule_id="q4.task_class_primary",
            ),
        ),
    )


def build_default_q4_qualification_routing_profile() -> LLMRoutingProfile:
    profile_a = build_profile_a()
    profile_b = build_profile_b()
    invoke_fail_profile = build_invoke_fail_profile()
    return build_q4_qualification_routing_profile(
        profile_a=profile_a,
        profile_b=profile_b,
        invoke_fail_profile=invoke_fail_profile,
    )


__all__ = ["build_default_q4_qualification_routing_profile", "build_q4_qualification_routing_profile"]
