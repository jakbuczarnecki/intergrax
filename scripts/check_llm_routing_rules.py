#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""M-LLM-X.9.9 — validate LLM routing profile allowlists on reference hosts."""

from __future__ import annotations

import sys

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.registry.profile import LLMProfile
from intergrax.llm_adapters.routing import (
    BudgetBelowRule,
    LLMRoutingEvaluator,
    LLMRoutingProfile,
    RoutingContext,
    is_profile_allowed,
)


def _synthetic_routing_profile() -> LLMRoutingProfile:
    primary = LLMProfile(provider=LLMProvider.OPENAI, model="gpt-4o-mini")
    local = LLMProfile(provider=LLMProvider.VLLM, model="meta-llama/Llama-3.1-8B")
    return LLMRoutingProfile(
        default_profile=primary,
        allowed_profiles=(primary, local),
        rules=(BudgetBelowRule(threshold=0.2, profile=local),),
    )


def main() -> int:
    profile = _synthetic_routing_profile()
    errors = LLMRoutingEvaluator.validate_rules(profile.rules)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    for allowed in profile.allowed_profiles:
        if not is_profile_allowed(allowed, profile.allowed_profiles):
            print(f"allowlist invalid for {allowed.model}", file=sys.stderr)
            return 1

    evaluation = LLMRoutingEvaluator().evaluate(
        profile,
        RoutingContext(budget_remaining_ratio=0.1),
    )
    if not is_profile_allowed(evaluation.selected_profile, profile.allowed_profiles):
        print("evaluation selected profile outside allowlist", file=sys.stderr)
        return 1

    product = ApplicationEnvironmentProfile.product_defaults()
    wiring = product.adaptive_profile
    if not wiring.live_model_routing_enabled:
        print("product_defaults must enable live_model_routing for AHI gate parity", file=sys.stderr)
        return 1

    print("OK: llm routing rules allowlist + evaluator")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
