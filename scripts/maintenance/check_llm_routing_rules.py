#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""M-LLM-X.9.9 / M-LLM-X.10.8 — validate routing catalog and reference hosts."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "agents", REPO_ROOT / "applications"):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from intergrax.applications._shared.lab_environment_profile import build_lab_environment_profile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.registry.profile import LLMProfile
from intergrax.llm_adapters.routing import (
    BUILTIN_ROUTING_RULE_TYPES,
    BudgetBelowRule,
    LLMRoutingEvaluator,
    LLMRoutingProfile,
    RoutingContext,
    is_profile_allowed,
)
from lab_application.host.settings import LabApplicationSettings


def _synthetic_routing_profile() -> LLMRoutingProfile:
    primary = LLMProfile(provider=LLMProvider.OPENAI, model="gpt-4o-mini")
    local = LLMProfile(provider=LLMProvider.VLLM, model="meta-llama/Llama-3.1-8B")
    return LLMRoutingProfile(
        default_profile=primary,
        allowed_profiles=(primary, local),
        rules=(BudgetBelowRule(threshold=0.2, profile=local),),
    )


def _validate_catalog_exports() -> list[str]:
    errors: list[str] = []
    if len(BUILTIN_ROUTING_RULE_TYPES) < 12:
        errors.append(
            f"BUILTIN_ROUTING_RULE_TYPES must export 12+ classes, got {len(BUILTIN_ROUTING_RULE_TYPES)}",
        )
    seen_ids: set[str] = set()
    for rule_type in BUILTIN_ROUTING_RULE_TYPES:
        if not rule_type.__name__.endswith("Rule"):
            errors.append(f"catalog type must end with Rule: {rule_type.__name__}")
    sample = BudgetBelowRule(
        threshold=0.5,
        profile=LLMProfile(provider=LLMProvider.OPENAI, model="gpt-4o-mini"),
    )
    if sample.rule_id in seen_ids:
        errors.append("duplicate sample rule_id in catalog validation")
    return errors


def _validate_lab_reference_host() -> list[str]:
    errors: list[str] = []
    settings = LabApplicationSettings()
    env = build_lab_environment_profile(settings)
    routing = env.llm_routing_profile
    if routing is None:
        errors.append("lab host must configure llm_routing_profile")
        return errors
    errors.extend(LLMRoutingEvaluator.validate_rules(routing.rules))
    for rule in routing.rules:
        if type(rule) not in BUILTIN_ROUTING_RULE_TYPES:
            errors.append(
                f"lab host rule {rule.rule_id} must use predefined class, got {type(rule).__name__}",
            )
    for allowed in routing.allowed_profiles:
        if not is_profile_allowed(allowed, routing.allowed_profiles):
            errors.append(f"lab allowlist invalid for {allowed.model}")
    return errors


def main() -> int:
    errors = _validate_catalog_exports()
    errors.extend(_validate_lab_reference_host())

    profile = _synthetic_routing_profile()
    errors.extend(LLMRoutingEvaluator.validate_rules(profile.rules))

    for allowed in profile.allowed_profiles:
        if not is_profile_allowed(allowed, profile.allowed_profiles):
            errors.append(f"allowlist invalid for {allowed.model}")

    evaluation = LLMRoutingEvaluator().evaluate(
        profile,
        RoutingContext(budget_remaining_ratio=0.1),
    )
    if not is_profile_allowed(evaluation.selected_profile, profile.allowed_profiles):
        errors.append("evaluation selected profile outside allowlist")

    product = ApplicationEnvironmentProfile.product_defaults()
    if not product.adaptive_profile.live_model_routing_enabled:
        errors.append("product_defaults must enable live_model_routing for AHI gate parity")

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print("OK: llm routing catalog + reference host + evaluator")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
