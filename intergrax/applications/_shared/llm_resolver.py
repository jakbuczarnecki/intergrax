# © Artur Czarnecki. All rights reserved.

"""LLM adapter precedence: agent factory > environment > platform (Phase H-APP.1.6)."""

from __future__ import annotations

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.registry.model_router import ModelRouter, ModelRoutingDecision
from intergrax.llm_adapters.registry.profile import LLMProfile, llm_profile_from_env
from intergrax.llm_adapters.routing import LLMRoutingEvaluator, RoutingContext


def resolve_llm_profile(
    env: ApplicationEnvironmentProfile | None,
) -> LLMProfile:
    """Resolve declarative LLM profile from environment or platform defaults."""
    if env is not None and env.llm_profile is not None:
        return env.llm_profile
    return llm_profile_from_env()


def evaluate_llm_routing(
    env: ApplicationEnvironmentProfile | None,
    *,
    routing_context: RoutingContext | None = None,
) -> tuple[LLMProfile, str | None, str | None]:
    """
    Evaluate ``LLMRoutingProfile`` rules when configured (M-LLM-X.9.5).

    Returns ``(selected_profile, policy_route_hint, routing_reason)``.
    """
    profile = resolve_llm_profile(env)
    hint: str | None = profile.routing_policy_hint
    reason: str | None = None
    if env is None or env.llm_routing_profile is None:
        return profile, hint, reason

    context = routing_context or RoutingContext()
    evaluation = LLMRoutingEvaluator().evaluate(env.llm_routing_profile, context)
    profile = evaluation.selected_profile
    if evaluation.policy_route_hint is not None:
        hint = evaluation.policy_route_hint
    reason = evaluation.routing_reason
    return profile, hint, reason


def resolve_runtime_llm_profile(
    env: ApplicationEnvironmentProfile | None,
    *,
    policy_route_hint: str | None = None,
    routing_context: RoutingContext | None = None,
) -> LLMProfile:
    """
    Resolve effective profile after optional routing hints (M-LLM-X.5.2).

    Returns the first profile in routing order — use ``resolve_llm_adapter`` for
    instantiated adapters including failover chains.
    """
    profile, rule_hint, _reason = evaluate_llm_routing(env, routing_context=routing_context)
    router = ModelRouter.from_profiles(
        profile,
        fallbacks=profile.fallback_profiles,
        policy_route_hint=policy_route_hint or rule_hint or profile.routing_policy_hint,
    )
    return router.ordered_profiles()[0]


def resolve_llm_routing_decision(
    env: ApplicationEnvironmentProfile | None,
    *,
    policy_route_hint: str | None = None,
    routing_context: RoutingContext | None = None,
) -> ModelRoutingDecision:
    """Expose routing decision for AHI / observability wiring."""
    profile, rule_hint, _reason = evaluate_llm_routing(env, routing_context=routing_context)
    router = ModelRouter.from_profiles(
        profile,
        fallbacks=profile.fallback_profiles,
        policy_route_hint=policy_route_hint or rule_hint or profile.routing_policy_hint,
    )
    return router.resolve()


def resolve_llm_adapter(
    env: ApplicationEnvironmentProfile | None,
    agent_override: LLMAdapter | None = None,
    *,
    policy_route_hint: str | None = None,
    routing_context: RoutingContext | None = None,
) -> LLMAdapter:
    """
    Resolve LLM adapter with explicit precedence.

    1. ``agent_override`` when provided by Tier-2 factory
    2. ``env.llm_profile`` when set on environment (with failover chain when configured)
    3. Platform default from ``INTERGRAX_LLM_*`` env vars

    When ``llm_routing_profile`` is set, evaluates rules via ``LLMRoutingEvaluator``
    before adapter creation. Live AHI routing may override hints on product hosts.
    """
    if agent_override is not None:
        return agent_override

    profile, rule_hint, _reason = evaluate_llm_routing(env, routing_context=routing_context)
    hint = policy_route_hint or rule_hint or profile.routing_policy_hint

    if env is not None:
        from intergrax.applications._shared.llm_routing_wiring import resolve_live_model_routing_wiring

        wiring = resolve_live_model_routing_wiring(env, routing_context=routing_context)
        if wiring.enabled and wiring.routing_decision is not None:
            ahi_hint = wiring.routing_decision.routing_reason.removeprefix("policy_hint_")
            if ahi_hint:
                hint = ahi_hint

    if profile.fallback_profiles or hint or profile.routing_policy_hint:
        return profile.create_adapter_with_failover(policy_route_hint=hint)
    return profile.create_adapter()
