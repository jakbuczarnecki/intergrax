# © Artur Czarnecki. All rights reserved.

"""LLM adapter precedence: agent factory > environment > platform (Phase H-APP.1.6)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.registry.model_router import ModelRouter, ModelRoutingDecision
from intergrax.llm_adapters.registry.profile import LLMProfile, llm_profile_from_env
from intergrax.llm_adapters.routing import LLMRoutingEvaluator, RoutingContext, RoutingEvaluation
from intergrax.llm_adapters.routing.context_bridge import build_routing_context_from_runtime
from intergrax.applications._shared.routing_evaluating_adapter import (
    RoutingContextProvider,
    wrap_routing_evaluating_adapter,
)
from intergrax.llm_adapters.routing.evaluator import AllowlistViolationError

_last_routing_evaluation: RoutingEvaluation | None = None


def consume_routing_evaluation() -> RoutingEvaluation | None:
    """Return and clear the last routing evaluation from resolver path."""
    global _last_routing_evaluation
    result = _last_routing_evaluation
    _last_routing_evaluation = None
    return result


def resolve_llm_profile(
    env: ApplicationEnvironmentProfile | None,
) -> LLMProfile:
    """Resolve declarative LLM profile from environment or platform defaults."""
    if env is not None and env.llm_profile is not None:
        return env.llm_profile
    return llm_profile_from_env()


def _record_routing_evaluation(evaluation: RoutingEvaluation) -> None:
    global _last_routing_evaluation
    _last_routing_evaluation = evaluation


def evaluate_llm_routing(
    env: ApplicationEnvironmentProfile | None,
    *,
    routing_context: RoutingContext | None = None,
    on_evaluated: Callable[[RoutingEvaluation], None] | None = None,
) -> tuple[LLMProfile, str | None, str | None]:
    """
    Evaluate ``LLMRoutingProfile`` rules when configured (M-LLM-X.9.5).

    Returns ``(selected_profile, policy_route_hint, routing_reason)``.
    """
    global _last_routing_evaluation
    profile = resolve_llm_profile(env)
    hint: str | None = profile.routing_policy_hint
    reason: str | None = None
    _last_routing_evaluation = None
    if env is None or env.llm_routing_profile is None:
        return profile, hint, reason

    context = routing_context or RoutingContext()
    try:
        evaluation = LLMRoutingEvaluator().evaluate(env.llm_routing_profile, context)
    except AllowlistViolationError:
        raise
    _last_routing_evaluation = evaluation
    if on_evaluated is not None:
        on_evaluated(evaluation)
    profile = evaluation.selected_profile
    if evaluation.policy_route_hint is not None:
        hint = evaluation.policy_route_hint
    reason = evaluation.routing_reason
    return profile, hint, reason


def create_adapter_for_routing_evaluation(
    env: ApplicationEnvironmentProfile,
    evaluation: RoutingEvaluation,
    routing_context: RoutingContext | None = None,
) -> LLMAdapter:
    """Instantiate adapter for a routing evaluation (M-LLM-X.11.1 · M-LLM-X.12.5)."""
    from intergrax.applications._shared.llm_routing_wiring import resolve_live_model_routing_wiring

    profile = evaluation.selected_profile
    hint = evaluation.policy_route_hint or profile.routing_policy_hint
    context = routing_context or RoutingContext()
    wiring = resolve_live_model_routing_wiring(env, routing_context=context)
    if wiring.enabled and wiring.routing_decision is not None:
        ahi_hint = wiring.routing_decision.routing_reason.removeprefix("policy_hint_")
        if ahi_hint:
            hint = ahi_hint
    if profile.fallback_profiles or hint or profile.routing_policy_hint:
        return profile.create_adapter_with_failover(policy_route_hint=hint)
    return profile.create_adapter()


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


def _resolve_routing_context(
    *,
    routing_context: RoutingContext | None,
    routing_metadata: Mapping[str, Any] | None,
    tenant_id: str | None,
    agent_id: str | None,
    context_provider: RoutingContextProvider | None,
) -> RoutingContext:
    if context_provider is not None:
        return context_provider()
    if routing_context is not None:
        return routing_context
    if routing_metadata is not None or tenant_id is not None or agent_id is not None:
        return build_routing_context_from_runtime(
            tenant_id=tenant_id,
            agent_id=agent_id,
            metadata=routing_metadata,
        )
    return RoutingContext()


def _create_base_llm_adapter(
    env: ApplicationEnvironmentProfile | None,
    profile: LLMProfile,
    *,
    hint: str | None,
) -> LLMAdapter:
    if profile.fallback_profiles or hint or profile.routing_policy_hint:
        return profile.create_adapter_with_failover(policy_route_hint=hint)
    return profile.create_adapter()


def resolve_llm_adapter(
    env: ApplicationEnvironmentProfile | None,
    agent_override: LLMAdapter | None = None,
    *,
    policy_route_hint: str | None = None,
    routing_context: RoutingContext | None = None,
    routing_metadata: dict[str, Any] | None = None,
    tenant_id: str | None = None,
    agent_id: str | None = None,
    context_provider: RoutingContextProvider | None = None,
) -> LLMAdapter:
    """
    Resolve LLM adapter with explicit precedence.

    1. ``agent_override`` when provided by Tier-2 factory
    2. ``env.llm_profile`` when set on environment (with failover chain when configured)
    3. Platform default from ``INTERGRAX_LLM_*`` env vars

    When ``llm_routing_profile`` is set, evaluates rules via ``LLMRoutingEvaluator``
    before adapter creation. Live AHI routing may override hints on product hosts.
    When ``context_provider`` is set, wraps with ``RoutingEvaluatingLLMAdapter`` (M-LLM-X.11).
    """
    if agent_override is not None:
        return agent_override

    def _provider() -> RoutingContext:
        return _resolve_routing_context(
            routing_context=routing_context,
            routing_metadata=routing_metadata,
            tenant_id=tenant_id,
            agent_id=agent_id,
            context_provider=context_provider,
        )

    context = _provider()
    profile, rule_hint, _reason = evaluate_llm_routing(env, routing_context=context)
    hint = policy_route_hint or rule_hint or profile.routing_policy_hint

    if env is not None:
        from intergrax.applications._shared.llm_routing_wiring import resolve_live_model_routing_wiring

        wiring = resolve_live_model_routing_wiring(env, routing_context=context)
        if wiring.enabled and wiring.routing_decision is not None:
            ahi_hint = wiring.routing_decision.routing_reason.removeprefix("policy_hint_")
            if ahi_hint:
                hint = ahi_hint

    adapter = _create_base_llm_adapter(env, profile, hint=hint)
    if env is not None and env.llm_routing_profile is not None:
        live_provider = context_provider or _provider
        return wrap_routing_evaluating_adapter(
            adapter,
            env,
            context_provider=live_provider,
            adapter_factory=lambda evaluation, ctx: create_adapter_for_routing_evaluation(
                env,
                evaluation,
                ctx,
            ),
            on_evaluated=_record_routing_evaluation,
        )
    return adapter


def resolve_environment_llm_adapter(
    env: ApplicationEnvironmentProfile,
    *,
    agent_override: LLMAdapter | None = None,
    tenant_id: str | None = None,
    agent_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> LLMAdapter:
    """Tier-3 helper — always builds routing context from available host fields (M-LLM-X.11.3)."""
    routing_context = build_routing_context_from_runtime(
        tenant_id=tenant_id,
        agent_id=agent_id,
        metadata=metadata,
    )
    return resolve_llm_adapter(
        env,
        agent_override=agent_override,
        routing_context=routing_context,
        tenant_id=tenant_id,
        agent_id=agent_id,
        routing_metadata=metadata,
    )
