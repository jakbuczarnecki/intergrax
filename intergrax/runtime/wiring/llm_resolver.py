# © Artur Czarnecki. All rights reserved.

"""LLM adapter resolution from neutral runtime environment profile."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from intergrax.contracts.runtime_environment import RuntimeEnvironmentProfile
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.registry.model_router import ModelRouter
from intergrax.llm_adapters.registry.profile import LLMProfile, llm_profile_from_env
from intergrax.llm_adapters.registry.registration_contract import LLMProviderNotConfiguredError
from intergrax.llm_adapters.routing import LLMRoutingEvaluator, RoutingContext, RoutingEvaluation
from intergrax.llm_adapters.routing.context_bridge import build_routing_context_from_runtime
from intergrax.llm_adapters.routing.evaluator import AllowlistViolationError

_last_routing_evaluation: RoutingEvaluation | None = None

_LLM_PROVIDER_NOT_CONFIGURED_MESSAGE = (
    "LLM provider not explicitly configured: set runtime profile llm_profile "
    "or INTERGRAX_LLM_PROVIDER before materializing an adapter."
)


def consume_routing_evaluation() -> RoutingEvaluation | None:
    """Return and clear the last routing evaluation from resolver path."""
    global _last_routing_evaluation
    result = _last_routing_evaluation
    _last_routing_evaluation = None
    return result


def resolve_optional_llm_profile(env: RuntimeEnvironmentProfile | None) -> LLMProfile | None:
    """Resolve declarative LLM profile when explicitly selected on host or in env."""
    if env is not None and env.llm_profile is not None:
        return env.llm_profile
    return llm_profile_from_env()


def resolve_llm_profile(env: RuntimeEnvironmentProfile | None) -> LLMProfile:
    """Resolve declarative LLM profile or fail closed when no provider is selected."""
    profile = resolve_optional_llm_profile(env)
    if profile is None:
        raise LLMProviderNotConfiguredError(_LLM_PROVIDER_NOT_CONFIGURED_MESSAGE)
    return profile


def evaluate_llm_routing(
    env: RuntimeEnvironmentProfile | None,
    *,
    routing_context: RoutingContext | None = None,
    on_evaluated: Callable[[RoutingEvaluation], None] | None = None,
) -> tuple[LLMProfile, str | None, str | None]:
    """Evaluate ``LLMRoutingProfile`` rules when configured."""
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


def _resolve_routing_context(
    *,
    routing_context: RoutingContext | None,
    routing_metadata: Mapping[str, Any] | None,
    tenant_id: str | None,
    agent_id: str | None,
) -> RoutingContext:
    if routing_context is not None:
        return routing_context
    if routing_metadata is not None or tenant_id is not None or agent_id is not None:
        return build_routing_context_from_runtime(
            tenant_id=tenant_id,
            agent_id=agent_id,
            metadata=routing_metadata,
        )
    return RoutingContext()


def _resolve_llm_adapter_impl(
    env: RuntimeEnvironmentProfile | None,
    *,
    policy_route_hint: str | None = None,
    routing_context: RoutingContext | None = None,
    routing_metadata: dict[str, Any] | None = None,
    tenant_id: str | None = None,
    agent_id: str | None = None,
) -> LLMAdapter:
    context = _resolve_routing_context(
        routing_context=routing_context,
        routing_metadata=routing_metadata,
        tenant_id=tenant_id,
        agent_id=agent_id,
    )
    profile, rule_hint, _reason = evaluate_llm_routing(env, routing_context=context)
    hint = policy_route_hint or rule_hint or profile.routing_policy_hint
    router = ModelRouter.from_profiles(
        profile,
        fallbacks=profile.fallback_profiles,
        policy_route_hint=hint,
    )
    selected = router.ordered_profiles()[0]
    if selected.fallback_profiles or hint or selected.routing_policy_hint:
        return selected.create_adapter_with_failover(policy_route_hint=hint)
    return selected.create_adapter()


def resolve_optional_llm_adapter(
    env: RuntimeEnvironmentProfile | None,
    agent_override: LLMAdapter | None = None,
    *,
    policy_route_hint: str | None = None,
    routing_context: RoutingContext | None = None,
    routing_metadata: dict[str, Any] | None = None,
    tenant_id: str | None = None,
    agent_id: str | None = None,
) -> LLMAdapter | None:
    """Resolve LLM adapter only when a provider is explicitly selected."""
    if agent_override is not None:
        return agent_override
    if resolve_optional_llm_profile(env) is None:
        return None
    return _resolve_llm_adapter_impl(
        env,
        policy_route_hint=policy_route_hint,
        routing_context=routing_context,
        routing_metadata=routing_metadata,
        tenant_id=tenant_id,
        agent_id=agent_id,
    )


def resolve_llm_adapter(
    env: RuntimeEnvironmentProfile | None,
    agent_override: LLMAdapter | None = None,
    *,
    policy_route_hint: str | None = None,
    routing_context: RoutingContext | None = None,
    routing_metadata: dict[str, Any] | None = None,
    tenant_id: str | None = None,
    agent_id: str | None = None,
) -> LLMAdapter:
    """Resolve LLM adapter from runtime environment profile."""
    if agent_override is not None:
        return agent_override
    if resolve_optional_llm_profile(env) is None:
        raise LLMProviderNotConfiguredError(_LLM_PROVIDER_NOT_CONFIGURED_MESSAGE)
    return _resolve_llm_adapter_impl(
        env,
        policy_route_hint=policy_route_hint,
        routing_context=routing_context,
        routing_metadata=routing_metadata,
        tenant_id=tenant_id,
        agent_id=agent_id,
    )
