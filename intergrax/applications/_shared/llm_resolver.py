# © Artur Czarnecki. All rights reserved.

"""LLM adapter precedence: agent factory > environment > platform (Phase H-APP.1.6).

Tier-3 façade over the canonical resolver ``intergrax.runtime.wiring.llm_resolver`` (decision
DP-16). Every public name is kept; each function delegates and passes the application environment
profile unchanged. The only Tier-3 addition is the live routing hint port, which binds the AHI live
routing wiring to the profile. This module creates no adapter and keeps no routing record.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from intergrax.applications._shared.routing_evaluating_adapter import (
    RoutingAdapterFactory,
    RoutingContextProvider,
    RoutingEvaluatingLLMAdapter,
    RoutingEvaluationObserver,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.llm_adapters.contracts.capabilities import LLMFeature
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.routing_evaluator import RoutingEvaluator
from intergrax.llm_adapters.registry.model_router import ModelRoutingDecision
from intergrax.llm_adapters.registry.profile import LLMProfile
from intergrax.llm_adapters.routing import RoutingContext, RoutingEvaluation
from intergrax.runtime.wiring import llm_resolver as _canonical
from intergrax.runtime.wiring.llm_resolver import LiveRoutingHintProvider, RoutingAdapterWrapper


def _live_routing_hint_provider(
    env: ApplicationEnvironmentProfile | None,
) -> LiveRoutingHintProvider | None:
    """Bind the AHI live routing wiring (product hosts) to ``env`` as a typed hint port."""
    if env is None:
        return None
    bound_env = env

    def _hint(context: RoutingContext) -> str | None:
        from intergrax.applications._shared.llm_routing_wiring import resolve_live_model_routing_wiring

        wiring = resolve_live_model_routing_wiring(bound_env, routing_context=context)
        if wiring.enabled and wiring.routing_decision is not None:
            ahi_hint = wiring.routing_decision.routing_reason.removeprefix("policy_hint_")
            if ahi_hint:
                return ahi_hint
        return None

    return _hint


def _routing_adapter_wrapper(env: ApplicationEnvironmentProfile | None) -> RoutingAdapterWrapper | None:
    """Keep the Tier-3 wrapper type (``RoutingEvaluatingLLMAdapter`` subclass) for application hosts."""
    if env is None:
        return None
    bound_env = env

    def _wrap(
        adapter: LLMAdapter,
        *,
        context_provider: RoutingContextProvider,
        adapter_factory: RoutingAdapterFactory,
        evaluator: RoutingEvaluator,
        on_evaluated: RoutingEvaluationObserver,
    ) -> LLMAdapter:
        return RoutingEvaluatingLLMAdapter(
            env=bound_env,
            inner=adapter,
            context_provider=context_provider,
            adapter_factory=adapter_factory,
            routing_evaluator=evaluator,
            on_evaluated=on_evaluated,
        )

    return _wrap


def consume_routing_evaluation(
    adapter: LLMAdapter,
    *,
    tenant_id: str | None,
) -> RoutingEvaluation | None:
    """Return and clear the routing evaluation recorded for ``adapter`` and ``tenant_id``."""
    return _canonical.consume_routing_evaluation(adapter, tenant_id=tenant_id)


def resolve_optional_llm_profile(
    env: ApplicationEnvironmentProfile | None,
) -> LLMProfile | None:
    """Resolve declarative LLM profile when explicitly selected on host or in env."""
    return _canonical.resolve_optional_llm_profile(env)


def resolve_llm_profile(
    env: ApplicationEnvironmentProfile | None,
) -> LLMProfile:
    """Resolve declarative LLM profile or fail closed when no provider is selected."""
    return _canonical.resolve_llm_profile(env)


def evaluate_llm_routing(
    env: ApplicationEnvironmentProfile | None,
    *,
    routing_context: RoutingContext | None = None,
    routing_evaluator: RoutingEvaluator | None = None,
    on_evaluated: Callable[[RoutingEvaluation], None] | None = None,
) -> tuple[LLMProfile, str | None, str | None]:
    """Evaluate ``LLMRoutingProfile`` rules when configured (M-LLM-X.9.5)."""
    return _canonical.evaluate_llm_routing(
        env,
        routing_context=routing_context,
        routing_evaluator=routing_evaluator,
        on_evaluated=on_evaluated,
    )


def create_adapter_for_routing_evaluation(
    env: ApplicationEnvironmentProfile,
    evaluation: RoutingEvaluation,
    routing_context: RoutingContext | None = None,
) -> LLMAdapter:
    """Instantiate adapter for a routing evaluation (M-LLM-X.11.1 · M-LLM-X.12.5)."""
    return _canonical.create_adapter_for_routing_evaluation(
        env,
        evaluation,
        routing_context,
        live_routing_hint_provider=_live_routing_hint_provider(env),
    )


def resolve_runtime_llm_profile(
    env: ApplicationEnvironmentProfile | None,
    *,
    policy_route_hint: str | None = None,
    routing_context: RoutingContext | None = None,
) -> LLMProfile:
    """Resolve effective profile after optional routing hints (M-LLM-X.5.2)."""
    return _canonical.resolve_runtime_llm_profile(
        env,
        policy_route_hint=policy_route_hint,
        routing_context=routing_context,
    )


def resolve_llm_routing_decision(
    env: ApplicationEnvironmentProfile | None,
    *,
    policy_route_hint: str | None = None,
    routing_context: RoutingContext | None = None,
) -> ModelRoutingDecision:
    """Expose routing decision for AHI / observability wiring."""
    return _canonical.resolve_llm_routing_decision(
        env,
        policy_route_hint=policy_route_hint,
        routing_context=routing_context,
    )


def resolve_optional_llm_adapter(
    env: ApplicationEnvironmentProfile | None,
    agent_override: LLMAdapter | None = None,
    *,
    policy_route_hint: str | None = None,
    routing_context: RoutingContext | None = None,
    routing_metadata: dict[str, Any] | None = None,
    tenant_id: str | None = None,
    agent_id: str | None = None,
    context_provider: RoutingContextProvider | None = None,
    routing_evaluator: RoutingEvaluator | None = None,
) -> LLMAdapter | None:
    """Resolve LLM adapter only when a provider is explicitly selected."""
    return _canonical.resolve_optional_llm_adapter(
        env,
        agent_override,
        policy_route_hint=policy_route_hint,
        routing_context=routing_context,
        routing_metadata=routing_metadata,
        tenant_id=tenant_id,
        agent_id=agent_id,
        context_provider=context_provider,
        routing_evaluator=routing_evaluator,
        live_routing_hint_provider=_live_routing_hint_provider(env),
        routing_adapter_wrapper=_routing_adapter_wrapper(env),
    )


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
    routing_evaluator: RoutingEvaluator | None = None,
    required_llm_features: Sequence[LLMFeature | str] = (),
) -> LLMAdapter:
    """Resolve LLM adapter with explicit precedence (delegates to the canonical resolver).

    1. ``agent_override`` when provided by Tier-2 factory
    2. ``env.llm_profile`` when set on environment (with failover chain when configured)
    3. Platform selection from ``INTERGRAX_LLM_*`` env vars when explicitly set

    Live AHI routing may override hints on product hosts through the live routing hint port.
    """
    return _canonical.resolve_llm_adapter(
        env,
        agent_override,
        policy_route_hint=policy_route_hint,
        routing_context=routing_context,
        routing_metadata=routing_metadata,
        tenant_id=tenant_id,
        agent_id=agent_id,
        context_provider=context_provider,
        routing_evaluator=routing_evaluator,
        required_llm_features=required_llm_features,
        live_routing_hint_provider=_live_routing_hint_provider(env),
        routing_adapter_wrapper=_routing_adapter_wrapper(env),
    )


def resolve_optional_environment_llm_adapter(
    env: ApplicationEnvironmentProfile,
    *,
    agent_override: LLMAdapter | None = None,
    tenant_id: str | None = None,
    agent_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    routing_evaluator: RoutingEvaluator | None = None,
) -> LLMAdapter | None:
    """Tier-3 helper — resolve adapter only when provider selection is explicit."""
    return _canonical.resolve_optional_environment_llm_adapter(
        env,
        agent_override=agent_override,
        tenant_id=tenant_id,
        agent_id=agent_id,
        metadata=metadata,
        routing_evaluator=routing_evaluator,
        live_routing_hint_provider=_live_routing_hint_provider(env),
        routing_adapter_wrapper=_routing_adapter_wrapper(env),
    )


def resolve_environment_llm_adapter(
    env: ApplicationEnvironmentProfile,
    *,
    agent_override: LLMAdapter | None = None,
    tenant_id: str | None = None,
    agent_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    routing_evaluator: RoutingEvaluator | None = None,
) -> LLMAdapter:
    """Tier-3 helper — always builds routing context from available host fields (M-LLM-X.11.3)."""
    return _canonical.resolve_environment_llm_adapter(
        env,
        agent_override=agent_override,
        tenant_id=tenant_id,
        agent_id=agent_id,
        metadata=metadata,
        routing_evaluator=routing_evaluator,
        live_routing_hint_provider=_live_routing_hint_provider(env),
        routing_adapter_wrapper=_routing_adapter_wrapper(env),
    )
