# © Artur Czarnecki. All rights reserved.

"""Canonical host LLM adapter resolution (decision DP-16).

This is the one resolver of the platform. The Tier-3 module
``intergrax.applications._shared.llm_resolver`` keeps its public names and only delegates here.
The resolver accepts any environment profile that carries ``llm_profile`` and
``llm_routing_profile`` (``LLMResolutionSource``), so Tier-3 hosts pass their profile unchanged.

The routing evaluation made while resolving an adapter is recorded per resolved adapter object
and per tenant of the routing context (no module-wide record): a consumer receives only the
evaluation that belongs to its adapter and tenant.
"""

from __future__ import annotations

import threading
import weakref
from collections.abc import Callable, Mapping, Sequence
from contextvars import ContextVar
from typing import Any, Protocol

from intergrax.llm_adapters.contracts.capabilities import LLMFeature
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.routing_evaluator import RoutingEvaluator
from intergrax.llm_adapters.contracts.routing_profile import LLMRoutingProfile
from intergrax.llm_adapters.registry.model_router import ModelRouter, ModelRoutingDecision
from intergrax.llm_adapters.registry.profile import (
    LLMProfile,
    create_adapter,
    create_adapter_with_failover,
    llm_profile_from_env,
)
from intergrax.llm_adapters.registry.registration_contract import LLMProviderNotConfiguredError
from intergrax.llm_adapters.routing import RoutingContext, RoutingEvaluation
from intergrax.llm_adapters.routing.composition import resolve_routing_evaluator
from intergrax.llm_adapters.routing.context_bridge import build_routing_context_from_runtime
from intergrax.llm_adapters.routing.contracts import AllowlistViolationError
from intergrax.llm_adapters.routing.evaluating_adapter import (
    RoutingAdapterFactory,
    RoutingContextProvider,
    RoutingEvaluatingLLMAdapter,
)
from intergrax.llm_adapters.routing.evaluating_hooks import RoutingEvaluationObserver

_LLM_PROVIDER_NOT_CONFIGURED_MESSAGE = (
    "LLM provider not explicitly configured: set the environment profile llm_profile "
    "or INTERGRAX_LLM_PROVIDER before materializing an adapter."
)

LiveRoutingHintProvider = Callable[[RoutingContext], str | None]
"""Typed port: returns a live routing hint for a routing context (for example AHI tuning), or ``None``."""


class RoutingAdapterWrapper(Protocol):
    """Typed port: wraps a resolved adapter for live routing re-evaluation.

    The canonical default builds the platform ``RoutingEvaluatingLLMAdapter``. A host may supply a
    wrapper that returns a host-specific subclass of it (Tier-3 keeps its own subclass type).
    """

    def __call__(
        self,
        adapter: LLMAdapter,
        *,
        context_provider: RoutingContextProvider,
        adapter_factory: RoutingAdapterFactory,
        evaluator: RoutingEvaluator,
        on_evaluated: RoutingEvaluationObserver,
    ) -> LLMAdapter: ...


class LLMResolutionSource(Protocol):
    """Read-only environment view the resolver needs (satisfied by Tier-1 and Tier-3 profiles)."""

    @property
    def llm_profile(self) -> LLMProfile | None: ...

    @property
    def llm_routing_profile(self) -> LLMRoutingProfile | None: ...


# --- routing record: bound to the resolved adapter object, scoped by tenant -------------------

_records: weakref.WeakKeyDictionary[LLMAdapter, dict[str, RoutingEvaluation]] = weakref.WeakKeyDictionary()
_records_lock = threading.Lock()
_evaluating_tenant: ContextVar[str | None] = ContextVar("llm_resolver_evaluating_tenant", default=None)


def record_routing_evaluation(
    adapter: LLMAdapter,
    evaluation: RoutingEvaluation,
    *,
    tenant_id: str,
) -> None:
    """Record the latest routing evaluation of ``adapter`` for ``tenant_id``."""
    try:
        with _records_lock:
            _records.setdefault(adapter, {})[tenant_id] = evaluation
    except TypeError:
        # Adapter objects that cannot be weakly referenced or hashed carry no record
        # (the consumer then emits nothing, as when no evaluation exists).
        return


def consume_routing_evaluation(
    adapter: LLMAdapter,
    *,
    tenant_id: str | None,
) -> RoutingEvaluation | None:
    """Return and clear the evaluation recorded for this adapter and tenant, else ``None``."""
    if tenant_id is None:
        return None
    try:
        with _records_lock:
            tenants = _records.get(adapter)
            if tenants is None:
                return None
            evaluation = tenants.pop(tenant_id, None)
            if not tenants:
                _records.pop(adapter, None)
            return evaluation
    except TypeError:
        return None


# --- profile resolution -------------------------------------------------------------------------


def resolve_optional_llm_profile(env: LLMResolutionSource | None) -> LLMProfile | None:
    """Resolve declarative LLM profile when explicitly selected on host or in env."""
    if env is not None and env.llm_profile is not None:
        return env.llm_profile
    return llm_profile_from_env()


def resolve_llm_profile(env: LLMResolutionSource | None) -> LLMProfile:
    """Resolve declarative LLM profile or fail closed when no provider is selected."""
    profile = resolve_optional_llm_profile(env)
    if profile is None:
        raise LLMProviderNotConfiguredError(_LLM_PROVIDER_NOT_CONFIGURED_MESSAGE)
    return profile


def evaluate_llm_routing(
    env: LLMResolutionSource | None,
    *,
    routing_context: RoutingContext | None = None,
    routing_evaluator: RoutingEvaluator | None = None,
    on_evaluated: Callable[[RoutingEvaluation], None] | None = None,
) -> tuple[LLMProfile, str | None, str | None]:
    """Evaluate ``LLMRoutingProfile`` rules when configured.

    Returns ``(selected_profile, policy_route_hint, routing_reason)``.
    """
    profile = resolve_llm_profile(env)
    hint: str | None = profile.routing_policy_hint
    reason: str | None = None
    if env is None or env.llm_routing_profile is None:
        return profile, hint, reason

    context = routing_context or RoutingContext()
    try:
        evaluation = resolve_routing_evaluator(routing_evaluator).evaluate(
            env.llm_routing_profile,
            context,
        )
    except AllowlistViolationError:
        raise
    if on_evaluated is not None:
        on_evaluated(evaluation)
    profile = evaluation.selected_profile
    if evaluation.policy_route_hint is not None:
        hint = evaluation.policy_route_hint
    reason = evaluation.routing_reason
    return profile, hint, reason


def resolve_runtime_llm_profile(
    env: LLMResolutionSource | None,
    *,
    policy_route_hint: str | None = None,
    routing_context: RoutingContext | None = None,
) -> LLMProfile:
    """Resolve effective profile after optional routing hints (M-LLM-X.5.2).

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
    env: LLMResolutionSource | None,
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


# --- adapter creation ---------------------------------------------------------------------------


def _create_base_llm_adapter(
    env: LLMResolutionSource | None,
    profile: LLMProfile,
    *,
    hint: str | None,
) -> LLMAdapter:
    if profile.fallback_profiles or hint or profile.routing_policy_hint:
        from intergrax.llm_adapters.registry.failover_policy import routing_authorisation_context

        return create_adapter_with_failover(
            profile,
            policy_route_hint=hint,
            routing_authorisation=routing_authorisation_context(
                env.llm_routing_profile if env is not None else None,
            ),
        )
    return create_adapter(profile)


def create_adapter_for_routing_evaluation(
    env: LLMResolutionSource,
    evaluation: RoutingEvaluation,
    routing_context: RoutingContext | None = None,
    *,
    live_routing_hint_provider: LiveRoutingHintProvider | None = None,
) -> LLMAdapter:
    """Instantiate the adapter for a routing evaluation (M-LLM-X.11.1 · M-LLM-X.12.5)."""
    profile = evaluation.selected_profile
    hint = evaluation.policy_route_hint or profile.routing_policy_hint
    context = routing_context or RoutingContext()
    if live_routing_hint_provider is not None:
        live_hint = live_routing_hint_provider(context)
        if live_hint:
            hint = live_hint
    return _create_base_llm_adapter(env, profile, hint=hint)


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


def _resolve_llm_adapter_impl(
    env: LLMResolutionSource | None,
    *,
    policy_route_hint: str | None = None,
    routing_context: RoutingContext | None = None,
    routing_metadata: dict[str, Any] | None = None,
    tenant_id: str | None = None,
    agent_id: str | None = None,
    context_provider: RoutingContextProvider | None = None,
    routing_evaluator: RoutingEvaluator | None = None,
    live_routing_hint_provider: LiveRoutingHintProvider | None = None,
    routing_adapter_wrapper: RoutingAdapterWrapper | None = None,
) -> LLMAdapter:
    def _provider() -> RoutingContext:
        context = _resolve_routing_context(
            routing_context=routing_context,
            routing_metadata=routing_metadata,
            tenant_id=tenant_id,
            agent_id=agent_id,
            context_provider=context_provider,
        )
        _evaluating_tenant.set(context.tenant_id)
        return context

    context = _provider()
    resolution_tenant = context.tenant_id
    resolution_evaluations: list[RoutingEvaluation] = []
    profile, rule_hint, _reason = evaluate_llm_routing(
        env,
        routing_context=context,
        routing_evaluator=routing_evaluator,
        on_evaluated=resolution_evaluations.append,
    )
    hint = policy_route_hint or rule_hint or profile.routing_policy_hint

    if env is not None and live_routing_hint_provider is not None:
        live_hint = live_routing_hint_provider(context)
        if live_hint:
            hint = live_hint

    adapter = _create_base_llm_adapter(env, profile, hint=hint)
    if env is not None and env.llm_routing_profile is not None:
        routing_env = env
        live_provider = context_provider or _provider
        adapter_ref: list[LLMAdapter] = []

        def _record_live_evaluation(evaluation: RoutingEvaluation) -> None:
            if adapter_ref:
                evaluated_tenant = _evaluating_tenant.get()
                record_routing_evaluation(
                    adapter_ref[0],
                    evaluation,
                    tenant_id=evaluated_tenant if evaluated_tenant is not None else resolution_tenant,
                )

        def _tracked_context() -> RoutingContext:
            live_context = live_provider()
            _evaluating_tenant.set(live_context.tenant_id)
            return live_context

        if not isinstance(adapter, RoutingEvaluatingLLMAdapter):

            def adapter_factory(evaluation: RoutingEvaluation, ctx: RoutingContext) -> LLMAdapter:
                return create_adapter_for_routing_evaluation(
                    routing_env,
                    evaluation,
                    ctx,
                    live_routing_hint_provider=live_routing_hint_provider,
                )

            evaluator = resolve_routing_evaluator(routing_evaluator)
            if routing_adapter_wrapper is not None:
                adapter = routing_adapter_wrapper(
                    adapter,
                    context_provider=_tracked_context,
                    adapter_factory=adapter_factory,
                    evaluator=evaluator,
                    on_evaluated=_record_live_evaluation,
                )
            else:
                adapter = RoutingEvaluatingLLMAdapter(
                    profile_source=routing_env,
                    inner=adapter,
                    context_provider=_tracked_context,
                    adapter_factory=adapter_factory,
                    evaluator=evaluator,
                    on_evaluated=_record_live_evaluation,
                )
        adapter_ref.append(adapter)
        if resolution_evaluations:
            record_routing_evaluation(adapter, resolution_evaluations[-1], tenant_id=resolution_tenant)
    return adapter


def resolve_optional_llm_adapter(
    env: LLMResolutionSource | None,
    agent_override: LLMAdapter | None = None,
    *,
    policy_route_hint: str | None = None,
    routing_context: RoutingContext | None = None,
    routing_metadata: dict[str, Any] | None = None,
    tenant_id: str | None = None,
    agent_id: str | None = None,
    context_provider: RoutingContextProvider | None = None,
    routing_evaluator: RoutingEvaluator | None = None,
    live_routing_hint_provider: LiveRoutingHintProvider | None = None,
    routing_adapter_wrapper: RoutingAdapterWrapper | None = None,
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
        context_provider=context_provider,
        routing_evaluator=routing_evaluator,
        live_routing_hint_provider=live_routing_hint_provider,
        routing_adapter_wrapper=routing_adapter_wrapper,
    )


def _require_llm_features(adapter: LLMAdapter, features: Sequence[LLMFeature | str]) -> LLMAdapter:
    """Fail closed at composition when the bound adapter does not declare a feature the agent needs."""
    if features:
        from intergrax.llm_adapters.llm_provider_registry import LLMAdapterRegistry

        LLMAdapterRegistry.require_capabilities(adapter, *(LLMFeature(item) for item in features))
    return adapter


def resolve_llm_adapter(
    env: LLMResolutionSource | None,
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
    live_routing_hint_provider: LiveRoutingHintProvider | None = None,
    routing_adapter_wrapper: RoutingAdapterWrapper | None = None,
) -> LLMAdapter:
    """Resolve LLM adapter with explicit precedence.

    1. ``agent_override`` when provided by a Tier-2 factory
    2. ``env.llm_profile`` when set on the environment (with failover chain when configured)
    3. Platform selection from ``INTERGRAX_LLM_*`` env vars when explicitly set

    When ``llm_routing_profile`` is set, the rules are evaluated through the injected or default
    ``RoutingEvaluator`` before the adapter is created and the adapter is wrapped with
    ``RoutingEvaluatingLLMAdapter`` (live re-evaluation, M-LLM-X.11). ``live_routing_hint_provider``
    is the typed port for host-supplied live hints (for example AHI routing tuning).

    When ``required_llm_features`` is non-empty (``AgentContract.required_llm_features``), the adapter
    is checked with ``LLMAdapterRegistry.require_capabilities`` before it is returned and
    ``LLMCapabilityMismatchError`` is raised on a mismatch. Empty means unchanged behaviour.
    """
    if agent_override is not None:
        return _require_llm_features(agent_override, required_llm_features)
    if resolve_optional_llm_profile(env) is None:
        raise LLMProviderNotConfiguredError(_LLM_PROVIDER_NOT_CONFIGURED_MESSAGE)
    adapter = _resolve_llm_adapter_impl(
        env,
        policy_route_hint=policy_route_hint,
        routing_context=routing_context,
        routing_metadata=routing_metadata,
        tenant_id=tenant_id,
        agent_id=agent_id,
        context_provider=context_provider,
        routing_evaluator=routing_evaluator,
        live_routing_hint_provider=live_routing_hint_provider,
        routing_adapter_wrapper=routing_adapter_wrapper,
    )
    return _require_llm_features(adapter, required_llm_features)


def resolve_optional_environment_llm_adapter(
    env: LLMResolutionSource,
    *,
    agent_override: LLMAdapter | None = None,
    tenant_id: str | None = None,
    agent_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    routing_evaluator: RoutingEvaluator | None = None,
    live_routing_hint_provider: LiveRoutingHintProvider | None = None,
    routing_adapter_wrapper: RoutingAdapterWrapper | None = None,
) -> LLMAdapter | None:
    """Resolve adapter only when provider selection is explicit (routing context from host fields)."""
    routing_context = build_routing_context_from_runtime(
        tenant_id=tenant_id,
        agent_id=agent_id,
        metadata=metadata,
    )
    return resolve_optional_llm_adapter(
        env,
        agent_override=agent_override,
        routing_context=routing_context,
        tenant_id=tenant_id,
        agent_id=agent_id,
        routing_metadata=metadata,
        routing_evaluator=routing_evaluator,
        live_routing_hint_provider=live_routing_hint_provider,
        routing_adapter_wrapper=routing_adapter_wrapper,
    )


def resolve_environment_llm_adapter(
    env: LLMResolutionSource,
    *,
    agent_override: LLMAdapter | None = None,
    tenant_id: str | None = None,
    agent_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    routing_evaluator: RoutingEvaluator | None = None,
    live_routing_hint_provider: LiveRoutingHintProvider | None = None,
    routing_adapter_wrapper: RoutingAdapterWrapper | None = None,
) -> LLMAdapter:
    """Always builds the routing context from the available host fields (M-LLM-X.11.3)."""
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
        routing_evaluator=routing_evaluator,
        live_routing_hint_provider=live_routing_hint_provider,
        routing_adapter_wrapper=routing_adapter_wrapper,
    )
