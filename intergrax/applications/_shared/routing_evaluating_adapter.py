# © Artur Czarnecki. All rights reserved.

"""Tier-3 composition shim for live routing re-evaluation (M-LLM-X.11.1)."""

from __future__ import annotations

from collections.abc import Callable

from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
)
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.routing.evaluating_adapter import (
    RoutingAdapterFactory,
    RoutingContextProvider,
    RoutingEvaluatingLLMAdapter as PlatformRoutingEvaluatingLLMAdapter,
)
from intergrax.llm_adapters.routing.evaluating_hooks import (
    AllowlistViolationObserver,
    InnerSwappedObserver,
    RoutingEvaluationObserver,
)
from intergrax.llm_adapters.contracts.routing_evaluator import RoutingEvaluator
from intergrax.llm_adapters.routing.composition import resolve_routing_evaluator
from intergrax.llm_adapters.routing.profile_source import RoutingProfileSource

__all__ = [
    "AllowlistViolationObserver",
    "InnerSwappedObserver",
    "RoutingAdapterFactory",
    "RoutingContextProvider",
    "RoutingEvaluatingLLMAdapter",
    "RoutingEvaluationObserver",
    "RoutingProfileSource",
    "wrap_routing_evaluating_adapter",
]


class RoutingEvaluatingLLMAdapter(PlatformRoutingEvaluatingLLMAdapter):
    """Application-bound adapter with default inner factory from host wiring."""

    def __init__(
        self,
        *,
        env: ApplicationEnvironmentProfile,
        inner: LLMAdapter,
        context_provider: RoutingContextProvider,
        adapter_factory: RoutingAdapterFactory | None = None,
        routing_evaluator: RoutingEvaluator | None = None,
        on_evaluated: RoutingEvaluationObserver | None = None,
        on_allowlist_violation: AllowlistViolationObserver | None = None,
        on_inner_swapped: InnerSwappedObserver | None = None,
        before_evaluate: Callable[[], None] | None = None,
    ) -> None:
        if adapter_factory is None:

            def _default_factory(
                evaluation,
                context,
            ) -> LLMAdapter:
                from intergrax.applications._shared.llm_resolver import (
                    create_adapter_for_routing_evaluation,
                )

                return create_adapter_for_routing_evaluation(env, evaluation, context)

            adapter_factory = _default_factory
        super().__init__(
            profile_source=env,
            inner=inner,
            context_provider=context_provider,
            adapter_factory=adapter_factory,
            evaluator=resolve_routing_evaluator(routing_evaluator),
            on_evaluated=on_evaluated,
            on_allowlist_violation=on_allowlist_violation,
            on_inner_swapped=on_inner_swapped,
            before_evaluate=before_evaluate,
        )


def wrap_routing_evaluating_adapter(
    adapter: LLMAdapter,
    env: ApplicationEnvironmentProfile,
    *,
    context_provider: RoutingContextProvider,
    adapter_factory: RoutingAdapterFactory,
    routing_evaluator: RoutingEvaluator | None = None,
    on_evaluated: RoutingEvaluationObserver | None = None,
    on_allowlist_violation: AllowlistViolationObserver | None = None,
    on_inner_swapped: InnerSwappedObserver | None = None,
    before_evaluate: Callable[[], None] | None = None,
) -> LLMAdapter:
    if env.llm_routing_profile is None or isinstance(
        adapter, RoutingEvaluatingLLMAdapter
    ):
        return adapter
    return RoutingEvaluatingLLMAdapter(
        env=env,
        inner=adapter,
        context_provider=context_provider,
        adapter_factory=adapter_factory,
        routing_evaluator=routing_evaluator,
        on_evaluated=on_evaluated,
        on_allowlist_violation=on_allowlist_violation,
        on_inner_swapped=on_inner_swapped,
        before_evaluate=before_evaluate,
    )
