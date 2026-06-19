# © Artur Czarnecki. All rights reserved.

"""Live routing re-evaluation wrapper (M-LLM-X.11.1 · M-LLM-X.12.2 Tier-3)."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any, Protocol

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.structured_result import LLMStructuredResult
from intergrax.llm_adapters.contracts.stream_event import LLMStreamEvent
from intergrax.llm_adapters.routing.contracts import RoutingContext, RoutingEvaluation
from intergrax.llm_adapters.routing.evaluator import (
    AllowlistViolationError,
    LLMRoutingEvaluator,
    profile_identity,
)

RoutingEvaluationObserver = Callable[[RoutingEvaluation], None]
AllowlistViolationObserver = Callable[[AllowlistViolationError, RoutingContext], None]
RoutingContextProvider = Callable[[], RoutingContext]
RoutingAdapterFactory = Callable[[RoutingEvaluation, RoutingContext], LLMAdapter]
InnerSwappedObserver = Callable[[LLMAdapter], None]


class RoutingProfileSource(Protocol):
  llm_routing_profile: Any


class RoutingEvaluatingLLMAdapter(LLMAdapter):
    """Re-evaluates ``LLMRoutingProfile`` before each LLM call and swaps inner adapter."""

    def __init__(
        self,
        *,
        env: ApplicationEnvironmentProfile,
        inner: LLMAdapter,
        context_provider: RoutingContextProvider,
        adapter_factory: RoutingAdapterFactory | None = None,
        on_evaluated: RoutingEvaluationObserver | None = None,
        on_allowlist_violation: AllowlistViolationObserver | None = None,
        on_inner_swapped: InnerSwappedObserver | None = None,
        before_evaluate: Callable[[], None] | None = None,
    ) -> None:
        super().__init__()
        self._env = env
        self._inner = inner
        self._context_provider = context_provider
        if adapter_factory is None:
            def _default_factory(
                evaluation: RoutingEvaluation,
                context: RoutingContext,
            ) -> LLMAdapter:
                from intergrax.applications._shared.llm_resolver import (
                    create_adapter_for_routing_evaluation,
                )

                return create_adapter_for_routing_evaluation(env, evaluation, context)

            adapter_factory = _default_factory
        self._adapter_factory = adapter_factory
        self._on_evaluated = on_evaluated
        self._on_allowlist_violation = on_allowlist_violation
        self._on_inner_swapped = on_inner_swapped
        self._before_evaluate = before_evaluate
        self._evaluator = LLMRoutingEvaluator()
        self._cached_identity: str | None = None
        self._sync_identity_from_inner()

    @property
    def inner_adapter(self) -> LLMAdapter:
        return self._inner

    def set_context_provider(self, provider: RoutingContextProvider) -> None:
        self._context_provider = provider

    def set_on_evaluated(self, observer: RoutingEvaluationObserver | None) -> None:
        self._on_evaluated = observer

    def set_on_allowlist_violation(self, observer: AllowlistViolationObserver | None) -> None:
        self._on_allowlist_violation = observer

    def set_on_inner_swapped(self, observer: InnerSwappedObserver | None) -> None:
        self._on_inner_swapped = observer

    def set_before_evaluate(self, hook: Callable[[], None] | None) -> None:
        self._before_evaluate = hook

    @property
    def context_window_tokens(self) -> int:
        return self._inner.context_window_tokens

    def _sync_identity_from_inner(self) -> None:
        self.provider = self._inner.provider
        self.model = self._inner.model

    def _evaluation_cache_key(self, evaluation: RoutingEvaluation) -> str:
        hint = evaluation.policy_route_hint or ""
        return f"{profile_identity(evaluation.selected_profile)}:{hint}"

    def _inner_matches_evaluation(self, evaluation: RoutingEvaluation) -> bool:
        profile = evaluation.selected_profile

        def _provider_key(provider: object) -> str:
            value = getattr(provider, "value", provider)
            return str(value)

        return (
            _provider_key(profile.provider) == _provider_key(self._inner.provider)
            and (profile.model or "") == (self._inner.model or "")
        )

    def _swap_inner(self, evaluation: RoutingEvaluation, context: RoutingContext) -> None:
        self._inner = self._adapter_factory(evaluation, context)
        self._cached_identity = self._evaluation_cache_key(evaluation)
        self._sync_identity_from_inner()
        if self._on_inner_swapped is not None:
            self._on_inner_swapped(self._inner)

    def _refresh_inner_adapter(self) -> None:
        routing_profile = self._env.llm_routing_profile
        if routing_profile is None:
            return
        if self._before_evaluate is not None:
            self._before_evaluate()
        context = self._context_provider()
        try:
            evaluation = self._evaluator.evaluate(routing_profile, context)
        except AllowlistViolationError as exc:
            if self._on_allowlist_violation is not None:
                self._on_allowlist_violation(exc, context)
            raise
        if self._on_evaluated is not None:
            self._on_evaluated(evaluation)
        cache_key = self._evaluation_cache_key(evaluation)
        if cache_key == self._cached_identity:
            return
        if self._cached_identity is None and self._inner_matches_evaluation(evaluation):
            self._cached_identity = cache_key
            return
        self._swap_inner(evaluation, context)

    def generate_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        run_id: str | None = None,
    ) -> LLMAdapterResponse:
        self._refresh_inner_adapter()
        return self._inner.generate_messages(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            run_id=run_id,
        )

    def stream_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        run_id: str | None = None,
    ) -> Iterable[LLMStreamEvent]:
        self._refresh_inner_adapter()
        return self._inner.stream_messages(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            run_id=run_id,
        )

    def generate_with_tools(
        self,
        messages: Sequence[ChatMessage],
        tools_schema: list[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        run_id: str | None = None,
    ) -> LLMAdapterResponse:
        self._refresh_inner_adapter()
        return self._inner.generate_with_tools(
            messages,
            tools_schema,
            temperature=temperature,
            max_tokens=max_tokens,
            tool_choice=tool_choice,
            run_id=run_id,
        )

    def generate_structured(
        self,
        messages: Sequence[ChatMessage],
        output_model: type,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        run_id: str | None = None,
    ) -> LLMStructuredResult[Any]:
        self._refresh_inner_adapter()
        return self._inner.generate_structured(
            messages,
            output_model,
            temperature=temperature,
            max_tokens=max_tokens,
            run_id=run_id,
        )

    def supports_streaming(self) -> bool:
        return self._inner.supports_streaming()

    def supports_structured_output(self) -> bool:
        return self._inner.supports_structured_output()

    def supports_tools(self) -> bool:
        return self._inner.supports_tools()

    def supports_vision(self) -> bool:
        return self._inner.supports_vision()

    def supports_audio_input(self) -> bool:
        return self._inner.supports_audio_input()

    def supports_audio_output(self) -> bool:
        return self._inner.supports_audio_output()


def wrap_routing_evaluating_adapter(
    adapter: LLMAdapter,
    env: ApplicationEnvironmentProfile,
    *,
    context_provider: RoutingContextProvider,
    adapter_factory: RoutingAdapterFactory,
    on_evaluated: RoutingEvaluationObserver | None = None,
    on_allowlist_violation: AllowlistViolationObserver | None = None,
    on_inner_swapped: InnerSwappedObserver | None = None,
    before_evaluate: Callable[[], None] | None = None,
) -> LLMAdapter:
    if env.llm_routing_profile is None or isinstance(adapter, RoutingEvaluatingLLMAdapter):
        return adapter
    return RoutingEvaluatingLLMAdapter(
        env=env,
        inner=adapter,
        context_provider=context_provider,
        adapter_factory=adapter_factory,
        on_evaluated=on_evaluated,
        on_allowlist_violation=on_allowlist_violation,
        on_inner_swapped=on_inner_swapped,
        before_evaluate=before_evaluate,
    )
