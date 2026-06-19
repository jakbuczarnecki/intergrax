# © Artur Czarnecki. All rights reserved.

"""Dynamic per-step LLM routing wrapper (M-LLM-X.9.7)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from intergrax.agents.authoring.llm_router import LlmStepResult, StepLLMRouter
from intergrax.contracts.agent_run_trace import LlmCallRecord
from intergrax.llm_adapters.routing import (
    LLMRoutingEvaluator,
    LLMRoutingProfile,
    RoutingContext,
    RoutingHint,
)
from intergrax.llm_adapters.routing.builtin_rules import cheapest_allowed_model_hint


@dataclass
class DynamicLLMRouter:
    """Re-evaluates routing rules before each LLM call."""

    _inner: StepLLMRouter
    _routing_profile: LLMRoutingProfile
    _context_provider: Callable[[], RoutingContext]
    _evaluator: LLMRoutingEvaluator = field(default_factory=LLMRoutingEvaluator)

    def list_allowed_models(self) -> list[str]:
        return self._inner.list_allowed_models()

    @property
    def effective_model(self) -> str:
        return self._inner.effective_model

    def resolve_model(self, model_hint: str | None) -> str:
        return self._inner.resolve_model(self._resolve_hint(model_hint))

    def drain_pending_calls(self) -> list[LlmCallRecord]:
        return self._inner.drain_pending_calls()

    def _resolve_hint(self, model_hint: str | None) -> str | None:
        evaluation = self._evaluator.evaluate(
            self._routing_profile,
            self._context_provider(),
        )
        if evaluation.target.hint is RoutingHint.CHEAPEST:
            return cheapest_allowed_model_hint(self._inner.list_allowed_models())
        if evaluation.selected_profile.model:
            allowed = self._inner.list_allowed_models()
            model = evaluation.selected_profile.model
            if model in allowed:
                return model
        return model_hint

    async def complete(self, prompt: str, *, model_hint: str | None = None) -> LlmStepResult:
        return await self._inner.complete(prompt, model_hint=self._resolve_hint(model_hint))


def wrap_dynamic_llm_router(
    router: StepLLMRouter,
    *,
    routing_profile: LLMRoutingProfile,
    context_provider: Callable[[], RoutingContext],
) -> DynamicLLMRouter:
    return DynamicLLMRouter(
        _inner=router,
        _routing_profile=routing_profile,
        _context_provider=context_provider,
    )
