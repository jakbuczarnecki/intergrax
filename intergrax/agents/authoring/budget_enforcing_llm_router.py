# © Artur Czarnecki. All rights reserved.

"""Budget-aware wrapper for :class:`StepLLMRouter` (§25.5 · ACP-TOK-2)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from intergrax.agents.acp_budget_enforcement_bridge import AcpBudgetExceededError
from intergrax.agents.authoring.llm_router import LlmStepResult, StepLLMRouter
from intergrax.contracts.acp_budget_enforcement import evaluate_hard_budget_violation
from intergrax.contracts.acp_state import AcpInvocationUsageView
from intergrax.contracts.agent_budget import ResolvedBudgetLimits
from intergrax.contracts.agent_run_trace import LlmCallRecord


@dataclass
class BudgetEnforcingLLMRouter:
    """Delegates to :class:`StepLLMRouter` and enforces hard token caps pre-LLM."""

    _inner: StepLLMRouter
    _limits: ResolvedBudgetLimits
    _usage_provider: Callable[[], AcpInvocationUsageView | None]
    _degrade_provider: Callable[[], bool] | None = None

    def list_allowed_models(self) -> list[str]:
        return self._inner.list_allowed_models()

    @property
    def effective_model(self) -> str:
        return self._inner.effective_model

    def resolve_model(self, model_hint: str | None) -> str:
        return self._inner.resolve_model(model_hint)

    def drain_pending_calls(self) -> list[LlmCallRecord]:
        return self._inner.drain_pending_calls()

    async def complete(self, prompt: str, *, model_hint: str | None = None) -> LlmStepResult:
        pending = sum(
            call.tokens_in + call.tokens_out for call in self._inner._pending_calls  # noqa: SLF001
        )
        violation = evaluate_hard_budget_violation(
            self._usage_provider(),
            self._limits,
            pending_agent_tokens=pending,
        )
        if violation is not None:
            raise AcpBudgetExceededError(violation)
        effective_hint = model_hint
        if self._degrade_provider is not None and self._degrade_provider():
            allowed = self._inner.list_allowed_models()
            if allowed:
                effective_hint = allowed[-1]
        return await self._inner.complete(prompt, model_hint=effective_hint)


def wrap_budget_enforcing_router(
    router: StepLLMRouter,
    *,
    limits: ResolvedBudgetLimits,
    usage_provider: Callable[[], AcpInvocationUsageView | None],
    degrade_provider: Callable[[], bool] | None = None,
) -> BudgetEnforcingLLMRouter:
    return BudgetEnforcingLLMRouter(
        _inner=router,
        _limits=limits,
        _usage_provider=usage_provider,
        _degrade_provider=degrade_provider,
    )
