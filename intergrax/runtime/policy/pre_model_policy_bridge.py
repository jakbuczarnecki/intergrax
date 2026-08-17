# © Artur Czarnecki. All rights reserved.

"""PRE_MODEL policy evaluation immediately before provider/model invocation (G3B)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from intergrax.agents.authoring.llm_router import LlmStepResult, StepLLMRouter
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.contracts.runtime_policy_context import PreModelPhase, PreModelPolicyContext
from intergrax.runtime.policy.policy_engine import PolicyEngine


class PreModelPolicyBlockedError(RuntimeError):
    """Raised when PRE_MODEL policy denies model/provider invocation."""

    def __init__(self, decision: PolicyDecision) -> None:
        self.decision = decision
        super().__init__(decision.reason or "pre_model_policy_denied")


def evaluate_pre_model_policy(
    policy_engine: PolicyEngine,
    *,
    tenant_id: str,
    agent_id: str,
    message_count: int = 1,
    context: PreModelPolicyContext | None = None,
) -> PolicyDecision:
    return policy_engine.evaluate_pre_llm(
        tenant_id=tenant_id,
        agent_id=agent_id,
        message_count=message_count,
        context=context,
    )


@dataclass
class PolicyEnforcingLLMRouter:
    """Evaluate PRE_MODEL policy before delegating to the inner LLM router."""

    _inner: StepLLMRouter
    _policy_engine: PolicyEngine
    _tenant_id: str
    _agent_id: str
    _message_count_provider: Callable[[], int] | None = None

    def list_allowed_models(self) -> list[str]:
        return self._inner.list_allowed_models()

    @property
    def effective_model(self) -> str:
        return self._inner.effective_model

    def resolve_model(self, model_hint: str | None) -> str:
        return self._inner.resolve_model(model_hint)

    def drain_pending_calls(self):
        return self._inner.drain_pending_calls()

    async def complete(self, prompt: str, *, model_hint: str | None = None) -> LlmStepResult:
        model_id = self.resolve_model(model_hint)
        message_count = 1 if self._message_count_provider is None else self._message_count_provider()
        decision = evaluate_pre_model_policy(
            self._policy_engine,
            tenant_id=self._tenant_id,
            agent_id=self._agent_id,
            message_count=message_count,
            context=PreModelPolicyContext(
                phase=PreModelPhase.AGENT_STEP,
                model_id=model_id,
            ),
        )
        if decision.action is PolicyAction.DENY:
            raise PreModelPolicyBlockedError(decision)
        return await self._inner.complete(prompt, model_hint=model_hint)


def wrap_policy_enforcing_llm_router(
    router: StepLLMRouter,
    *,
    policy_engine: PolicyEngine,
    tenant_id: str,
    agent_id: str,
    message_count_provider: Callable[[], int] | None = None,
) -> PolicyEnforcingLLMRouter:
    return PolicyEnforcingLLMRouter(
        _inner=router,
        _policy_engine=policy_engine,
        _tenant_id=tenant_id,
        _agent_id=agent_id,
        _message_count_provider=message_count_provider,
    )
