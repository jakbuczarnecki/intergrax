# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Awaitable, Callable, List, Optional

from intergrax.agents.agent_contract import Agent
from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.contracts.validation import ValidationResult
from intergrax.runtime.hooks.governance_hooks import hook_context_for_task, run_hook_pair
from intergrax.runtime.hooks.hook_point import HookPoint
from intergrax.runtime.middleware.pipeline import MiddlewarePipeline
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.contracts.resilience_policy import FailureClass, FailureResponse, ResiliencePolicy
from intergrax.runtime.nexus.retry.retry_types import RetryDecision, RetryRecord
from intergrax.runtime.resilience.policy_resolver import resolve_failure_action

ExecuteFn = Callable[[Agent], Awaitable[AgentExecutionResult]]


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 1
    retry_alternate_agent: bool = True


class RetryEngine:
    """Controlled retry with optional alternate agent (§31, Phase B.5)."""

    def __init__(
        self,
        registry: AgentRegistry,
        *,
        policy: Optional[RetryPolicy] = None,
        middleware: Optional[MiddlewarePipeline] = None,
    ) -> None:
        self._registry = registry
        self._policy = policy or RetryPolicy()
        self._middleware = middleware

    def decide(
        self,
        task: Task,
        *,
        agent_id: str,
        validation: ValidationResult,
        attempt: int,
    ) -> RetryDecision:
        if validation.valid:
            return RetryDecision(should_retry=False)
        if attempt >= self._policy.max_retries:
            return RetryDecision(should_retry=False, reason="max_retries_exceeded")

        if not self._policy.retry_alternate_agent:
            return RetryDecision(should_retry=False, reason="retry_disabled")

        alternate = self._find_alternate_agent(task, excluded={agent_id})
        resilience_policy = _resilience_policy_from_task(task)
        if resilience_policy is not None:
            return _retry_decision_from_resilience_policy(
                policy=resilience_policy,
                attempt=attempt,
                alternate_agent_id=alternate,
            )
        if alternate is None:
            return RetryDecision(should_retry=False, reason="no_alternate_agent")

        return RetryDecision(
            should_retry=True,
            reason="validation_failed",
            alternate_agent_id=alternate,
        )

    async def execute_with_retry(
        self,
        task: Task,
        initial_agent: Agent,
        execute_fn: ExecuteFn,
        *,
        validate_fn: Callable[[AgentExecutionResult, Agent], ValidationResult],
        on_retry: Optional[Callable[[RetryRecord], None]] = None,
    ) -> tuple[AgentExecutionResult, List[RetryRecord], ValidationResult]:
        agent = initial_agent
        records: List[RetryRecord] = []
        attempt = 0
        validation = ValidationResult(valid=False, errors=["not executed"])

        while True:
            execution = await execute_fn(agent)
            if execution.status == AgentExecutionStatus.NEEDS_INPUT:
                return (
                    execution,
                    records,
                    ValidationResult(valid=False, errors=["awaiting_human_input"]),
                )
            validation = validate_fn(execution, agent)
            if validation.valid:
                return execution, records, validation

            decision = self.decide(
                task,
                agent_id=agent.get_contract().id,
                validation=validation,
                attempt=attempt,
            )
            if not decision.should_retry or not decision.alternate_agent_id:
                return execution, records, validation

            ctx = hook_context_for_task(
                task_id=task.task_id,
                run_id=task.task_id,
                agent_id=agent.get_contract().id,
                phase=ExecutionPhase.RETRY_HANDLING,
                runtime_state={"reason": decision.reason, "attempt": attempt + 1},
            )
            await run_hook_pair(
                self._middleware,
                HookPoint.BEFORE_RETRY,
                HookPoint.AFTER_RETRY,
                ctx,
            )

            attempt += 1
            record = RetryRecord(
                attempt=attempt,
                agent_id=agent.get_contract().id,
                reason=decision.reason,
                alternate_agent_id=decision.alternate_agent_id,
            )
            records.append(record)
            if on_retry is not None:
                on_retry(record)

            agent = self._registry.get(decision.alternate_agent_id)

    def _find_alternate_agent(
        self,
        task: Task,
        *,
        excluded: set[str],
    ) -> Optional[str]:
        capability = task.context.capability
        if capability:
            for agent in self._registry.find_by_capability(capability):
                aid = agent.get_contract().id
                if aid not in excluded:
                    return aid

        for aid in self._registry.list_agent_ids():
            if aid not in excluded:
                return aid
        return None


def _retry_decision_from_resilience_policy(
    *,
    policy: ResiliencePolicy,
    attempt: int,
    alternate_agent_id: str | None,
) -> RetryDecision:
    resolution = resolve_failure_action(
        FailureClass.QUALITY_ERROR,
        policy=policy,
        attempt=attempt,
    )
    if resolution.response is FailureResponse.RETRY_ALTERNATE and alternate_agent_id:
        return RetryDecision(
            should_retry=True,
            reason=resolution.reason,
            alternate_agent_id=alternate_agent_id,
        )
    if resolution.response is FailureResponse.RETRY:
        return RetryDecision(should_retry=True, reason=resolution.reason)
    return RetryDecision(should_retry=False, reason=resolution.reason)


def _resilience_policy_from_task(task: Task) -> ResiliencePolicy | None:
    raw = task.metadata.get("resilience_policy.v1")
    if isinstance(raw, dict):
        return ResiliencePolicy.model_validate(raw)
    return None
