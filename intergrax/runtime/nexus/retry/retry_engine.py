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
from intergrax.runtime.nexus.retry.retry_types import RetryDecision, RetryRecord

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
