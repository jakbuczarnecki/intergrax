# © Artur Czarnecki. All rights reserved.

"""Test doubles for admitted compensation side-effect execution."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from intergrax.agents.persistence.compensation_tool_invoke_session import (
    bound_compensation_tool_invoke_session,
)
from intergrax.agents.persistence.declarative_tool_executor import DeclarativeToolInvokeResult
from intergrax.contracts.compensation_side_effect_execution import (
    CompensationSideEffectExecutionPort,
)
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_bound_declarative_tool_invocation import (
    ExecutionBoundDeclarativeToolInvoker,
)
from intergrax.knowledge.contracts.validation import JsonObject
from intergrax.runtime.execution.compensation_side_effect import (
    build_runtime_compensation_side_effect_execution,
)


@dataclass
class RecordingExecutionBoundDeclarativeToolInvoker:
    """Execution-bound test invoker that records per-call identity on invoke."""

    _invoke_fn: Callable[..., Awaitable[DeclarativeToolInvokeResult]]
    last_tenant_id: str | None = field(default=None, init=False)
    last_run_id: str | None = field(default=None, init=False)
    last_task_id: str | None = field(default=None, init=False)
    last_agent_id: str | None = field(default=None, init=False)

    async def invoke(
        self,
        *,
        tenant_id: str,
        run_id: str,
        task_id: str,
        agent_id: str,
        tool_id: str,
        args: JsonObject,
        idempotency_key: str | None,
    ) -> DeclarativeToolInvokeResult:
        self.last_tenant_id = tenant_id
        self.last_run_id = run_id
        self.last_task_id = task_id
        self.last_agent_id = agent_id
        return await self._invoke_fn(
            tenant_id=tenant_id,
            run_id=run_id,
            task_id=task_id,
            agent_id=agent_id,
            tool_id=tool_id,
            args=dict(args),
            idempotency_key=idempotency_key,
        )


def build_test_admitted_compensation_side_effect_execution(
    invoker: ExecutionBoundDeclarativeToolInvoker,
    *,
    authority: ParentExecutionAuthority | None = None,
) -> CompensationSideEffectExecutionPort:
    return build_runtime_compensation_side_effect_execution(
        tool_session=bound_compensation_tool_invoke_session(invoker),
        authority=authority or ParentExecutionAuthority.unrestricted_root(),
    )
