# © Artur Czarnecki. All rights reserved.

"""Admitted compensation tool side-effect execution (Platform Execution Unification U2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from intergrax.knowledge.contracts.validation import JsonObject


@dataclass(frozen=True, slots=True)
class CompensationSideEffectInput:
    """Durable compensation job payload admitted through ExecutionRuntime."""

    tenant_id: str
    run_id: str
    agent_id: str
    step_index: int
    task_id: str
    compensation_tool_id: str
    args: JsonObject
    idempotency_key: str
    original_side_effect_id: str


@dataclass(frozen=True, slots=True)
class CompensationSideEffectInvokeResult:
    status: Literal["success", "failed", "denied"]
    error: str | None = None


@runtime_checkable
class CompensationToolInvokeSession(Protocol):
    """Tool invoke bound to active execution identity (implemented in ACP persistence)."""

    async def invoke(
        self,
        *,
        tenant_id: str,
        run_id: str,
        task_id: str,
        agent_id: str,
        tool_id: str,
        args: JsonObject,
        idempotency_key: str,
    ) -> CompensationSideEffectInvokeResult:
        ...


@runtime_checkable
class CompensationSideEffectExecutionPort(Protocol):
    """Canonical ExecutionRuntime admission for one compensation side effect."""

    async def execute(self, work: CompensationSideEffectInput) -> CompensationSideEffectInvokeResult:
        ...
