# © Artur Czarnecki. All rights reserved.

"""Declarative ``requested_actions`` execution with ledger commit (ACP-PROD-2)."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from intergrax.agents.persistence.side_effect_ledger import SideEffectLedger

DeclarativeToolStatus = Literal[
    "success",
    "failed",
    "denied",
    "replay_skipped",
    "skipped_no_invoker",
]


@dataclass(frozen=True)
class DeclarativeToolInvokeResult:
    status: Literal["success", "failed", "denied"]
    output: dict[str, Any] | None = None
    external_ref: str | None = None
    error: str | None = None
    duration_ms: int = 0


@runtime_checkable
class DeclarativeToolInvoker(Protocol):
    async def invoke(
        self,
        *,
        tool_id: str,
        args: dict[str, Any],
        idempotency_key: str | None,
    ) -> DeclarativeToolInvokeResult:
        ...


@dataclass(frozen=True)
class DeclarativeActionExecution:
    tool_id: str
    status: DeclarativeToolStatus
    idempotency_key: str | None = None
    output: dict[str, Any] | None = None
    external_ref: str | None = None
    error: str | None = None
    replay_skipped: bool = False
    duration_ms: int = 0


@dataclass
class DeclarativeExecutionResult:
    results: list[DeclarativeActionExecution] = field(default_factory=list)

    @property
    def failed_tool_id(self) -> str | None:
        for item in self.results:
            if item.status in ("failed", "denied"):
                return item.tool_id
        return None

    @property
    def replay_skipped_count(self) -> int:
        return sum(1 for item in self.results if item.replay_skipped)


def _ledger_external_ref(ledger: SideEffectLedger, idempotency_key: str) -> str | None:
    for record in ledger.records():
        if record.idempotency_key == idempotency_key:
            return record.external_ref
    return None


async def execute_declarative_actions(
    *,
    actions: list[dict[str, Any]],
    ledger: SideEffectLedger | None,
    invoker: DeclarativeToolInvoker | None,
) -> DeclarativeExecutionResult:
    """
    Execute validated declarative tool actions.

  * Committed ledger keys are replay-skipped (no invoke).
  * Successful invokes commit ``external_ref`` to the ledger when a key is present.
    """
    result = DeclarativeExecutionResult()
    for action in actions:
        tool_id = str(action.get("tool_id", ""))
        idempotency_key = action.get("idempotency_key")
        key = idempotency_key if isinstance(idempotency_key, str) else None
        args = action.get("args") if isinstance(action.get("args"), dict) else {}

        if action.get("replay_skipped"):
            external_ref = _ledger_external_ref(ledger, key) if ledger is not None and key else None
            result.results.append(
                DeclarativeActionExecution(
                    tool_id=tool_id,
                    status="replay_skipped",
                    idempotency_key=key,
                    external_ref=external_ref,
                    replay_skipped=True,
                )
            )
            continue

        if invoker is None:
            result.results.append(
                DeclarativeActionExecution(
                    tool_id=tool_id,
                    status="skipped_no_invoker",
                    idempotency_key=key,
                )
            )
            continue

        started = time.perf_counter()
        invoke_result = await invoker.invoke(
            tool_id=tool_id,
            args=args,
            idempotency_key=key,
        )
        duration_ms = invoke_result.duration_ms or int((time.perf_counter() - started) * 1000)

        if invoke_result.status == "success":
            if ledger is not None and key:
                ledger.commit(key, external_ref=invoke_result.external_ref)
            result.results.append(
                DeclarativeActionExecution(
                    tool_id=tool_id,
                    status="success",
                    idempotency_key=key,
                    output=invoke_result.output,
                    external_ref=invoke_result.external_ref,
                    duration_ms=duration_ms,
                )
            )
            continue

        result.results.append(
            DeclarativeActionExecution(
                tool_id=tool_id,
                status=invoke_result.status,
                idempotency_key=key,
                error=invoke_result.error,
                duration_ms=duration_ms,
            )
        )
    return result


@dataclass
class CallableDeclarativeToolInvoker:
    """Test and host adapter wrapping an async invoke callable."""

    _invoke_fn: Callable[..., Awaitable[DeclarativeToolInvokeResult]]

    async def invoke(
        self,
        *,
        tool_id: str,
        args: dict[str, Any],
        idempotency_key: str | None,
    ) -> DeclarativeToolInvokeResult:
        return await self._invoke_fn(
            tool_id=tool_id,
            args=args,
            idempotency_key=idempotency_key,
        )
