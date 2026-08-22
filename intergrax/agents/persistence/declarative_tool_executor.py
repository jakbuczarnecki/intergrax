# © Artur Czarnecki. All rights reserved.

"""Declarative ``requested_actions`` execution with ledger commit (ACP-PROD-2)."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable
from uuid import uuid4

from intergrax.agents.persistence.idempotency_ledger_bridge import (
    SideEffectCommitPayload,
    resolve_external_ref_from_store,
)
from intergrax.agents.persistence.side_effect_ledger import SideEffectLedger
from intergrax.contracts.idempotency_store import (
    ClaimOutcome,
    ClaimResult,
    IdempotencyStore,
    InvocationClaim,
    InvocationUncertaintyError,
)
from intergrax.tools.execution_models import ToolExecutionResult

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


_DEFAULT_LEASE_SECONDS = 300


def _ledger_external_ref(ledger: SideEffectLedger, idempotency_key: str) -> str | None:
    for record in ledger.records():
        if record.idempotency_key == idempotency_key:
            return record.external_ref
    return None


def _replay_terminal_status(completed: ToolExecutionResult[Any]) -> DeclarativeToolStatus:
    """Map a durable completed failure to declarative terminal status."""
    if completed.success:
        return "replay_skipped"
    if completed.error is not None and completed.error.error_code == "declarative.denied":
        return "denied"
    return "failed"


def _replay_external_ref(
    *,
    claim_result: ClaimResult,
    idempotency_store: IdempotencyStore,
    tenant_id: str,
    idempotency_key: str,
) -> str | None:
    cached = claim_result.completed_result
    if cached is not None and cached.success and cached.output is not None:
        output = cached.output
        if isinstance(output, SideEffectCommitPayload):
            return output.external_ref
        external_ref = output.model_dump().get("external_ref")
        return external_ref if isinstance(external_ref, str) else None
    return resolve_external_ref_from_store(
        idempotency_store=idempotency_store,
        tenant_id=tenant_id,
        idempotency_key=idempotency_key,
    )


def _complete_declarative_claim(
    *,
    idempotency_store: IdempotencyStore,
    tenant_id: str,
    idempotency_key: str,
    claim: InvocationClaim,
    invoke_result: DeclarativeToolInvokeResult,
    tool_id: str,
) -> None:
    if invoke_result.status == "success":
        payload = SideEffectCommitPayload(
            tool_id=tool_id,
            external_ref=invoke_result.external_ref,
        )
        result: ToolExecutionResult[SideEffectCommitPayload] = ToolExecutionResult.ok(payload)
    else:
        result = ToolExecutionResult.fail(
            code=f"declarative.{invoke_result.status}",
            message=invoke_result.error or invoke_result.status,
        )
    idempotency_store.complete_with_claim(tenant_id, idempotency_key, claim, result)


async def execute_declarative_actions(
    *,
    actions: list[dict[str, Any]],
    ledger: SideEffectLedger | None,
    invoker: DeclarativeToolInvoker | None,
    idempotency_store: IdempotencyStore | None = None,
    tenant_id: str = "default",
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
            external_ref = None
            if key:
                external_ref = _ledger_external_ref(ledger, key) if ledger is not None else None
                if external_ref is None:
                    external_ref = resolve_external_ref_from_store(
                        idempotency_store=idempotency_store,
                        tenant_id=tenant_id,
                        idempotency_key=key,
                    )
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

        claim: InvocationClaim | None = None
        if key and idempotency_store is not None:
            claim_result = idempotency_store.claim(
                tenant_id,
                key,
                f"declarative-attempt-{uuid4().hex}",
                _DEFAULT_LEASE_SECONDS,
            )
            if claim_result.outcome == ClaimOutcome.REPLAY_COMPLETED:
                completed = claim_result.completed_result
                if completed is not None and completed.success:
                    external_ref = _replay_external_ref(
                        claim_result=claim_result,
                        idempotency_store=idempotency_store,
                        tenant_id=tenant_id,
                        idempotency_key=key,
                    )
                    result.results.append(
                        DeclarativeActionExecution(
                            tool_id=tool_id,
                            status="replay_skipped",
                            idempotency_key=key,
                            external_ref=external_ref,
                            replay_skipped=True,
                        )
                    )
                else:
                    error_message = None
                    if completed is not None and completed.error is not None:
                        error_message = completed.error.error_message
                    terminal_status = (
                        _replay_terminal_status(completed)
                        if completed is not None
                        else "failed"
                    )
                    result.results.append(
                        DeclarativeActionExecution(
                            tool_id=tool_id,
                            status=terminal_status,
                            idempotency_key=key,
                            error=error_message,
                        )
                    )
                continue
            if claim_result.outcome == ClaimOutcome.BLOCKED_ACTIVE:
                result.results.append(
                    DeclarativeActionExecution(
                        tool_id=tool_id,
                        status="failed",
                        idempotency_key=key,
                        error=(
                            f"Invocation already claimed for key={key}. "
                            "Blocking concurrent execution."
                        ),
                    )
                )
                continue
            if claim_result.outcome == ClaimOutcome.UNCERTAIN:
                raise InvocationUncertaintyError(
                    f"Invocation outcome uncertain for key={key}. "
                    "Reconciliation required before retry.",
                )
            claim = claim_result.claim
            if claim is None:
                raise RuntimeError("Ledger inconsistency: ACQUIRED without claim.")

        started = time.perf_counter()
        invoke_result = await invoker.invoke(
            tool_id=tool_id,
            args=args,
            idempotency_key=key,
        )
        duration_ms = invoke_result.duration_ms or int((time.perf_counter() - started) * 1000)

        if key and idempotency_store is not None and claim is not None:
            _complete_declarative_claim(
                idempotency_store=idempotency_store,
                tenant_id=tenant_id,
                idempotency_key=key,
                claim=claim,
                invoke_result=invoke_result,
                tool_id=tool_id,
            )

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
