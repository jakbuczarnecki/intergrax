# © Artur Czarnecki. All rights reserved.

"""R3 declarative claim-before-invoke coordination tests (PCM-SIDE-EFFECT-COORDINATION-INTEGRITY)."""

from __future__ import annotations

import time
from typing import Any

import pytest
from pydantic import BaseModel

from intergrax.agents.persistence.declarative_tool_executor import (
    CallableDeclarativeToolInvoker,
    DeclarativeToolInvokeResult,
    execute_declarative_actions,
)
from intergrax.agents.persistence.idempotency_ledger_bridge import SideEffectCommitPayload
from intergrax.agents.persistence.side_effect_ledger import SideEffectLedger
from intergrax.contracts.idempotency_store import (
    ClaimOutcome,
    InvocationClaim,
    InvocationStatus,
    InvocationUncertaintyError,
)
from intergrax.contracts.side_effect import SideEffectKind, SideEffectStatus
from intergrax.agents.persistence.idempotency_keys import build_default_idempotency_key
from intergrax.runtime.tools.in_memory_idempotency_store import InMemoryIdempotencyStore
from intergrax.tools.execution_models import ToolExecutionResult

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]

TOOL_ID = "email.send"
TENANT = "tenant-a"
KEY = "declarative:key:1"


class _SimulatedCrashError(RuntimeError):
    pass


class _OrderingStore(InMemoryIdempotencyStore):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[str] = []
        self.completed_claims: list[InvocationClaim] = []

    def claim(
        self,
        tenant_id: str,
        key: str,
        owner_id: str,
        lease_seconds: int,
        operation_identity=None,
    ):
        self.events.append("claim")
        return super().claim(
            tenant_id,
            key,
            owner_id,
            lease_seconds,
            operation_identity=operation_identity,
        )

    def complete_with_claim(
        self,
        tenant_id: str,
        key: str,
        claim: InvocationClaim,
        result: ToolExecutionResult[BaseModel],
        completed_ttl_seconds: int | None = None,
    ) -> None:
        self.completed_claims.append(claim)
        return super().complete_with_claim(
            tenant_id,
            key,
            claim,
            result,
            completed_ttl_seconds,
        )


class _CrashOnCompleteStore(InMemoryIdempotencyStore):
    def complete_with_claim(
        self,
        tenant_id: str,
        key: str,
        claim: InvocationClaim,
        result: ToolExecutionResult[BaseModel],
        completed_ttl_seconds: int | None = None,
    ) -> None:
        raise _SimulatedCrashError("crash before complete_with_claim")


class _ShortLeaseCrashStore(_CrashOnCompleteStore):
    def claim(
        self,
        tenant_id: str,
        key: str,
        owner_id: str,
        lease_seconds: int,
        operation_identity=None,
    ):
        del lease_seconds
        return super().claim(
            tenant_id,
            key,
            owner_id,
            1,
            operation_identity=operation_identity,
        )


def _action() -> dict[str, Any]:
    return {"tool_id": TOOL_ID, "idempotency_key": KEY, "args": {}}


@pytest.mark.asyncio
async def test_r3_1_claim_before_declarative_invoke() -> None:
    store = _OrderingStore()
    invoke_events: list[str] = []

    async def _invoke(**kwargs):  # type: ignore[no-untyped-def]
        invoke_events.append("invoke")
        return DeclarativeToolInvokeResult(status="success", external_ref="msg-1")

    await execute_declarative_actions(
        actions=[_action()],
        ledger=None,
        invoker=CallableDeclarativeToolInvoker(_invoke),
        idempotency_store=store,
        tenant_id=TENANT,
    )
    assert store.events == ["claim"]
    assert invoke_events == ["invoke"]
    assert store.events[0] == "claim"


@pytest.mark.asyncio
async def test_r3_2_completed_replay_does_not_invoke() -> None:
    store = InMemoryIdempotencyStore()
    claim = store.claim(TENANT, KEY, "setup-owner", lease_seconds=300)
    assert claim.claim is not None
    payload = SideEffectCommitPayload(tool_id=TOOL_ID, external_ref="msg-replay")
    store.complete_with_claim(
        TENANT,
        KEY,
        claim.claim,
        ToolExecutionResult.ok(payload),
    )
    invoke_count = 0

    async def _invoke(**kwargs):  # type: ignore[no-untyped-def]
        nonlocal invoke_count
        invoke_count += 1
        return DeclarativeToolInvokeResult(status="success")

    execution = await execute_declarative_actions(
        actions=[_action()],
        ledger=None,
        invoker=CallableDeclarativeToolInvoker(_invoke),
        idempotency_store=store,
        tenant_id=TENANT,
    )
    assert invoke_count == 0
    assert execution.results[0].status == "replay_skipped"
    assert execution.results[0].external_ref == "msg-replay"


@pytest.mark.asyncio
async def test_r3_3_active_claim_does_not_invoke() -> None:
    store = InMemoryIdempotencyStore()
    store.claim(TENANT, KEY, "other-owner", lease_seconds=300)
    invoke_count = 0

    async def _invoke(**kwargs):  # type: ignore[no-untyped-def]
        nonlocal invoke_count
        invoke_count += 1
        return DeclarativeToolInvokeResult(status="success")

    execution = await execute_declarative_actions(
        actions=[_action()],
        ledger=None,
        invoker=CallableDeclarativeToolInvoker(_invoke),
        idempotency_store=store,
        tenant_id=TENANT,
    )
    assert invoke_count == 0
    assert execution.results[0].status == "failed"
    assert "already claimed" in (execution.results[0].error or "")


@pytest.mark.asyncio
async def test_r3_4_uncertain_does_not_invoke() -> None:
    store = InMemoryIdempotencyStore()
    claim = store.claim(TENANT, KEY, "crash-owner", lease_seconds=1)
    assert claim.outcome == ClaimOutcome.ACQUIRED
    time.sleep(1.2)
    invoke_count = 0

    async def _invoke(**kwargs):  # type: ignore[no-untyped-def]
        nonlocal invoke_count
        invoke_count += 1
        return DeclarativeToolInvokeResult(status="success")

    with pytest.raises(InvocationUncertaintyError):
        await execute_declarative_actions(
            actions=[_action()],
            ledger=None,
            invoker=CallableDeclarativeToolInvoker(_invoke),
            idempotency_store=store,
            tenant_id=TENANT,
        )
    assert invoke_count == 0
    assert store.get_status(TENANT, KEY) == InvocationStatus.UNCERTAIN


@pytest.mark.asyncio
async def test_r3_5_crash_after_effect_no_second_invoke() -> None:
    store = _ShortLeaseCrashStore()
    invoke_count = 0

    async def _invoke(**kwargs):  # type: ignore[no-untyped-def]
        nonlocal invoke_count
        invoke_count += 1
        return DeclarativeToolInvokeResult(status="success", external_ref="msg-crash")

    with pytest.raises(_SimulatedCrashError):
        await execute_declarative_actions(
            actions=[_action()],
            ledger=None,
            invoker=CallableDeclarativeToolInvoker(_invoke),
            idempotency_store=store,
            tenant_id=TENANT,
        )
    assert invoke_count == 1
    time.sleep(1.2)

    with pytest.raises(InvocationUncertaintyError):
        await execute_declarative_actions(
            actions=[_action()],
            ledger=None,
            invoker=CallableDeclarativeToolInvoker(_invoke),
            idempotency_store=store,
            tenant_id=TENANT,
        )
    assert invoke_count == 1


@pytest.mark.asyncio
async def test_r3_6_success_completes_with_exact_claim() -> None:
    store = _OrderingStore()

    async def _invoke(**kwargs):  # type: ignore[no-untyped-def]
        return DeclarativeToolInvokeResult(status="success", external_ref="msg-exact")

    await execute_declarative_actions(
        actions=[_action()],
        ledger=None,
        invoker=CallableDeclarativeToolInvoker(_invoke),
        idempotency_store=store,
        tenant_id=TENANT,
    )
    assert len(store.completed_claims) == 1
    completed = store.get_completed_result(TENANT, KEY)
    assert completed is not None and completed.success
    entry = store._store[(TENANT, KEY)]  # noqa: SLF001
    assert entry.claim is not None
    assert store.completed_claims[0].owner_id == entry.claim.owner_id
    assert store.completed_claims[0].fence == entry.claim.fence


@pytest.mark.asyncio
async def test_r3_7_external_ref_replay() -> None:
    store = InMemoryIdempotencyStore()
    invoke_count = 0

    async def _invoke(**kwargs):  # type: ignore[no-untyped-def]
        nonlocal invoke_count
        invoke_count += 1
        return DeclarativeToolInvokeResult(status="success", external_ref="msg-42")

    first = await execute_declarative_actions(
        actions=[_action()],
        ledger=None,
        invoker=CallableDeclarativeToolInvoker(_invoke),
        idempotency_store=store,
        tenant_id=TENANT,
    )
    second = await execute_declarative_actions(
        actions=[_action()],
        ledger=None,
        invoker=CallableDeclarativeToolInvoker(_invoke),
        idempotency_store=store,
        tenant_id=TENANT,
    )
    assert invoke_count == 1
    assert first.results[0].external_ref == "msg-42"
    assert second.results[0].status == "replay_skipped"
    assert second.results[0].external_ref == "msg-42"


@pytest.mark.asyncio
async def test_r3_8_failure_semantics_do_not_mark_completed_success() -> None:
    store = InMemoryIdempotencyStore()

    async def _invoke(**kwargs):  # type: ignore[no-untyped-def]
        return DeclarativeToolInvokeResult(status="failed", error="smtp_down")

    execution = await execute_declarative_actions(
        actions=[_action()],
        ledger=None,
        invoker=CallableDeclarativeToolInvoker(_invoke),
        idempotency_store=store,
        tenant_id=TENANT,
    )
    assert execution.results[0].status == "failed"
    completed = store.get_completed_result(TENANT, KEY)
    assert completed is not None
    assert completed.success is False
    assert completed.error is not None
    assert completed.error.error_code == "declarative.failed"


@pytest.mark.asyncio
async def test_r3_9_side_effect_ledger_positive_control() -> None:
    ledger = SideEffectLedger()
    key = build_default_idempotency_key(
        run_id="run-1",
        step_index=0,
        kind=SideEffectKind.TOOL,
        target=TOOL_ID,
        args={},
    )
    ledger.register(
        idempotency_key=key,
        run_id="run-1",
        step_index=0,
        target=TOOL_ID,
    )
    store = InMemoryIdempotencyStore()

    async def _invoke(**kwargs):  # type: ignore[no-untyped-def]
        return DeclarativeToolInvokeResult(status="success", external_ref="msg-ledger")

    await execute_declarative_actions(
        actions=[{"tool_id": TOOL_ID, "idempotency_key": key, "args": {}}],
        ledger=ledger,
        invoker=CallableDeclarativeToolInvoker(_invoke),
        idempotency_store=store,
        tenant_id=TENANT,
    )
    assert ledger.records()[0].status == SideEffectStatus.COMMITTED
    assert ledger.records()[0].external_ref == "msg-ledger"
