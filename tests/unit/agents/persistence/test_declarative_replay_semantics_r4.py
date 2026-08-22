# © Artur Czarnecki. All rights reserved.

"""R4 declarative replay-semantics correction tests (PCM-SIDE-EFFECT-COORDINATION-INTEGRITY)."""

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
from intergrax.agents.persistence.idempotency_ledger_bridge import (
    SideEffectCommitPayload,
    should_skip_side_effect_replay,
)
from intergrax.agents.persistence.side_effect_ledger import SideEffectLedger
from intergrax.agents.persistence.tool_action_validation import validate_requested_actions
from intergrax.contracts.agent_run_enums import SideEffectMode
from intergrax.contracts.idempotency_store import (
    ClaimOutcome,
    InvocationClaim,
    InvocationStatus,
    InvocationUncertaintyError,
)
from intergrax.runtime.tools.in_memory_idempotency_store import InMemoryIdempotencyStore
from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.execution_models import ToolExecutionResult
from intergrax.tools.tool_execution_profile import build_profile_map
from pydantic import BaseModel as PydanticBaseModel

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]

TOOL_ID = "email.send"
TENANT = "tenant-a"
KEY = "declarative:r4:key"


class _In(PydanticBaseModel):
    pass


class _Out(PydanticBaseModel):
    pass


_MUTATING = ToolContract(
    tool_id=TOOL_ID,
    name=TOOL_ID,
    description="send",
    input_schema=_In,
    output_schema=_Out,
    error_mapping={},
    side_effects=True,
    risk_level=ToolRiskLevel.HIGH,
)


def _action() -> dict[str, Any]:
    return {"tool_id": TOOL_ID, "idempotency_key": KEY, "args": {}}


def _complete_failed(
    store: InMemoryIdempotencyStore,
    *,
    error_code: str = "declarative.failed",
    error_message: str = "smtp_down",
) -> None:
    claim = store.claim(TENANT, KEY, "setup-owner", lease_seconds=300)
    assert claim.claim is not None
    store.complete_with_claim(
        TENANT,
        KEY,
        claim.claim,
        ToolExecutionResult.fail(code=error_code, message=error_message),
    )


def _complete_success(
    store: InMemoryIdempotencyStore,
    *,
    external_ref: str = "msg-ok",
) -> None:
    claim = store.claim(TENANT, KEY, "setup-owner", lease_seconds=300)
    assert claim.claim is not None
    payload = SideEffectCommitPayload(tool_id=TOOL_ID, external_ref=external_ref)
    store.complete_with_claim(
        TENANT,
        KEY,
        claim.claim,
        ToolExecutionResult.ok(payload),
    )


def test_r4_1_helper_does_not_skip_failed_completed() -> None:
    store = InMemoryIdempotencyStore()
    _complete_failed(store)
    assert store.get_status(TENANT, KEY) == InvocationStatus.COMPLETED
    assert should_skip_side_effect_replay(
        idempotency_key=KEY,
        idempotency_store=store,
        tenant_id=TENANT,
    ) is False


def test_r4_2_helper_skips_successful_completed() -> None:
    store = InMemoryIdempotencyStore()
    _complete_success(store)
    assert should_skip_side_effect_replay(
        idempotency_key=KEY,
        idempotency_store=store,
        tenant_id=TENANT,
    ) is True


def test_r4_3_validator_does_not_mark_failed_result_replay_skipped() -> None:
    store = InMemoryIdempotencyStore()
    _complete_failed(store)
    profiles = build_profile_map([_MUTATING])
    normalized = validate_requested_actions(
        requested_actions=[_action()],
        side_effect_mode=SideEffectMode.DECLARATIVE,
        tool_profiles=profiles,
        run_id="run-new",
        step_index=0,
        ledger=SideEffectLedger(),
        idempotency_store=store,
        tenant_id=TENANT,
    )
    assert normalized[0].get("replay_skipped") is not True


@pytest.mark.asyncio
async def test_r4_4_failed_replay_preserves_failure() -> None:
    store = InMemoryIdempotencyStore()
    invoke_count = 0

    async def _invoke(**kwargs):  # type: ignore[no-untyped-def]
        nonlocal invoke_count
        invoke_count += 1
        return DeclarativeToolInvokeResult(status="failed", error="smtp_down")

    invoker = CallableDeclarativeToolInvoker(_invoke)
    first = await execute_declarative_actions(
        actions=[_action()],
        ledger=None,
        invoker=invoker,
        idempotency_store=store,
        tenant_id=TENANT,
    )
    assert first.results[0].status == "failed"
    assert first.results[0].error == "smtp_down"

    second = await execute_declarative_actions(
        actions=[_action()],
        ledger=None,
        invoker=invoker,
        idempotency_store=store,
        tenant_id=TENANT,
    )
    assert invoke_count == 1
    assert second.results[0].status == "failed"
    assert second.results[0].error == "smtp_down"
    assert second.results[0].replay_skipped is False


@pytest.mark.asyncio
async def test_r4_5_denied_replay_preserves_denial() -> None:
    store = InMemoryIdempotencyStore()
    invoke_count = 0

    async def _invoke(**kwargs):  # type: ignore[no-untyped-def]
        nonlocal invoke_count
        invoke_count += 1
        return DeclarativeToolInvokeResult(status="denied", error="policy_block")

    invoker = CallableDeclarativeToolInvoker(_invoke)
    first = await execute_declarative_actions(
        actions=[_action()],
        ledger=None,
        invoker=invoker,
        idempotency_store=store,
        tenant_id=TENANT,
    )
    assert first.results[0].status == "denied"

    second = await execute_declarative_actions(
        actions=[_action()],
        ledger=None,
        invoker=invoker,
        idempotency_store=store,
        tenant_id=TENANT,
    )
    assert invoke_count == 1
    assert second.results[0].status == "denied"
    assert second.results[0].error == "policy_block"
    assert second.results[0].replay_skipped is False


@pytest.mark.asyncio
async def test_r4_6_successful_replay_still_skips() -> None:
    store = InMemoryIdempotencyStore()
    invoke_count = 0

    async def _invoke(**kwargs):  # type: ignore[no-untyped-def]
        nonlocal invoke_count
        invoke_count += 1
        return DeclarativeToolInvokeResult(status="success", external_ref="msg-42")

    invoker = CallableDeclarativeToolInvoker(_invoke)
    await execute_declarative_actions(
        actions=[_action()],
        ledger=None,
        invoker=invoker,
        idempotency_store=store,
        tenant_id=TENANT,
    )
    second = await execute_declarative_actions(
        actions=[_action()],
        ledger=None,
        invoker=invoker,
        idempotency_store=store,
        tenant_id=TENANT,
    )
    assert invoke_count == 1
    assert second.results[0].status == "replay_skipped"
    assert second.results[0].replay_skipped is True
    assert second.results[0].external_ref == "msg-42"


@pytest.mark.asyncio
async def test_r4_7_unknown_failed_code_fails_safe() -> None:
    store = InMemoryIdempotencyStore()
    _complete_failed(store, error_code="declarative.unknown", error_message="weird")
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
    assert execution.results[0].error == "weird"


@pytest.mark.asyncio
async def test_r4_8_uncertain_regression() -> None:
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


class _SimulatedCrashError(RuntimeError):
    pass


class _ShortLeaseCrashStore(InMemoryIdempotencyStore):
    def claim(
        self,
        tenant_id: str,
        key: str,
        owner_id: str,
        lease_seconds: int,
    ):
        del lease_seconds
        return super().claim(tenant_id, key, owner_id, 1)

    def complete_with_claim(
        self,
        tenant_id: str,
        key: str,
        claim: InvocationClaim,
        result: ToolExecutionResult[BaseModel],
        completed_ttl_seconds: int | None = None,
    ) -> None:
        raise _SimulatedCrashError("crash before complete_with_claim")


@pytest.mark.asyncio
async def test_r4_9_crash_regression() -> None:
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
async def test_r4_10_pcm_03_regression() -> None:
    from intergrax.agents.persistence.compensation_enqueue import build_compensation_idempotency_key
    from intergrax.agents.persistence.compensation_queue_store import (
        CompensationJob,
        CompensationJobStatus,
        InMemoryCompensationQueueStore,
    )
    from intergrax.agents.persistence.compensation_queue_worker import drain_pending_compensation_jobs
    from intergrax.contracts.side_effect import CompensationRequest

    store = InMemoryCompensationQueueStore()
    key = build_compensation_idempotency_key("acp:r4:pcm03")
    store.enqueue(
        CompensationJob(
            run_id="run-1",
            tenant_id=TENANT,
            agent_id="agent-a",
            step_index=0,
            request=CompensationRequest(
                original_side_effect_id="se-1",
                compensation_tool_id="email.recall",
                args={"original_external_ref": "msg-1"},
                idempotency_key=key,
            ),
        )
    )
    claim = store.claim_pending(TENANT, "worker-1", lease_seconds=1, limit=1)[0]

    async def _invoke(**kwargs):  # type: ignore[no-untyped-def]
        return DeclarativeToolInvokeResult(status="success")

    invoker = CallableDeclarativeToolInvoker(_invoke)
    await invoker.invoke(
        tool_id=claim.job.request.compensation_tool_id,
        args=claim.job.request.args,
        idempotency_key=claim.job.request.idempotency_key,
    )
    time.sleep(1.2)
    assert store.claim_pending(TENANT, "worker-2", lease_seconds=300, limit=1) == []
    assert store.list_uncertain(TENANT)

    invoke_count = 0

    async def _counting_invoke(**kwargs):  # type: ignore[no-untyped-def]
        nonlocal invoke_count
        invoke_count += 1
        return DeclarativeToolInvokeResult(status="success")

    await drain_pending_compensation_jobs(
        store,
        tenant_id=TENANT,
        invoker=CallableDeclarativeToolInvoker(_counting_invoke),
        owner_id="worker-2",
        limit=1,
    )
    assert invoke_count == 0
    loaded = store.get_by_idempotency_key(TENANT, key)
    assert loaded is not None
    assert loaded.status == CompensationJobStatus.UNCERTAIN
