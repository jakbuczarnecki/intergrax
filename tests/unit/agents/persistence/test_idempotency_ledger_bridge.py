# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.agents.persistence.declarative_tool_executor import (
    CallableDeclarativeToolInvoker,
    DeclarativeToolInvokeResult,
    execute_declarative_actions,
)
from intergrax.agents.persistence.idempotency_ledger_bridge import (
    record_side_effect_commit,
    resolve_external_ref_from_store,
    should_skip_side_effect_replay,
)
from intergrax.agents.persistence.side_effect_ledger import SideEffectLedger
from intergrax.agents.persistence.tool_action_validation import validate_requested_actions
from intergrax.contracts.agent_run_enums import SideEffectMode
from intergrax.contracts.idempotency_store import InvocationStatus
from intergrax.runtime.tools.in_memory_idempotency_store import InMemoryIdempotencyStore
from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.tool_execution_profile import build_profile_map
from pydantic import BaseModel

pytestmark = [pytest.mark.unit, pytest.mark.gate]

TOOL_ID = "email.send"
IDEMPOTENCY_KEY = "cross-run:send:1"
TENANT = "tenant-a"


class _In(BaseModel):
    pass


class _Out(BaseModel):
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


def test_store_marks_cross_run_replay_skip_without_ledger() -> None:
    store = InMemoryIdempotencyStore()
    record_side_effect_commit(
        idempotency_store=store,
        tenant_id=TENANT,
        idempotency_key=IDEMPOTENCY_KEY,
        tool_id=TOOL_ID,
        external_ref="msg-99",
    )
    assert should_skip_side_effect_replay(
        idempotency_key=IDEMPOTENCY_KEY,
        idempotency_store=store,
        tenant_id=TENANT,
    )
    assert (
        resolve_external_ref_from_store(
            idempotency_store=store,
            tenant_id=TENANT,
            idempotency_key=IDEMPOTENCY_KEY,
        )
        == "msg-99"
    )


def test_validate_requested_actions_skips_from_store_on_new_run() -> None:
    store = InMemoryIdempotencyStore()
    record_side_effect_commit(
        idempotency_store=store,
        tenant_id=TENANT,
        idempotency_key=IDEMPOTENCY_KEY,
        tool_id=TOOL_ID,
        external_ref="msg-store",
    )
    profiles = build_profile_map([_MUTATING])
    normalized = validate_requested_actions(
        requested_actions=[{"tool_id": TOOL_ID, "idempotency_key": IDEMPOTENCY_KEY, "args": {}}],
        side_effect_mode=SideEffectMode.DECLARATIVE,
        tool_profiles=profiles,
        run_id="run-new",
        step_index=0,
        ledger=SideEffectLedger(),
        idempotency_store=store,
        tenant_id=TENANT,
    )
    assert normalized[0].get("replay_skipped") is True


@pytest.mark.asyncio
async def test_execute_declarative_actions_persists_commit_to_store() -> None:
    store = InMemoryIdempotencyStore()
    invoke_count = 0

    async def _invoke(**kwargs):  # type: ignore[no-untyped-def]
        nonlocal invoke_count
        invoke_count += 1
        return DeclarativeToolInvokeResult(status="success", external_ref="msg-1")

    invoker = CallableDeclarativeToolInvoker(_invoke)
    ledger = SideEffectLedger()
    action = {"tool_id": TOOL_ID, "idempotency_key": IDEMPOTENCY_KEY, "args": {}}
    profiles = build_profile_map([_MUTATING])
    validated = validate_requested_actions(
        requested_actions=[action],
        side_effect_mode=SideEffectMode.DECLARATIVE,
        tool_profiles=profiles,
        run_id="run-1",
        step_index=0,
        ledger=ledger,
        idempotency_store=store,
        tenant_id=TENANT,
    )
    await execute_declarative_actions(
        actions=validated,
        ledger=ledger,
        invoker=invoker,
        idempotency_store=store,
        tenant_id=TENANT,
    )
    assert invoke_count == 1
    assert store.get_status(TENANT, IDEMPOTENCY_KEY) == InvocationStatus.COMPLETED

    replay_validated = validate_requested_actions(
        requested_actions=[action],
        side_effect_mode=SideEffectMode.DECLARATIVE,
        tool_profiles=profiles,
        run_id="run-2",
        step_index=0,
        ledger=SideEffectLedger(),
        idempotency_store=store,
        tenant_id=TENANT,
    )
    execution = await execute_declarative_actions(
        actions=replay_validated,
        ledger=SideEffectLedger(),
        invoker=invoker,
        idempotency_store=store,
        tenant_id=TENANT,
    )
    assert invoke_count == 1
    assert execution.results[0].replay_skipped is True
    assert execution.results[0].external_ref == "msg-1"
