# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.agents.persistence.compensation_enqueue import (
    build_compensation_idempotency_key,
    enqueue_compensations_for_step_failure,
)
from intergrax.agents.persistence.declarative_tool_executor import (
    CallableDeclarativeToolInvoker,
    DeclarativeToolInvokeResult,
)
from intergrax.agents.persistence.idempotency_keys import build_default_idempotency_key
from intergrax.agents.persistence.side_effect_ledger import SideEffectLedger
from intergrax.contracts.side_effect import SideEffectKind, SideEffectStatus
from intergrax.tools.tool_execution_profile import (
    ToolExecutionProfile,
    ToolMutability,
    ToolReversibility,
)


def _committed_ledger() -> tuple[SideEffectLedger, str]:
    ledger = SideEffectLedger()
    key = build_default_idempotency_key(
        run_id="run-1",
        step_index=0,
        kind=SideEffectKind.TOOL,
        target="email.send",
        args={"to": "a@example.com"},
    )
    ledger.register(
        idempotency_key=key,
        run_id="run-1",
        step_index=0,
        target="email.send",
    )
    ledger.commit(key, external_ref="msg-42")
    return ledger, key


@pytest.mark.unit
@pytest.mark.gate
def test_compensation_idempotency_key_is_distinct() -> None:
    original = "acp:abc123"
    assert build_compensation_idempotency_key(original) == "comp:acp:abc123"
    assert build_compensation_idempotency_key(original) != original


@pytest.mark.unit
@pytest.mark.gate
async def test_compensation_enqueue_invokes_registered_tool() -> None:
    ledger, key = _committed_ledger()
    invoked: list[str] = []

    async def _invoke(**kwargs):  # type: ignore[no-untyped-def]
        invoked.append(kwargs["tool_id"])
        return DeclarativeToolInvokeResult(status="success")

    profiles = {
        "email.send": ToolExecutionProfile(
            tool_id="email.send",
            mutability=ToolMutability.MUTATING,
            reversibility=ToolReversibility.COMPENSATABLE,
            requires_idempotency_key=True,
            compensation_tool_id="email.recall",
        ),
    }
    result = await enqueue_compensations_for_step_failure(
        ledger=ledger,
        tool_profiles=profiles,
        step_index=0,
        invoker=CallableDeclarativeToolInvoker(_invoke),
        action_args={"email.send": {"to": "a@example.com"}},
    )
    assert invoked == ["email.recall"]
    assert result.actions[0].status == "compensated"
    assert ledger.records()[0].status == SideEffectStatus.COMPENSATED
    assert "original_external_ref" in result.actions[0].request.args
    assert result.actions[0].request.idempotency_key == build_compensation_idempotency_key(key)


@pytest.mark.unit
@pytest.mark.gate
async def test_compensation_enqueue_manual_marks_failed() -> None:
    ledger, _ = _committed_ledger()
    profiles = {
        "email.send": ToolExecutionProfile(
            tool_id="email.send",
            mutability=ToolMutability.MUTATING,
            reversibility=ToolReversibility.MANUAL,
            requires_idempotency_key=True,
        ),
    }
    result = await enqueue_compensations_for_step_failure(
        ledger=ledger,
        tool_profiles=profiles,
        step_index=0,
    )
    assert result.actions[0].status == "manual_required"
    assert ledger.records()[0].status == SideEffectStatus.FAILED


@pytest.mark.unit
@pytest.mark.gate
async def test_compensation_enqueue_without_handler_skips() -> None:
    ledger, _ = _committed_ledger()
    profiles = {
        "email.send": ToolExecutionProfile(
            tool_id="email.send",
            mutability=ToolMutability.MUTATING,
            reversibility=ToolReversibility.COMPENSATABLE,
            requires_idempotency_key=True,
            compensation_tool_id=None,
        ),
    }
    result = await enqueue_compensations_for_step_failure(
        ledger=ledger,
        tool_profiles=profiles,
        step_index=0,
    )
    assert result.actions[0].status == "skipped"
    assert ledger.records()[0].status == SideEffectStatus.FAILED
