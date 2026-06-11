# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.agents.persistence.declarative_tool_executor import (
    CallableDeclarativeToolInvoker,
    DeclarativeToolInvokeResult,
    execute_declarative_actions,
)
from intergrax.agents.persistence.idempotency_keys import build_default_idempotency_key
from intergrax.agents.persistence.side_effect_ledger import SideEffectLedger
from intergrax.contracts.side_effect import SideEffectKind, SideEffectStatus


@pytest.mark.unit
@pytest.mark.gate
async def test_declarative_executor_commits_on_success() -> None:
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
    invoke_count = 0

    async def _invoke(**kwargs):  # type: ignore[no-untyped-def]
        nonlocal invoke_count
        invoke_count += 1
        return DeclarativeToolInvokeResult(
            status="success",
            external_ref="msg-99",
            output={"sent": True},
        )

    invoker = CallableDeclarativeToolInvoker(_invoke)
    result = await execute_declarative_actions(
        actions=[{"tool_id": "email.send", "idempotency_key": key, "args": {"to": "a@example.com"}}],
        ledger=ledger,
        invoker=invoker,
    )
    assert invoke_count == 1
    assert result.results[0].status == "success"
    assert ledger.records()[0].status == SideEffectStatus.COMMITTED
    assert ledger.records()[0].external_ref == "msg-99"


@pytest.mark.unit
@pytest.mark.gate
async def test_declarative_executor_skips_replay_without_invoke() -> None:
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
    invoke_count = 0

    async def _invoke(**kwargs):  # type: ignore[no-untyped-def]
        nonlocal invoke_count
        invoke_count += 1
        return DeclarativeToolInvokeResult(status="success")

    invoker = CallableDeclarativeToolInvoker(_invoke)
    result = await execute_declarative_actions(
        actions=[
            {
                "tool_id": "email.send",
                "idempotency_key": key,
                "args": {"to": "a@example.com"},
                "replay_skipped": True,
            }
        ],
        ledger=ledger,
        invoker=invoker,
    )
    assert invoke_count == 0
    assert result.results[0].status == "replay_skipped"
    assert result.results[0].external_ref == "msg-42"
    assert result.replay_skipped_count == 1


@pytest.mark.unit
@pytest.mark.gate
async def test_declarative_executor_surfaces_failed_invoke() -> None:
    async def _invoke(**kwargs):  # type: ignore[no-untyped-def]
        return DeclarativeToolInvokeResult(status="failed", error="smtp_down")

    invoker = CallableDeclarativeToolInvoker(_invoke)
    result = await execute_declarative_actions(
        actions=[{"tool_id": "email.send", "args": {}}],
        ledger=SideEffectLedger(),
        invoker=invoker,
    )
    assert result.failed_tool_id == "email.send"
    assert result.results[0].error == "smtp_down"
