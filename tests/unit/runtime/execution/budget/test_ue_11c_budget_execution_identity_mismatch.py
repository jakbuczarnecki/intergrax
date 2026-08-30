# © Artur Czarnecki. All rights reserved.

"""UE-11C — budget vs active execution identity mismatch fail-closed proof."""

from __future__ import annotations

from collections.abc import Callable
from contextvars import Token
from typing import Literal

import pytest

from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    reset_active_execution_identity,
)
from intergrax.runtime.execution.active_execution_budget import (
    ActiveExecutionBudgetState,
    bind_active_execution_budget,
    reset_active_execution_budget,
)
from intergrax.runtime.execution.budget.consumption import consume_llm_call, consume_tool_call
from intergrax.runtime.execution.budget.ledger import (
    InMemoryExecutionBudgetLedger,
    create_execution_budget_ledger,
)
from intergrax.runtime.execution.budget.models import ExecutionBudgetAllocationMode
from intergrax.runtime.execution.budget.snapshot import RunBudgetLedgerSnapshot
from intergrax.runtime.nexus.budget.budget_models import RunBudget

pytestmark = pytest.mark.unit

_MismatchConsumptionKind = Literal["llm", "tool"]


def _bind_identity_and_budget(
    ledger: InMemoryExecutionBudgetLedger,
    *,
    run_id: RunId,
    attempt_id: AttemptId,
    active_execution_id: ExecutionId,
    budget_execution_id: ExecutionId,
) -> tuple[Token, Token]:
    identity_token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=active_execution_id,
    )
    budget_token = bind_active_execution_budget(
        ActiveExecutionBudgetState(
            execution_id=budget_execution_id,
            mode=ExecutionBudgetAllocationMode.SHARED,
            ledger=ledger,
        )
    )
    return identity_token, budget_token


def _reset_identity_and_budget(identity_token: Token, budget_token: Token) -> None:
    reset_active_execution_budget(budget_token)
    reset_active_execution_identity(identity_token)


def _record_execution_ids(snapshot: RunBudgetLedgerSnapshot) -> frozenset[ExecutionId]:
    return frozenset(record.execution_id for record in snapshot.records)


def _assert_mismatch_fails_closed_before_consumption(
    *,
    consume: Callable[[], None],
    active_execution_id: ExecutionId,
    snapshot_before: RunBudgetLedgerSnapshot,
    ledger: InMemoryExecutionBudgetLedger,
    attempt_id: AttemptId,
    consumption_kind: _MismatchConsumptionKind,
) -> None:
    participant_ids_before = _record_execution_ids(snapshot_before)
    with pytest.raises(
        RuntimeError,
        match="execution budget execution_id mismatch",
    ):
        consume()
    snapshot_after = ledger.export_snapshot(attempt_id)
    participant_ids_after = _record_execution_ids(snapshot_after)

    assert len(snapshot_after.records) == len(snapshot_before.records), (
        "ledger record count must remain unchanged after rejected mismatch consumption"
    )
    assert snapshot_after == snapshot_before, (
        "ledger snapshot must remain unchanged after rejected mismatch consumption"
    )
    if consumption_kind == "llm":
        assert snapshot_after.root_shared_consumed.llm_calls == (
            snapshot_before.root_shared_consumed.llm_calls
        ), "mismatch must not increment llm_calls before governed consumption"
    else:
        assert snapshot_after.root_shared_consumed.tool_calls == (
            snapshot_before.root_shared_consumed.tool_calls
        ), "mismatch must not increment tool_calls before governed consumption"
    assert active_execution_id not in participant_ids_after or (
        active_execution_id in participant_ids_before
    ), (
        "mismatch consumption must not register active execution participant "
        f"{active_execution_id!r}"
    )


def _run_mismatch_consumption_proof(
    consume: Callable[[], None],
    *,
    consumption_kind: _MismatchConsumptionKind,
) -> None:
    ledger = create_execution_budget_ledger(RunBudget())
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    active_execution_id = mint_execution_id()
    budget_execution_id = mint_execution_id()
    assert active_execution_id != budget_execution_id

    identity_token, budget_token = _bind_identity_and_budget(
        ledger,
        run_id=run_id,
        attempt_id=attempt_id,
        active_execution_id=active_execution_id,
        budget_execution_id=budget_execution_id,
    )
    try:
        snapshot_before = ledger.export_snapshot(attempt_id)
        _assert_mismatch_fails_closed_before_consumption(
            consume=consume,
            active_execution_id=active_execution_id,
            snapshot_before=snapshot_before,
            ledger=ledger,
            attempt_id=attempt_id,
            consumption_kind=consumption_kind,
        )
    finally:
        _reset_identity_and_budget(identity_token, budget_token)


def test_budget_execution_id_mismatch_fails_closed_before_llm_consumption() -> None:
    _run_mismatch_consumption_proof(consume_llm_call, consumption_kind="llm")


def test_budget_execution_id_mismatch_fails_closed_before_tool_consumption() -> None:
    _run_mismatch_consumption_proof(consume_tool_call, consumption_kind="tool")


def test_matching_budget_execution_id_allows_governed_consumption() -> None:
    ledger = create_execution_budget_ledger(RunBudget())
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()

    identity_token, budget_token = _bind_identity_and_budget(
        ledger,
        run_id=run_id,
        attempt_id=attempt_id,
        active_execution_id=execution_id,
        budget_execution_id=execution_id,
    )
    try:
        snapshot_before = ledger.export_snapshot(attempt_id)
        consume_llm_call()
        snapshot_after = ledger.export_snapshot(attempt_id)
        assert snapshot_after.root_shared_consumed.llm_calls == (
            snapshot_before.root_shared_consumed.llm_calls + 1
        )
    finally:
        _reset_identity_and_budget(identity_token, budget_token)
