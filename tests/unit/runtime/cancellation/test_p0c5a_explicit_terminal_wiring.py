# © Artur Czarnecki. All rights reserved.

"""P0C-5A — explicit terminal authority wiring and provider-neutral persistence."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from intergrax.applications._shared.task_control import governed_resume_checkpoint_task
from intergrax.applications._shared.task_control_wiring import (
    resolve_harness_task_control_execution_terminal,
)
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.contracts.control_plane_mutation import ControlPlaneMutationRequest
from intergrax.contracts.execution_identity import mint_attempt_id, mint_execution_id, mint_run_id, mint_task_id
from intergrax.contracts.execution_terminal import (
    ExecutionTerminalError,
    ExecutionTerminalPersistenceCapability,
    ExecutionTerminalRecord,
)
from intergrax.contracts.runtime_policy import EnforcementLevel, PolicyAction, PolicyDecision
from intergrax.runtime.execution.execution_terminal import (
    CheckpointStoreExecutionTerminalStore,
    ExecutionTerminalService,
    InMemoryExecutionTerminalStore,
    wire_execution_terminal_store,
)
from intergrax.runtime.execution.execution_terminal.durability_policy import (
    DURABLE_EXECUTION_TERMINAL_REQUIRED_MSG,
    validate_durable_execution_terminal_for_composition,
)
from intergrax.runtime.execution.execution_terminal.persistence import (
    CheckpointStoreExecutionTerminalStore as AdapterClass,
)
from intergrax.runtime.long_running.execution_tree_checkpoint import minimal_runtime_checkpoint
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.long_running.persistence_contract import TaskCheckpointPersistence
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from intergrax.runtime.governance.control_plane_mutation_authorization import (
    ControlPlaneMutationAuthorizationBoundary,
)
from intergrax.runtime.task.task import Task, TaskState

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_TASK_CONTROL_SOURCE = _REPO_ROOT / "intergrax/applications/_shared/task_control.py"
_PERSISTENCE_SOURCE = _REPO_ROOT / "intergrax/runtime/execution/execution_terminal/persistence.py"
_TENANT = "tenant-p0c5a"


class _FakeTerminalCapableCheckpointStore(TaskCheckpointPersistence):
    """Deterministic durable store with terminal capability — not SQLite."""

    def __init__(self) -> None:
        self._records: Dict[tuple[str, str], ExecutionTerminalRecord] = {}

    def list_for_task(self, task_id: str, tenant_id: str) -> list[TaskCheckpoint]:
        return []

    def get_latest(self, task_id: str, tenant_id: str) -> TaskCheckpoint | None:
        return None

    def get_by_token(
        self,
        task_id: str,
        tenant_id: str,
        resume_token: str,
    ) -> TaskCheckpoint | None:
        return None

    def list_paused(self) -> list[TaskCheckpoint]:
        return []

    def save(self, checkpoint: TaskCheckpoint) -> TaskCheckpoint:
        return checkpoint

    def get_terminal_record(self, *, tenant_id: str, task_id: str) -> ExecutionTerminalRecord | None:
        return self._records.get((tenant_id, task_id))

    def put_terminal_record_if_absent(self, record: ExecutionTerminalRecord) -> bool:
        key = (record.tenant_id, record.task_id)
        if key in self._records:
            return False
        self._records[key] = record
        return True


class _FakeCheckpointStoreWithoutTerminalCapability(TaskCheckpointPersistence):
    def list_for_task(self, task_id: str, tenant_id: str) -> list[TaskCheckpoint]:
        return []

    def get_latest(self, task_id: str, tenant_id: str) -> TaskCheckpoint | None:
        return None

    def get_by_token(
        self,
        task_id: str,
        tenant_id: str,
        resume_token: str,
    ) -> TaskCheckpoint | None:
        return None

    def list_paused(self) -> list[TaskCheckpoint]:
        return []

    def save(self, checkpoint: TaskCheckpoint) -> TaskCheckpoint:
        return checkpoint


@dataclass
class _AllowEvaluator:
    decision: PolicyDecision = field(
        default_factory=lambda: PolicyDecision(
            action=PolicyAction.ALLOW,
            reason="test_allow",
            enforcement_level=EnforcementLevel.MANDATORY,
            policy_rule_id="task_control.test_allow",
            decision_id="dec-allow",
        ),
    )

    def evaluate(self, request: ControlPlaneMutationRequest) -> PolicyDecision:
        return self.decision


def _paused_checkpoint() -> TaskCheckpoint:
    task_id = str(mint_task_id())
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    task = Task(
        task_id=task_id,
        tenant_id=_TENANT,
        user_id="user",
        message="paused",
        state=TaskState.WAITING_FOR_HUMAN,
    )
    return TaskCheckpoint(
        task_id=task_id,
        tenant_id=_TENANT,
        resume_token="rt-p0c5a",
        task_state=TaskState.WAITING_FOR_HUMAN,
        task_snapshot=task.model_dump(mode="json"),
        runtime=minimal_runtime_checkpoint(
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            root_execution_id=mint_execution_id(),
        ),
    )


class _StaticCheckpointStore:
    def __init__(self, checkpoint: TaskCheckpoint) -> None:
        self._checkpoint = checkpoint

    def get_by_token(self, task_id: str, tenant_id: str, resume_token: str) -> TaskCheckpoint | None:
        if (
            task_id == self._checkpoint.task_id
            and tenant_id == self._checkpoint.tenant_id
            and resume_token == self._checkpoint.resume_token
        ):
            return self._checkpoint
        return None


def test_task_control_has_no_runner_terminal_introspection() -> None:
    source = _TASK_CONTROL_SOURCE.read_text(encoding="utf-8")
    assert "getattr(runner" not in source
    assert "runner.nexus_loop" not in source


def test_generic_terminal_adapter_has_no_sqlite_coupling() -> None:
    source = _PERSISTENCE_SOURCE.read_text(encoding="utf-8")
    assert "SQLiteTaskCheckpointStore" not in source


def test_custom_terminal_capability_wires_durable_adapter() -> None:
    store = _FakeTerminalCapableCheckpointStore()
    assert isinstance(store, ExecutionTerminalPersistenceCapability)

    wired = wire_execution_terminal_store(checkpoint_store=store)
    assert isinstance(wired, CheckpointStoreExecutionTerminalStore)
    assert wired.is_durable is True

    terminal = ExecutionTerminalService(wired)
    task_id = str(mint_task_id())
    terminal.record_cancellation(tenant_id=_TENANT, task_id=task_id, reason="custom_provider")
    assert store.get_terminal_record(tenant_id=_TENANT, task_id=task_id) is not None


def test_store_without_terminal_capability_uses_in_memory_in_dev() -> None:
    store = _FakeCheckpointStoreWithoutTerminalCapability()
    assert not isinstance(store, ExecutionTerminalPersistenceCapability)

    wired = wire_execution_terminal_store(checkpoint_store=store)
    assert isinstance(wired, InMemoryExecutionTerminalStore)
    assert wired.is_durable is False


def test_production_without_terminal_capability_fails_closed() -> None:
    store = _FakeCheckpointStoreWithoutTerminalCapability()
    terminal_store = wire_execution_terminal_store(checkpoint_store=store)
    with pytest.raises(ExecutionTerminalError, match=DURABLE_EXECUTION_TERMINAL_REQUIRED_MSG):
        validate_durable_execution_terminal_for_composition(
            production_mode=True,
            checkpoint_store=store,
            store=terminal_store,
        )


def test_sqlite_checkpoint_store_still_wires_durable_terminal(tmp_path) -> None:
    sqlite_store = SQLiteTaskCheckpointStore(db_path=tmp_path / "ckpt.db")
    wired = wire_execution_terminal_store(checkpoint_store=sqlite_store)
    assert isinstance(wired, AdapterClass)
    assert wired.is_durable is True


@pytest.mark.asyncio
async def test_governed_resume_uses_explicit_terminal_without_runner_nexus() -> None:
    checkpoint = _paused_checkpoint()
    terminal = ExecutionTerminalService(InMemoryExecutionTerminalStore())
    terminal.record_cancellation(
        tenant_id=checkpoint.tenant_id,
        task_id=checkpoint.task_id,
        reason="operator_cancel",
    )
    runner = AsyncMock()

    boundary = ControlPlaneMutationAuthorizationBoundary(evaluator=_AllowEvaluator())
    with patch(
        "intergrax.applications._shared.task_control._resume_task_with_token",
        new_callable=AsyncMock,
    ) as resume_call:
        outcome = await governed_resume_checkpoint_task(
            runner,
            task_id=checkpoint.task_id,
            tenant_id=checkpoint.tenant_id,
            resume_token=checkpoint.resume_token,
            mutation_id="mut-explicit",
            principal=RequestIdentity(
                tenant_id=_TENANT,
                user_id="operator-1",
                principal_type=PrincipalType.USER,
                auth_subject="operator-1",
            ),
            mutation_boundary=boundary,
            checkpoint_store=_StaticCheckpointStore(checkpoint),
            execution_terminal=terminal,
        )
    assert outcome.accepted is False
    assert outcome.blocked is not None
    assert outcome.blocked.detail == "execution_terminally_cancelled"
    resume_call.assert_not_called()


@pytest.mark.asyncio
async def test_governed_resume_without_terminal_record_continues_flow() -> None:
    checkpoint = _paused_checkpoint()
    terminal = ExecutionTerminalService(InMemoryExecutionTerminalStore())
    runner = AsyncMock()
    boundary = ControlPlaneMutationAuthorizationBoundary(evaluator=_AllowEvaluator())

    with patch(
        "intergrax.applications._shared.task_control._resume_task_with_token",
        new_callable=AsyncMock,
        return_value=MagicMock(),
    ) as resume_call:
        outcome = await governed_resume_checkpoint_task(
            runner,
            task_id=checkpoint.task_id,
            tenant_id=checkpoint.tenant_id,
            resume_token=checkpoint.resume_token,
            mutation_id="mut-no-terminal",
            principal=RequestIdentity(
                tenant_id=_TENANT,
                user_id="operator-1",
                principal_type=PrincipalType.USER,
                auth_subject="operator-1",
            ),
            mutation_boundary=boundary,
            checkpoint_store=_StaticCheckpointStore(checkpoint),
            execution_terminal=terminal,
        )
    assert outcome.accepted is True
    resume_call.assert_called_once()


def test_resolve_harness_task_control_execution_terminal_prefers_explicit() -> None:
    explicit = ExecutionTerminalService(InMemoryExecutionTerminalStore())
    runtime = MagicMock()
    resolved = resolve_harness_task_control_execution_terminal(
        runtime=runtime,
        execution_terminal=explicit,
    )
    assert resolved is explicit
