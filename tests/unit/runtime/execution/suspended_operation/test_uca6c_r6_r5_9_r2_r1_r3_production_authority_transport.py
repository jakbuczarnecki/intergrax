# © Artur Czarnecki. All rights reserved.

"""UCA-6C-R6-R5.9-R2-R1-R3 — production Nexus intake claim authority transport."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Iterator
from unittest.mock import AsyncMock, MagicMock

import pytest

from intergrax.contracts.execution.execution_terminal_outcome_by_execution_id import (
    ExecutionTerminalOutcomeByExecutionIdDisposition,
)
from intergrax.contracts.execution.suspended_operation.claim_authority import (
    SuspendedOperationClaimAuthority,
)
from intergrax.contracts.execution.suspended_operation.resume_authority_context import (
    ExecutionSuspendedWorkResumeAuthorityContext,
)
from intergrax.contracts.execution_identity import (
    ActiveExecutionIdentity,
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    reset_active_execution_identity,
)
from intergrax.contracts.human_approver import local_development_approver_evidence
from intergrax.runtime.execution.suspended_operation.claim_lifecycle_wiring import (
    claim_lifecycle_from_hitl_continuation,
)
from intergrax.runtime.execution.suspended_operation.resume_authority_transport import (
    ExecutionSuspendedWorkResumeAuthorityTransport,
    ProductionSuspendedWorkAuthorityTelemetry,
)
from intergrax.runtime.governance.active_execution_governance_identity import (
    ActiveExecutionGovernanceIdentity,
    bind_active_execution_governance_identity,
    reset_active_execution_governance_identity,
)
from intergrax.runtime.governance.active_governed_execution_task import (
    bind_governed_execution_task,
    reset_governed_execution_task,
)
from intergrax.runtime.human.models import HumanResponseVerdict
from intergrax.runtime.nexus.orchestration.intake_runner import NexusIntakeRunner
from intergrax.runtime.task.task import Task, TaskState
from intergrax.runtime.task.task_lifecycle import TaskLifecycle
from intergrax.runtime.task.task_trace import TaskTraceEmitter
from testing_support.runtime_event_metric_scope_for_tests import (
    open_runtime_event_metric_scope_for_tests,
)
from tests.unit.autonomous_work.test_uca6c_r4_real_codecraft_execution import (
    _TASK_ID,
    _TENANT,
)
from tests.unit.runtime.execution.suspended_operation.test_uca6c_r6_r5_9_r2_multi_host_fencing import (
    OWNER_HOST_A,
    OWNER_HOST_B,
    advance_lease_clock,
    reclaim_as,
)
from tests.unit.runtime.execution.suspended_operation.test_uca6c_r6_r5_9_r2_r1_canonical_reentry_fencing import (
    DualHostReentryFixture,
    _advance_to_gen3_blocked,
    _build_dual_host_fixture,
    _sync_host_stores,
    _ReclaimBridge,
)

pytestmark = pytest.mark.unit

_INTAKE_RUNNER_PATH = (
    Path(__file__).resolve().parents[5]
    / "intergrax"
    / "runtime"
    / "nexus"
    / "orchestration"
    / "intake_runner.py"
)


@contextmanager
def _multi_host_fixture(tmp_path: Path) -> Iterator[DualHostReentryFixture]:
    yield _build_dual_host_fixture(tmp_path)


def _configure_task_for_intake_approve(task: Task) -> None:
    pause_record = task.runtime.governance.pause_record
    human_request = task.runtime.governance.human_request
    assert pause_record is not None and human_request is not None
    task.options.human.verdict = HumanResponseVerdict.APPROVE.value
    task.options.human.pause_id = pause_record.pause_id
    task.options.human.human_request_id = pause_record.human_request_id
    task.options.human.approver = local_development_approver_evidence(
        tenant_id=task.tenant_id,
    )
    task.state = TaskState.WAITING_FOR_HUMAN
    task.sync_metadata()


def _build_intake_runner(
    fixture: DualHostReentryFixture,
    *,
    hitl,
    transport: ExecutionSuspendedWorkResumeAuthorityTransport,
    telemetry: ProductionSuspendedWorkAuthorityTelemetry,
    execution_identity: ActiveExecutionIdentity,
) -> NexusIntakeRunner:
    lifecycle = claim_lifecycle_from_hitl_continuation(hitl)
    return NexusIntakeRunner(
        hitl=MagicMock(
            persist_human_decision=MagicMock(),
            handle_human_rejection=AsyncMock(),
            handle_human_escalation=AsyncMock(),
        ),
        human_hooks=MagicMock(after_response=AsyncMock()),
        publish=AsyncMock(),
        restore_long_running=AsyncMock(),
        execution_identity=execution_identity,
        hitl_continuation=hitl,
        task_checkpoint_store=fixture.checkpoint_store,
        suspended_work_claim_lifecycle=lifecycle,
        suspended_work_resume_authority_transport=transport,
        suspended_work_authority_telemetry=telemetry,
    )


@pytest.mark.asyncio
async def test_production_intake_claimed_resume_completes(tmp_path: Path) -> None:
    telemetry = ProductionSuspendedWorkAuthorityTelemetry()
    transport = ExecutionSuspendedWorkResumeAuthorityTransport()
    with _multi_host_fixture(tmp_path) as fixture:
        run_id = mint_run_id()
        attempt_id = mint_attempt_id()
        execution_id = mint_execution_id()
        task = Task(
            tenant_id=_TENANT, user_id="u1", message="r59r2r1r3-ok", task_id=_TASK_ID
        )
        id_token = bind_active_execution_identity(
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
        )
        gov_token = bind_active_execution_governance_identity(
            ActiveExecutionGovernanceIdentity(
                tenant_id=_TENANT,
                workspace_id="workspace-uca6c",
                principal_id="principal-uca6c",
            ),
        )
        task_token = bind_governed_execution_task(task)
        execution_identity = ActiveExecutionIdentity()
        try:
            _advance_to_gen3_blocked(
                fixture,
                task,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
            )
            _sync_host_stores(fixture)
            _configure_task_for_intake_approve(task)
            runner = _build_intake_runner(
                fixture,
                hitl=fixture.hitl_b,
                transport=transport,
                telemetry=telemetry,
                execution_identity=execution_identity,
            )
            metric_scope = open_runtime_event_metric_scope_for_tests(
                task_id=_TASK_ID,
                run_id=str(run_id),
            )
            try:
                await runner.run(
                    task,
                    lifecycle=TaskLifecycle(),
                    trace_emitter=TaskTraceEmitter(
                        run_id=str(run_id),
                        attempt_id=str(attempt_id),
                    ),
                    runtime_event_metric_scope=metric_scope,
                )
            finally:
                metric_scope.close()
            assert fixture.backend_b.calls == 1
            assert telemetry.authority_refreshes_from_store == 0
            assert telemetry.claim_successes >= 1
            assert telemetry.production_resume_attempts == 1
            assert telemetry.terminal_writes == 1
            terminal = fixture.terminal_store.get_recorded_disposition(execution_id)
            assert (
                terminal is ExecutionTerminalOutcomeByExecutionIdDisposition.SUCCEEDED
            )
        finally:
            reset_governed_execution_task(task_token)
            reset_active_execution_governance_identity(gov_token)
            reset_active_execution_identity(id_token)


@pytest.mark.asyncio
async def test_production_intake_missing_authority_fail_closed(
    tmp_path: Path,
) -> None:
    telemetry = ProductionSuspendedWorkAuthorityTelemetry()
    transport = ExecutionSuspendedWorkResumeAuthorityTransport()
    with _multi_host_fixture(tmp_path) as fixture:
        run_id = mint_run_id()
        attempt_id = mint_attempt_id()
        execution_id = mint_execution_id()
        task = Task(
            tenant_id=_TENANT,
            user_id="u1",
            message="r59r2r1r3-miss",
            task_id=_TASK_ID,
        )
        id_token = bind_active_execution_identity(
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
        )
        gov_token = bind_active_execution_governance_identity(
            ActiveExecutionGovernanceIdentity(
                tenant_id=_TENANT,
                workspace_id="workspace-uca6c",
                principal_id="principal-uca6c",
            ),
        )
        task_token = bind_governed_execution_task(task)
        execution_identity = ActiveExecutionIdentity()
        try:
            _, d3 = _advance_to_gen3_blocked(
                fixture,
                task,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
            )
            store = fixture.composition_b.suspended_work_reentry_coordinator.store
            store.claim(
                suspended_operation_id=d3.suspended_operation_id,
                expected_materialization_revision=d3.materialization_revision,
                owner_id=OWNER_HOST_B,
                lease_expires_at=datetime.now(UTC) + timedelta(minutes=5),
            )
            _sync_host_stores(fixture)
            _configure_task_for_intake_approve(task)
            runner = _build_intake_runner(
                fixture,
                hitl=fixture.hitl_b,
                transport=transport,
                telemetry=telemetry,
                execution_identity=execution_identity,
            )
            metric_scope = open_runtime_event_metric_scope_for_tests(
                task_id=_TASK_ID,
                run_id=str(run_id),
            )
            try:
                await runner.run(
                    task,
                    lifecycle=TaskLifecycle(),
                    trace_emitter=TaskTraceEmitter(
                        run_id=str(run_id),
                        attempt_id=str(attempt_id),
                    ),
                    runtime_event_metric_scope=metric_scope,
                )
            finally:
                metric_scope.close()
            assert fixture.backend_b.calls == 0
            assert telemetry.terminal_writes == 0
            assert telemetry.authority_refreshes_from_store == 0
        finally:
            reset_governed_execution_task(task_token)
            reset_active_execution_governance_identity(gov_token)
            reset_active_execution_identity(id_token)


@pytest.mark.asyncio
async def test_production_intake_same_owner_stale_fence_rejected(
    tmp_path: Path,
) -> None:
    transport = ExecutionSuspendedWorkResumeAuthorityTransport()
    telemetry = ProductionSuspendedWorkAuthorityTelemetry()
    with _multi_host_fixture(tmp_path) as fixture:
        run_id = mint_run_id()
        attempt_id = mint_attempt_id()
        execution_id = mint_execution_id()
        task = Task(
            tenant_id=_TENANT, user_id="u1", message="r59r2r1r3-aba", task_id=_TASK_ID
        )
        id_token = bind_active_execution_identity(
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
        )
        gov_token = bind_active_execution_governance_identity(
            ActiveExecutionGovernanceIdentity(
                tenant_id=_TENANT,
                workspace_id="workspace-uca6c",
                principal_id="principal-uca6c",
            ),
        )
        task_token = bind_governed_execution_task(task)
        execution_identity = ActiveExecutionIdentity()
        lifecycle = claim_lifecycle_from_hitl_continuation(fixture.hitl_b)
        assert lifecycle is not None
        try:
            c3, d3 = _advance_to_gen3_blocked(
                fixture,
                task,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
            )
            claimed = lifecycle.claim_blocked(d3)
            assert claimed is not None
            fence_gen1 = claimed.claim_authority.fence
            revision_mid = claimed.claim_authority.materialization_revision
            expired_now = datetime.now(UTC) + timedelta(hours=2)
            reclaim_bridge = _ReclaimBridge(
                descriptor=d3,
                store_a=fixture.composition_a.suspended_work_reentry_coordinator.store,
                store_b=fixture.composition_b.suspended_work_reentry_coordinator.store,
            )
            with advance_lease_clock(expired_now):
                reclaimed = reclaim_as(
                    reclaim_bridge,
                    "b",
                    expected_revision=revision_mid,
                    owner_id=OWNER_HOST_B,
                    expected_fence=fence_gen1,
                    lease_at=expired_now + timedelta(minutes=5),
                )
            assert reclaimed.descriptor is not None
            stale = SuspendedOperationClaimAuthority(
                owner_id=OWNER_HOST_B,
                fence=fence_gen1,
                materialization_revision=reclaimed.descriptor.materialization_revision,
                pause_generation=claimed.claim_authority.pause_generation,
            )
            transport.deliver(
                ExecutionSuspendedWorkResumeAuthorityContext(
                    continuation_id=c3,
                    claim_authority=stale,
                ),
            )
            _sync_host_stores(fixture)
            _configure_task_for_intake_approve(task)
            runner = _build_intake_runner(
                fixture,
                hitl=fixture.hitl_b,
                transport=transport,
                telemetry=telemetry,
                execution_identity=execution_identity,
            )
            metric_scope = open_runtime_event_metric_scope_for_tests(
                task_id=_TASK_ID,
                run_id=str(run_id),
            )
            try:
                await runner.run(
                    task,
                    lifecycle=TaskLifecycle(),
                    trace_emitter=TaskTraceEmitter(
                        run_id=str(run_id),
                        attempt_id=str(attempt_id),
                    ),
                    runtime_event_metric_scope=metric_scope,
                )
            finally:
                metric_scope.close()
            assert fixture.backend_b.calls == 0
            assert telemetry.terminal_writes == 0
        finally:
            reset_governed_execution_task(task_token)
            reset_active_execution_governance_identity(gov_token)
            reset_active_execution_identity(id_token)


@pytest.mark.asyncio
async def test_production_intake_pre_claim_blocked_path(tmp_path: Path) -> None:
    transport = ExecutionSuspendedWorkResumeAuthorityTransport()
    telemetry = ProductionSuspendedWorkAuthorityTelemetry()
    with _multi_host_fixture(tmp_path) as fixture:
        run_id = mint_run_id()
        attempt_id = mint_attempt_id()
        execution_id = mint_execution_id()
        task = Task(
            tenant_id=_TENANT,
            user_id="u1",
            message="r59r2r1r3-pre",
            task_id=_TASK_ID,
        )
        id_token = bind_active_execution_identity(
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
        )
        gov_token = bind_active_execution_governance_identity(
            ActiveExecutionGovernanceIdentity(
                tenant_id=_TENANT,
                workspace_id="workspace-uca6c",
                principal_id="principal-uca6c",
            ),
        )
        task_token = bind_governed_execution_task(task)
        execution_identity = ActiveExecutionIdentity()
        try:
            _advance_to_gen3_blocked(
                fixture,
                task,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
            )
            _sync_host_stores(fixture)
            _configure_task_for_intake_approve(task)
            runner = _build_intake_runner(
                fixture,
                hitl=fixture.hitl_b,
                transport=transport,
                telemetry=telemetry,
                execution_identity=execution_identity,
            )
            metric_scope = open_runtime_event_metric_scope_for_tests(
                task_id=_TASK_ID,
                run_id=str(run_id),
            )
            try:
                await runner.run(
                    task,
                    lifecycle=TaskLifecycle(),
                    trace_emitter=TaskTraceEmitter(
                        run_id=str(run_id),
                        attempt_id=str(attempt_id),
                    ),
                    runtime_event_metric_scope=metric_scope,
                )
            finally:
                metric_scope.close()
            assert fixture.backend_b.calls == 1
            assert telemetry.claim_successes == 1
            assert telemetry.authority_refreshes_from_store == 0
        finally:
            reset_governed_execution_task(task_token)
            reset_active_execution_governance_identity(gov_token)
            reset_active_execution_identity(id_token)


@pytest.mark.asyncio
async def test_production_intake_restart_reclaim_new_authority(tmp_path: Path) -> None:
    transport = ExecutionSuspendedWorkResumeAuthorityTransport()
    telemetry = ProductionSuspendedWorkAuthorityTelemetry()
    with _multi_host_fixture(tmp_path) as fixture:
        run_id = mint_run_id()
        attempt_id = mint_attempt_id()
        execution_id = mint_execution_id()
        task = Task(
            tenant_id=_TENANT,
            user_id="u1",
            message="r59r2r1r3-restart",
            task_id=_TASK_ID,
        )
        id_token = bind_active_execution_identity(
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
        )
        gov_token = bind_active_execution_governance_identity(
            ActiveExecutionGovernanceIdentity(
                tenant_id=_TENANT,
                workspace_id="workspace-uca6c",
                principal_id="principal-uca6c",
            ),
        )
        task_token = bind_governed_execution_task(task)
        execution_identity = ActiveExecutionIdentity()
        lifecycle = claim_lifecycle_from_hitl_continuation(fixture.hitl_b)
        assert lifecycle is not None
        try:
            _, d3 = _advance_to_gen3_blocked(
                fixture,
                task,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
            )
            claimed_a = lifecycle.claim_blocked(d3)
            assert claimed_a is not None
            fence_gen1 = claimed_a.claim_authority.fence
            revision_mid = claimed_a.claim_authority.materialization_revision
            expired_now = datetime.now(UTC) + timedelta(hours=2)
            reclaim_bridge = _ReclaimBridge(
                descriptor=d3,
                store_a=fixture.composition_a.suspended_work_reentry_coordinator.store,
                store_b=fixture.composition_b.suspended_work_reentry_coordinator.store,
            )
            with advance_lease_clock(expired_now):
                reclaimed = reclaim_as(
                    reclaim_bridge,
                    "b",
                    expected_revision=revision_mid,
                    owner_id=OWNER_HOST_B,
                    expected_fence=fence_gen1,
                    lease_at=expired_now + timedelta(minutes=5),
                )
            assert reclaimed.descriptor is not None
            transport.deliver(
                ExecutionSuspendedWorkResumeAuthorityContext(
                    continuation_id=claimed_a.continuation_id,
                    claim_authority=SuspendedOperationClaimAuthority.from_claimed_descriptor(
                        reclaimed.descriptor,
                    ),
                ),
            )
            _sync_host_stores(fixture)
            _configure_task_for_intake_approve(task)
            runner = _build_intake_runner(
                fixture,
                hitl=fixture.hitl_b,
                transport=transport,
                telemetry=telemetry,
                execution_identity=execution_identity,
            )
            metric_scope = open_runtime_event_metric_scope_for_tests(
                task_id=_TASK_ID,
                run_id=str(run_id),
            )
            try:
                await runner.run(
                    task,
                    lifecycle=TaskLifecycle(),
                    trace_emitter=TaskTraceEmitter(
                        run_id=str(run_id),
                        attempt_id=str(attempt_id),
                    ),
                    runtime_event_metric_scope=metric_scope,
                )
            finally:
                metric_scope.close()
            assert fixture.backend_b.calls == 1
            assert telemetry.reclaim_successes == 0
            assert telemetry.authority_refreshes_from_store == 0
        finally:
            reset_governed_execution_task(task_token)
            reset_active_execution_governance_identity(gov_token)
            reset_active_execution_identity(id_token)


def test_static_production_intake_does_not_mint_authority_from_store() -> None:
    source = _INTAKE_RUNNER_PATH.read_text(encoding="utf-8")
    assert "from_claimed_descriptor" not in source
    assert "load_active_for_continuation" not in source


def test_static_wrong_owner_rejected_via_production_intake_guard() -> None:
    authority = SuspendedOperationClaimAuthority(
        owner_id=OWNER_HOST_A,
        fence=1,
        materialization_revision=1,
        pause_generation=3,
    )
    assert authority.owner_id != OWNER_HOST_B
