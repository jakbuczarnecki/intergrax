# © Artur Czarnecki. All rights reserved.

"""UCA-6C-R6-R5.9-R2-R1-R2 — caller-held claim authority through canonical resume."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Iterator

import pytest

from intergrax.contracts.execution.execution_terminal_outcome_by_execution_id import (
    ExecutionTerminalOutcomeByExecutionIdDisposition,
)
from intergrax.contracts.execution.suspended_operation.claim_authority import (
    SuspendedOperationClaimAuthority,
)
from intergrax.contracts.execution.suspended_operation.reentry import (
    ExecutionSuspendedWorkReentryDisposition,
)
from intergrax.contracts.execution.suspended_operation.descriptor import (
    SuspendedOperationMaterializationState,
)
from intergrax.contracts.execution_continuation import ExecutionContinuationLookup
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    reset_active_execution_identity,
)
from intergrax.contracts.human_approver import local_development_approver_evidence
from intergrax.runtime.execution.suspended_operation.authorized_resume_reentry import (
    SuspendedOperationClaimAuthorityResumeTelemetry,
    resume_authorized_continuation_with_suspended_work_reentry,
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
from intergrax.runtime.human.agent_governance_human_approval_grant import (
    AgentGovernanceHumanApprovalGrantCoordinator,
)
from intergrax.runtime.human.declarative_hitl_grant import (
    DeclarativeHitlGrantCoordinator,
)
from intergrax.runtime.human.governed_continuation_grant import (
    GovernedContinuationGrantCoordinator,
)
from intergrax.runtime.human.models import HumanResponseVerdict
from intergrax.runtime.human.pause import HumanPauseCoordinator
from intergrax.runtime.nexus.orchestration.internal_continuation_orchestration import (
    InternalOrchestrationContinuation,
)
from intergrax.runtime.task.task import Task
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
    _claim_host_a_short_lease,
    _sync_host_stores,
    _ReclaimBridge,
)

pytestmark = pytest.mark.unit

_AUTHORIZED_RESUME_REENTRY_PATH = (
    Path(__file__).resolve().parents[5]
    / "intergrax"
    / "runtime"
    / "execution"
    / "suspended_operation"
    / "authorized_resume_reentry.py"
)


@contextmanager
def _multi_host_fixture(tmp_path: Path) -> Iterator[DualHostReentryFixture]:
    yield _build_dual_host_fixture(tmp_path)


def _authorize_gen3(
    task: Task,
    *,
    hitl: InternalOrchestrationContinuation,
    continuation_id: str,
    run_id,
    attempt_id,
    execution_id,
    checkpoint_store,
):
    pending = hitl.port.get_pending(
        ExecutionContinuationLookup(continuation_id=continuation_id),
    )
    pause_record = task.runtime.governance.pause_record
    human_request = task.runtime.governance.human_request
    assert pause_record is not None and human_request is not None
    if pending.governed_correlation is not None:
        task.runtime.governance.human_request = human_request.model_copy(
            update={"governed_continuation": pending.governed_correlation},
        )
    approver = local_development_approver_evidence(tenant_id=task.tenant_id)
    authorized = HumanPauseCoordinator.resolve_human_response_and_apply_canonical(
        task,
        HumanResponseVerdict.APPROVE,
        approver=approver,
        continuation=hitl.port,
        pause_id=pause_record.pause_id,
        human_request_id=human_request.request_id,
        run_id=str(run_id),
        attempt_id=str(attempt_id),
        execution_id=str(execution_id),
    )
    declarative_pending = task.runtime.governance.declarative_hitl_pending
    if (
        declarative_pending is not None
        and declarative_pending.pause_id == pause_record.pause_id
    ):
        DeclarativeHitlGrantCoordinator.create_grant_from_pending(task)
    agent_pending = task.runtime.governance.agent_governance_hitl_pending
    if (
        agent_pending is not None
        and checkpoint_store is not None
        and agent_pending.pause_id == pause_record.pause_id
    ):
        AgentGovernanceHumanApprovalGrantCoordinator.persist_available_grant_from_human_approve(
            task,
            checkpoint_store=checkpoint_store,
            approver=approver,
        )
    if task.runtime.governance.human_request is not None:
        GovernedContinuationGrantCoordinator.create_grant_from_approval(task)
    return authorized


def _canonical_resume_reentry(
    task: Task,
    *,
    hitl: InternalOrchestrationContinuation,
    continuation_id: str,
    run_id,
    attempt_id,
    execution_id,
    checkpoint_store,
    claim_authority: SuspendedOperationClaimAuthority | None = None,
    telemetry: SuspendedOperationClaimAuthorityResumeTelemetry | None = None,
):
    authorized = _authorize_gen3(
        task,
        hitl=hitl,
        continuation_id=continuation_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        checkpoint_store=checkpoint_store,
    )
    return resume_authorized_continuation_with_suspended_work_reentry(
        task,
        authorized,
        capability=hitl,
        reentry_coordinator=hitl.suspended_work_reentry_coordinator,
        claim_authority=claim_authority,
        authority_telemetry=telemetry,
    )


def test_static_canonical_resume_does_not_refresh_claimed_authority_from_store() -> (
    None
):
    source = _AUTHORIZED_RESUME_REENTRY_PATH.read_text(encoding="utf-8")
    assert "from_claimed_descriptor" not in source


def test_same_owner_stale_fence_canonical_resume_rejected(tmp_path: Path) -> None:
    telemetry = SuspendedOperationClaimAuthorityResumeTelemetry()
    with _multi_host_fixture(tmp_path) as fixture:
        run_id = mint_run_id()
        attempt_id = mint_attempt_id()
        execution_id = mint_execution_id()
        task = Task(
            tenant_id=_TENANT, user_id="u1", message="r59r2r1r2-aba", task_id=_TASK_ID
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
        try:
            c3, d3 = _advance_to_gen3_blocked(
                fixture,
                task,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
            )
            store = fixture.composition_b.suspended_work_reentry_coordinator.store
            lease_at = datetime.now(UTC) + timedelta(minutes=5)
            claimed_b = store.claim(
                suspended_operation_id=d3.suspended_operation_id,
                expected_materialization_revision=d3.materialization_revision,
                owner_id=OWNER_HOST_B,
                lease_expires_at=lease_at,
            )
            assert claimed_b.descriptor is not None
            assert claimed_b.descriptor.claim_ownership is not None
            retained = SuspendedOperationClaimAuthority.from_claimed_descriptor(
                claimed_b.descriptor,
            )
            fence_gen1 = retained.fence
            revision_mid = retained.materialization_revision
            pause_gen = retained.pause_generation

            expired_now = datetime.now(UTC) + timedelta(hours=2)
            reclaim_bridge = _ReclaimBridge(
                descriptor=d3,
                store_a=fixture.composition_a.suspended_work_reentry_coordinator.store,
                store_b=store,
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
            assert reclaimed.descriptor.claim_ownership is not None
            assert reclaimed.descriptor.claim_ownership.fence > fence_gen1
            stale_same_owner = SuspendedOperationClaimAuthority(
                owner_id=OWNER_HOST_B,
                fence=fence_gen1,
                materialization_revision=reclaimed.descriptor.materialization_revision,
                pause_generation=pause_gen,
            )

            _sync_host_stores(fixture)
            _, result = _canonical_resume_reentry(
                task,
                hitl=fixture.hitl_b,
                continuation_id=c3,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
                checkpoint_store=fixture.checkpoint_store,
                claim_authority=stale_same_owner,
                telemetry=telemetry,
            )
            assert (
                result is not None
                and result.disposition
                is ExecutionSuspendedWorkReentryDisposition.FAILED
            )
            assert result.reason_detail == "stale_claim_fence"
            assert fixture.backend_a.calls == 0
            assert fixture.backend_b.calls == 0
            assert fixture.counters.terminal_writes == 0
            assert telemetry.authority_refreshes_from_store == 0
            assert telemetry.claim_authority_passed == 1
        finally:
            reset_governed_execution_task(task_token)
            reset_active_execution_governance_identity(gov_token)
            reset_active_execution_identity(id_token)


def test_same_owner_current_fence_canonical_resume_completes(tmp_path: Path) -> None:
    telemetry = SuspendedOperationClaimAuthorityResumeTelemetry()
    with _multi_host_fixture(tmp_path) as fixture:
        run_id = mint_run_id()
        attempt_id = mint_attempt_id()
        execution_id = mint_execution_id()
        task = Task(
            tenant_id=_TENANT, user_id="u1", message="r59r2r1r2-ok", task_id=_TASK_ID
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
        try:
            c3, d3 = _advance_to_gen3_blocked(
                fixture,
                task,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
            )
            store = fixture.composition_b.suspended_work_reentry_coordinator.store
            lease_at = datetime.now(UTC) + timedelta(minutes=5)
            claimed_b = store.claim(
                suspended_operation_id=d3.suspended_operation_id,
                expected_materialization_revision=d3.materialization_revision,
                owner_id=OWNER_HOST_B,
                lease_expires_at=lease_at,
            )
            assert claimed_b.descriptor is not None
            fence_gen1 = claimed_b.descriptor.claim_ownership.fence
            revision_mid = claimed_b.descriptor.materialization_revision

            expired_now = datetime.now(UTC) + timedelta(hours=2)
            reclaim_bridge = _ReclaimBridge(
                descriptor=d3,
                store_a=fixture.composition_a.suspended_work_reentry_coordinator.store,
                store_b=store,
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
            current = SuspendedOperationClaimAuthority.from_claimed_descriptor(
                reclaimed.descriptor,
            )
            _sync_host_stores(fixture)
            _, result = _canonical_resume_reentry(
                task,
                hitl=fixture.hitl_b,
                continuation_id=c3,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
                checkpoint_store=fixture.checkpoint_store,
                claim_authority=current,
                telemetry=telemetry,
            )
            assert (
                result is not None
                and result.disposition
                is ExecutionSuspendedWorkReentryDisposition.COMPLETED
            )
            assert fixture.backend_b.calls == 1
            assert fixture.counters.terminal_writes == 1
            terminal = fixture.terminal_store.get_recorded_disposition(execution_id)
            assert (
                terminal is ExecutionTerminalOutcomeByExecutionIdDisposition.SUCCEEDED
            )
            assert telemetry.authority_refreshes_from_store == 0
            assert telemetry.claim_authority_passed == 1
        finally:
            reset_governed_execution_task(task_token)
            reset_active_execution_governance_identity(gov_token)
            reset_active_execution_identity(id_token)


def test_missing_claim_authority_fail_closed(tmp_path: Path) -> None:
    with _multi_host_fixture(tmp_path) as fixture:
        run_id = mint_run_id()
        attempt_id = mint_attempt_id()
        execution_id = mint_execution_id()
        task = Task(
            tenant_id=_TENANT, user_id="u1", message="r59r2r1r2-miss", task_id=_TASK_ID
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
        try:
            c3, d3 = _advance_to_gen3_blocked(
                fixture,
                task,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
            )
            store = fixture.composition_b.suspended_work_reentry_coordinator.store
            lease_at = datetime.now(UTC) + timedelta(minutes=5)
            store.claim(
                suspended_operation_id=d3.suspended_operation_id,
                expected_materialization_revision=d3.materialization_revision,
                owner_id=OWNER_HOST_B,
                lease_expires_at=lease_at,
            )
            _sync_host_stores(fixture)
            _, result = _canonical_resume_reentry(
                task,
                hitl=fixture.hitl_b,
                continuation_id=c3,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
                checkpoint_store=fixture.checkpoint_store,
                claim_authority=None,
            )
            assert (
                result is not None
                and result.disposition
                is ExecutionSuspendedWorkReentryDisposition.FAILED
            )
            assert result.reason_detail == "missing_caller_claim_authority"
            assert fixture.backend_b.calls == 0
            assert fixture.counters.terminal_writes == 0
        finally:
            reset_governed_execution_task(task_token)
            reset_active_execution_governance_identity(gov_token)
            reset_active_execution_identity(id_token)


@pytest.mark.parametrize(
    ("field_name", "mutator", "expected_detail"),
    [
        (
            "revision",
            lambda a, d: a.model_copy(
                update={"materialization_revision": d.materialization_revision - 1},
            ),
            "stale_materialization_revision",
        ),
        (
            "pause_generation",
            lambda a, _: a.model_copy(
                update={"pause_generation": a.pause_generation - 1}
            ),
            "stale_pause_generation",
        ),
        (
            "owner",
            lambda a, _: a.model_copy(update={"owner_id": OWNER_HOST_A}),
            "claim_owner_mismatch",
        ),
    ],
)
def test_stale_authority_fields_through_canonical_resume(
    tmp_path: Path,
    field_name: str,
    mutator,
    expected_detail: str,
) -> None:
    with _multi_host_fixture(tmp_path) as fixture:
        run_id = mint_run_id()
        attempt_id = mint_attempt_id()
        execution_id = mint_execution_id()
        task = Task(
            tenant_id=_TENANT,
            user_id="u1",
            message=f"r59r2r1r2-{field_name}",
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
        try:
            c3, d3 = _advance_to_gen3_blocked(
                fixture,
                task,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
            )
            fence_a, revision_a, _ = _claim_host_a_short_lease(fixture, d3)
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
                    expected_revision=revision_a,
                    owner_id=OWNER_HOST_B,
                    expected_fence=fence_a,
                    lease_at=expired_now + timedelta(minutes=5),
                )
            assert reclaimed.descriptor is not None
            current = SuspendedOperationClaimAuthority.from_claimed_descriptor(
                reclaimed.descriptor,
            )
            stale = mutator(current, reclaimed.descriptor)
            _sync_host_stores(fixture)
            _, result = _canonical_resume_reentry(
                task,
                hitl=fixture.hitl_b,
                continuation_id=c3,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
                checkpoint_store=fixture.checkpoint_store,
                claim_authority=stale,
            )
            assert result is not None
            assert result.disposition in {
                ExecutionSuspendedWorkReentryDisposition.FAILED,
                ExecutionSuspendedWorkReentryDisposition.REJECTED,
            }
            assert result.reason_detail == expected_detail
            assert fixture.backend_b.calls == 0
            assert fixture.counters.terminal_writes == 0
        finally:
            reset_governed_execution_task(task_token)
            reset_active_execution_governance_identity(gov_token)
            reset_active_execution_identity(id_token)


def test_expired_lease_with_current_authority_canonical_resume(tmp_path: Path) -> None:
    with _multi_host_fixture(tmp_path) as fixture:
        run_id = mint_run_id()
        attempt_id = mint_attempt_id()
        execution_id = mint_execution_id()
        task = Task(
            tenant_id=_TENANT, user_id="u1", message="r59r2r1r2-lease", task_id=_TASK_ID
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
        try:
            c3, d3 = _advance_to_gen3_blocked(
                fixture,
                task,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
            )
            store = fixture.composition_b.suspended_work_reentry_coordinator.store
            short_lease = datetime.now(UTC) + timedelta(seconds=30)
            claimed = store.claim(
                suspended_operation_id=d3.suspended_operation_id,
                expected_materialization_revision=d3.materialization_revision,
                owner_id=OWNER_HOST_B,
                lease_expires_at=short_lease,
            )
            assert claimed.descriptor is not None
            retained = SuspendedOperationClaimAuthority.from_claimed_descriptor(
                claimed.descriptor,
            )
            expired_now = datetime.now(UTC) + timedelta(hours=2)
            _sync_host_stores(fixture)
            with advance_lease_clock(expired_now):
                _, result = _canonical_resume_reentry(
                    task,
                    hitl=fixture.hitl_b,
                    continuation_id=c3,
                    run_id=run_id,
                    attempt_id=attempt_id,
                    execution_id=execution_id,
                    checkpoint_store=fixture.checkpoint_store,
                    claim_authority=retained,
                )
            assert (
                result is not None
                and result.disposition
                is ExecutionSuspendedWorkReentryDisposition.FAILED
            )
            assert result.reason_detail == "claim_lease_expired"
            assert fixture.backend_b.calls == 0
            assert fixture.counters.terminal_writes == 0
        finally:
            reset_governed_execution_task(task_token)
            reset_active_execution_governance_identity(gov_token)
            reset_active_execution_identity(id_token)


def test_pre_claim_canonical_resume_without_caller_authority(tmp_path: Path) -> None:
    telemetry = SuspendedOperationClaimAuthorityResumeTelemetry()
    with _multi_host_fixture(tmp_path) as fixture:
        run_id = mint_run_id()
        attempt_id = mint_attempt_id()
        execution_id = mint_execution_id()
        task = Task(
            tenant_id=_TENANT, user_id="u1", message="r59r2r1r2-pre", task_id=_TASK_ID
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
        try:
            c3, d3 = _advance_to_gen3_blocked(
                fixture,
                task,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
            )
            _sync_host_stores(fixture)
            _, result = _canonical_resume_reentry(
                task,
                hitl=fixture.hitl_b,
                continuation_id=c3,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
                checkpoint_store=fixture.checkpoint_store,
                claim_authority=None,
                telemetry=telemetry,
            )
            assert (
                result is not None
                and result.disposition
                is ExecutionSuspendedWorkReentryDisposition.COMPLETED
            )
            assert fixture.backend_b.calls == 1
            assert telemetry.claim_authority_created == 1
            assert telemetry.authority_refreshes_from_store == 0
            consumed = (
                fixture.composition_b.suspended_work_reentry_coordinator.store.load(
                    d3.suspended_operation_id,
                )
            )
            assert consumed is not None
            assert (
                consumed.materialization_state
                is SuspendedOperationMaterializationState.CONSUMED
            )
        finally:
            reset_governed_execution_task(task_token)
            reset_active_execution_governance_identity(gov_token)
            reset_active_execution_identity(id_token)
