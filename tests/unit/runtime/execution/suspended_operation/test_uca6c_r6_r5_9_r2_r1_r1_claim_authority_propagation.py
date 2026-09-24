# © Artur Czarnecki. All rights reserved.

"""UCA-6C-R6-R5.9-R2-R1-R1 — claim authority propagation through canonical reentry."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Iterator

import pytest

from intergrax.contracts.execution.suspended_operation.claim_authority import (
    SuspendedOperationClaimAuthority,
)
from intergrax.contracts.execution.suspended_operation.reentry import (
    ExecutionSuspendedWorkReentryDisposition,
)
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    reset_active_execution_identity,
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
from intergrax.runtime.task.task import Task
from tests.unit.autonomous_work.test_uca6c_r4_real_codecraft_execution import (
    _TASK_ID,
    _TENANT,
)
from tests.unit.runtime.execution.suspended_operation.test_uca6c_r6_r5_9_r2_multi_host_fencing import (
    OWNER_HOST_B,
    advance_lease_clock,
    reclaim_as,
)
from tests.unit.runtime.execution.suspended_operation.test_uca6c_r6_r5_9_r2_r1_canonical_reentry_fencing import (
    DualHostReentryFixture,
    _advance_to_gen3_blocked,
    _build_dual_host_fixture,
    _reenter,
    _resume_gen3_without_reentry,
    _ReclaimBridge,
    _sync_host_stores,
)

pytestmark = pytest.mark.unit


@contextmanager
def _multi_host_fixture(tmp_path: Path) -> Iterator[DualHostReentryFixture]:
    yield _build_dual_host_fixture(tmp_path)


def test_same_owner_stale_fence_rejected_before_toolruntime(tmp_path: Path) -> None:
    """ABA: same owner_id, old fence after reclaim — must fail before ToolRuntime."""
    with _multi_host_fixture(tmp_path) as fixture:
        run_id = mint_run_id()
        attempt_id = mint_attempt_id()
        execution_id = mint_execution_id()
        task = Task(
            tenant_id=_TENANT, user_id="u1", message="r59r2r1r1-aba", task_id=_TASK_ID
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
            pause_gen = d3.pause_generation
            assert claimed_b.descriptor is not None
            assert claimed_b.descriptor.claim_ownership is not None
            fence_b_gen1 = claimed_b.descriptor.claim_ownership.fence
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
                    expected_fence=fence_b_gen1,
                    lease_at=expired_now + timedelta(minutes=5),
                )
            assert reclaimed.descriptor is not None
            assert reclaimed.descriptor.claim_ownership is not None
            fence_b_gen2 = reclaimed.descriptor.claim_ownership.fence
            revision_current = reclaimed.descriptor.materialization_revision
            assert reclaimed.descriptor.claim_ownership.owner_id == OWNER_HOST_B
            assert fence_b_gen2 > fence_b_gen1

            _sync_host_stores(fixture)
            _resume_gen3_without_reentry(
                fixture,
                task,
                continuation_id=c3,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
                hitl=fixture.hitl_b,
            )
            reentry_b = fixture.composition_b.suspended_work_reentry_coordinator
            assert reentry_b is not None
            stale_same_owner = SuspendedOperationClaimAuthority(
                owner_id=OWNER_HOST_B,
                fence=fence_b_gen1,
                materialization_revision=revision_current,
                pause_generation=pause_gen,
            )
            result = _reenter(
                reentry_b,
                continuation_id=c3,
                identity=d3.identity,
                task=task,
                counters=fixture.counters,
                host="b",
                claim_authority=stale_same_owner,
            )
            assert result.disposition is ExecutionSuspendedWorkReentryDisposition.FAILED
            assert result.reason_detail == "stale_claim_fence"
            assert fixture.backend_a.calls == 0
            assert fixture.backend_b.calls == 0
            assert fixture.counters.terminal_writes == 0
        finally:
            reset_governed_execution_task(task_token)
            reset_active_execution_governance_identity(gov_token)
            reset_active_execution_identity(id_token)


def test_claim_authority_contract_rejects_malformed() -> None:
    with pytest.raises(ValueError):
        SuspendedOperationClaimAuthority(
            owner_id="  ",
            fence=0,
            materialization_revision=1,
            pause_generation=1,
        )
    with pytest.raises(ValueError):
        SuspendedOperationClaimAuthority(
            owner_id="host",
            fence=0,
            materialization_revision=1,
            pause_generation=0,
        )
