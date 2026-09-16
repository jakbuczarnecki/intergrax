# © Artur Czarnecki. All rights reserved.

"""MP-4R6 — generic checkpoint restore must not synthesize approver authority."""

from __future__ import annotations

import pytest

from intergrax.contracts.execution_identity import mint_task_id
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.contracts.human_approver import (
    HumanApproverAuthMode,
    HumanApproverEvidence,
    local_development_approver_evidence,
)
from intergrax.runtime.human.models import HumanResponseVerdict
from intergrax.runtime.nexus.orchestration.human_response import (
    HitlCheckpointRestoreError,
    prepare_hitl_resume_after_checkpoint_restore,
)
from intergrax.runtime.task.task import Task
from intergrax.runtime.task.task_contract import HumanApprovalResolution, TaskPauseRecord
from intergrax.runtime.task.task_metadata_bridge import promote_legacy_human_verdict_from_metadata
from intergrax.runtime.task.task_metadata_keys import TaskMetadataKey

pytestmark = pytest.mark.unit

_PAUSE = "pause-1"
_HR = "hr-1"
_TASK = mint_task_id()
_TENANT = "tenant-a"


def _task_with_pause(*, verdict: str, approver: HumanApproverEvidence | None = None) -> Task:
    task = Task(
        task_id=_TASK,
        tenant_id=_TENANT,
        user_id="task-subject-user",
        agent_id="agent-1",
        input_text="x",
    )
    task.runtime.governance.pause_record = TaskPauseRecord(
        pause_id=_PAUSE,
        task_id=_TASK,
        human_request_id=_HR,
    )
    task.options.human.verdict = verdict
    if approver is not None:
        task.options.human.approver = approver
    return task


def _identity_provider_approver(*, tenant_id: str = _TENANT) -> HumanApproverEvidence:
    return HumanApproverEvidence(
        tenant_id=tenant_id,
        user_id="operator-1",
        principal_type=PrincipalType.USER,
        auth_subject="idp-subject-abc",
        auth_mode=HumanApproverAuthMode.IDENTITY_PROVIDER,
    )


def _api_key_approver() -> HumanApproverEvidence:
    return HumanApproverEvidence(
        tenant_id=_TENANT,
        user_id="operator-api",
        principal_type=PrincipalType.USER,
        auth_subject="api-subject-xyz",
        auth_mode=HumanApproverAuthMode.API_KEY,
    )


@pytest.mark.parametrize(
    "verdict",
    [
        HumanResponseVerdict.APPROVE.value,
        HumanResponseVerdict.REJECT.value,
        HumanResponseVerdict.ESCALATE.value,
    ],
)
def test_missing_approver_fail_closed_no_local_development(verdict: str) -> None:
    task = _task_with_pause(verdict=verdict, approver=None)
    with pytest.raises(HitlCheckpointRestoreError, match="approver evidence missing"):
        prepare_hitl_resume_after_checkpoint_restore(task)
    assert task.options.human.approver is None


def test_restores_pause_correlation_without_synthesizing_approver() -> None:
    task = _task_with_pause(verdict=HumanResponseVerdict.APPROVE.value, approver=None)
    task.options.human.pause_id = None
    task.options.human.human_request_id = None
    with pytest.raises(HitlCheckpointRestoreError):
        prepare_hitl_resume_after_checkpoint_restore(task)
    assert task.options.human.pause_id == _PAUSE
    assert task.options.human.human_request_id == _HR


def test_exact_authenticated_approver_round_trip_from_task_human_input() -> None:
    original = _identity_provider_approver()
    task = _task_with_pause(verdict=HumanResponseVerdict.APPROVE.value, approver=original)
    prepare_hitl_resume_after_checkpoint_restore(task)
    restored = task.options.human.approver
    assert restored is not None
    assert restored == original
    assert restored.auth_mode is HumanApproverAuthMode.IDENTITY_PROVIDER
    assert restored.auth_subject == "idp-subject-abc"
    assert restored.principal_type is PrincipalType.USER


def test_exact_api_key_approver_not_local_development() -> None:
    original = _api_key_approver()
    task = _task_with_pause(verdict=HumanResponseVerdict.APPROVE.value, approver=original)
    prepare_hitl_resume_after_checkpoint_restore(task)
    restored = task.options.human.approver
    assert restored is not None
    assert restored.auth_mode is HumanApproverAuthMode.API_KEY
    assert restored.auth_mode is not HumanApproverAuthMode.LOCAL_DEVELOPMENT


def test_restores_approver_from_canonical_hitl_resolution_when_human_input_empty() -> None:
    original = _identity_provider_approver()
    task = _task_with_pause(verdict=HumanResponseVerdict.APPROVE.value, approver=None)
    task.runtime.governance.hitl_resolution = HumanApprovalResolution(
        task_id=_TASK,
        pause_id=_PAUSE,
        human_request_id=_HR,
        verdict=HumanResponseVerdict.APPROVE,
        approver=original,
        resolved_at="2026-01-01T00:00:00Z",
    )
    prepare_hitl_resume_after_checkpoint_restore(task)
    assert task.options.human.approver == original


def test_existing_human_approver_not_overwritten_by_resolution() -> None:
    on_task = _identity_provider_approver()
    on_resolution = _api_key_approver()
    task = _task_with_pause(verdict=HumanResponseVerdict.APPROVE.value, approver=on_task)
    task.runtime.governance.hitl_resolution = HumanApprovalResolution(
        task_id=_TASK,
        pause_id=_PAUSE,
        human_request_id=_HR,
        verdict=HumanResponseVerdict.APPROVE,
        approver=on_resolution,
        resolved_at="2026-01-01T00:00:00Z",
    )
    prepare_hitl_resume_after_checkpoint_restore(task)
    assert task.options.human.approver == on_task


def test_persisted_local_development_round_trips_when_explicitly_present() -> None:
    original = local_development_approver_evidence(tenant_id=_TENANT, actor_id="explicit-dev-op")
    task = _task_with_pause(verdict=HumanResponseVerdict.APPROVE.value, approver=original)
    prepare_hitl_resume_after_checkpoint_restore(task)
    assert task.options.human.approver == original


def test_legacy_verdict_metadata_without_approver_does_not_fabricate_approver() -> None:
    task = Task(
        task_id=_TASK,
        tenant_id=_TENANT,
        user_id="task-subject-user",
        agent_id="agent-1",
        input_text="x",
    )
    task.metadata[TaskMetadataKey.HUMAN_APPROVED] = True
    task.runtime.governance.pause_record = TaskPauseRecord(
        pause_id=_PAUSE,
        task_id=_TASK,
        human_request_id=_HR,
    )
    promote_legacy_human_verdict_from_metadata(task)
    assert task.options.human.verdict == HumanResponseVerdict.APPROVE.value
    assert task.options.human.approver is None
    with pytest.raises(HitlCheckpointRestoreError):
        prepare_hitl_resume_after_checkpoint_restore(task)


def test_no_verdict_skips_restore_helper() -> None:
    task = _task_with_pause(verdict=HumanResponseVerdict.APPROVE.value, approver=None)
    task.options.human.verdict = None
    prepare_hitl_resume_after_checkpoint_restore(task)
    assert task.options.human.approver is None
