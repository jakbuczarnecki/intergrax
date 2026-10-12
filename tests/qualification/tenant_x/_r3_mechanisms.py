# © Artur Czarnecki. All rights reserved.

"""TENANT-X-R3 bounded recovery-family inventory (A–J)."""

from __future__ import annotations

from typing import Final

from tests.qualification.tenant_x._r1_structural_io import all_structural_patterns_present
from tests.qualification.tenant_x._r3_types import R3FamilyBinding, R3RecoveryFamilyRow

_CHECKPOINT_WRITE_CHECKS: Final[tuple[tuple[str, str], ...]] = (
    (
        "intergrax/runtime/long_running/store.py",
        "checkpoint.tenant_id",
    ),
    (
        "intergrax/runtime/long_running/store.py",
        "WHERE tenant_id = ? AND task_id = ?",
    ),
)

_CHECKPOINT_LOAD_CHECKS: Final[tuple[tuple[str, str], ...]] = (
    (
        "intergrax/runtime/long_running/checkpoint_resume_validation.py",
        "checkpoint.tenant_id != target_tenant_id",
    ),
    (
        "intergrax/agents/persistence/checkpoint_store.py",
        "checkpoint.tenant_id != tenant_id",
    ),
)

_TASK_CONTROL_RESUME_CHECKS: Final[tuple[tuple[str, str], ...]] = (
    (
        "intergrax/applications/_shared/task_control.py",
        "target_tenant_id=tenant_id",
    ),
    (
        "intergrax/applications/_shared/task_control.py",
        "task_tenant_id=checkpoint.tenant_id",
    ),
)


def _binding_from_checks(
    checks: tuple[tuple[str, str], ...],
) -> tuple[R3FamilyBinding, tuple[str, ...]]:
    matched = all_structural_patterns_present(checks)
    if len(matched) == len(checks):
        return R3FamilyBinding.TENANT_BOUND, matched
    return R3FamilyBinding.UNKNOWN, ("discovery: structural markers incomplete",)


def build_r3_recovery_families() -> tuple[R3RecoveryFamilyRow, ...]:
    write_binding, write_ev = _binding_from_checks(_CHECKPOINT_WRITE_CHECKS)
    load_binding, load_ev = _binding_from_checks(_CHECKPOINT_LOAD_CHECKS)
    resume_binding, resume_ev = _binding_from_checks(_TASK_CONTROL_RESUME_CHECKS)
    return (
        R3RecoveryFamilyRow(
            family_id="A.durable_execution_state",
            family_label="durable execution state",
            binding=R3FamilyBinding.TENANT_BOUND,
            authoritative_tenant_source="TaskCheckpoint.tenant_id / AgentRunCheckpoint.tenant_id",
            evidence=(
                "intergrax/runtime/long_running/models.py::TaskCheckpoint",
                "intergrax/agents/persistence/checkpoint_store.py::AgentRunCheckpoint",
            ),
        ),
        R3RecoveryFamilyRow(
            family_id="B.checkpoint_creation",
            family_label="checkpoint creation",
            binding=R3FamilyBinding.TENANT_BOUND,
            authoritative_tenant_source="task.tenant_id at checkpoint build",
            evidence=(
                "intergrax/runtime/long_running/checkpoint_builder.py::checkpoint carries task tenant",
            ),
        ),
        R3RecoveryFamilyRow(
            family_id="C.checkpoint_persistence_load",
            family_label="checkpoint persistence/load",
            binding=write_binding if load_binding == R3FamilyBinding.TENANT_BOUND else R3FamilyBinding.UNKNOWN,
            authoritative_tenant_source="checkpoint payload + composite durable keys (tenant_id, task_id|run_id)",
            evidence=write_ev + load_ev,
        ),
        R3RecoveryFamilyRow(
            family_id="D.retry_state",
            family_label="retry state",
            binding=R3FamilyBinding.TENANT_BOUND,
            authoritative_tenant_source="ExecutionRetryService.transition_for_retry(tenant_id=…)",
            evidence=(
                "intergrax/runtime/execution/retry/service.py::transition_for_retry tenant_id required",
                "intergrax/runtime/execution/attempt_lifecycle/service.py::transition_to_next_attempt(tenant_id=…)",
            ),
        ),
        R3RecoveryFamilyRow(
            family_id="E.resume",
            family_label="resume",
            binding=resume_binding,
            authoritative_tenant_source="checkpoint.tenant_id validated against caller target_tenant_id",
            evidence=resume_ev
            + (
                "intergrax/runtime/long_running/checkpoint_resume_validation.py::validate_checkpoint_identity_binding",
            ),
        ),
        R3RecoveryFamilyRow(
            family_id="F.recovery",
            family_label="recovery",
            binding=R3FamilyBinding.TENANT_BOUND,
            authoritative_tenant_source="checkpoint.tenant_id vs recovery request tenant_id",
            evidence=(
                "intergrax/runtime/execution/fan_out_partial_recovery.py::checkpoint.tenant_id != tenant_id",
                "intergrax/runtime/execution/orchestration.py::REJECT_TENANT governance mismatch",
            ),
        ),
        R3RecoveryFamilyRow(
            family_id="G.replay",
            family_label="replay",
            binding=R3FamilyBinding.TENANT_BOUND,
            authoritative_tenant_source="historical record tenant vs reconstruction scope",
            evidence=(
                "intergrax/runtime/runtime_inspection/adapters/execution_reconstruction.py::tenant mismatch",
            ),
        ),
        R3RecoveryFamilyRow(
            family_id="H.restore",
            family_label="restore / vendor sync checkpoint",
            binding=R3FamilyBinding.TENANT_BOUND,
            authoritative_tenant_source="checkpoint.tenant_id + partition key",
            evidence=(
                "intergrax/runtime/vendor_knowledge/sync_document_store.py::_checkpoint_partition_key",
                "intergrax/runtime/vendor_knowledge/sync_document_store.py::checkpoint.tenant_id != expected_tenant",
            ),
        ),
        R3RecoveryFamilyRow(
            family_id="I.compensation_recovery_queue",
            family_label="compensation/recovery queue",
            binding=R3FamilyBinding.TENANT_BOUND,
            authoritative_tenant_source="job.tenant_id + PRIMARY KEY (tenant_id, idempotency_key)",
            evidence=(
                "intergrax/agents/persistence/compensation_queue_store.py::(tenant_id, idempotency_key)",
            ),
        ),
        R3RecoveryFamilyRow(
            family_id="J.idempotency_recovery_eligibility",
            family_label="idempotency/recovery eligibility",
            binding=R3FamilyBinding.TENANT_BOUND,
            authoritative_tenant_source="idempotency_key.tenant_id bound to scope.tenant_id",
            evidence=(
                "intergrax/contracts/collaborative_activity.py::idempotency_key tenant_id must match scope",
            ),
        ),
    )


def r3_critical_unknown_families() -> tuple[str, ...]:
    return tuple(
        f.family_id for f in build_r3_recovery_families() if f.binding == R3FamilyBinding.UNKNOWN
    )
