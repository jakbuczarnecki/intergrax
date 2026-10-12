# © Artur Czarnecki. All rights reserved.

"""TENANT-X-R3 required qualification matrices."""

from __future__ import annotations

from typing import Final

from tests.qualification.tenant_x._r3_mechanisms import build_r3_recovery_families
from tests.qualification.tenant_x._r3_types import (
    R3ChainVerdict,
    R3CrossTenantClass,
    R3MatrixRow,
    R3RecoveryAuthorityRole,
    R3RecoveryPathClass,
    R3SubstitutionClass,
)

STATE_TENANT_CONTINUITY_MATRIX: Final[tuple[R3MatrixRow, ...]] = tuple(
    R3MatrixRow(
        row_id=f.family_id,
        surface_hint=f.family_label,
        verdict=R3ChainVerdict.PROVEN_CURRENT_HEAD
        if f.binding.value == "TENANT_BOUND"
        else R3ChainVerdict.MISSING,
        evidence=f.evidence,
        tenant_expression=f.authoritative_tenant_source,
    )
    for f in build_r3_recovery_families()
    if f.family_id in {"A.durable_execution_state", "C.checkpoint_persistence_load"}
)

CHECKPOINT_WRITE_MATRIX: Final[tuple[R3MatrixRow, ...]] = (
    R3MatrixRow(
        row_id="task_checkpoint.sqlite_save",
        surface_hint="intergrax/runtime/long_running/store.py::SQLiteTaskCheckpointStore.save",
        verdict=R3ChainVerdict.PROVEN_CURRENT_HEAD,
        evidence=(
            "_checkpoint_write_payload includes checkpoint.tenant_id",
            "INSERT stores tenant_id column",
        ),
        tenant_expression="checkpoint.tenant_id from TaskCheckpoint model",
        durable_key_scope="(tenant_id, task_id, checkpoint_revision)",
    ),
    R3MatrixRow(
        row_id="agent_checkpoint.sqlite_save",
        surface_hint="intergrax/agents/persistence/checkpoint_store.py::SQLiteAgentCheckpointStore",
        verdict=R3ChainVerdict.PROVEN_CURRENT_HEAD,
        evidence=("WHERE run_id = ? AND tenant_id = ?", "payload carries tenant_id"),
        tenant_expression="AgentRunCheckpoint.tenant_id",
        durable_key_scope="(run_id, tenant_id)",
    ),
    R3MatrixRow(
        row_id="vendor_sync.checkpoint_finalize",
        surface_hint="intergrax/runtime/vendor_knowledge/sync_document_store.py",
        verdict=R3ChainVerdict.PROVEN_CURRENT_HEAD,
        evidence=("partition_key=_checkpoint_partition_key(checkpoint.tenant_id)",),
        tenant_expression="checkpoint.tenant_id",
        durable_key_scope="partition_key(tenant_id)",
    ),
)

CHECKPOINT_LOAD_MATRIX: Final[tuple[R3MatrixRow, ...]] = (
    R3MatrixRow(
        row_id="resume.validate_identity_binding",
        surface_hint="checkpoint_resume_validation.py::validate_checkpoint_identity_binding",
        verdict=R3RecoveryPathClass.CALLER_TENANT_VALIDATED_AGAINST_DURABLE,
        evidence=("checkpoint.tenant_id != target_tenant_id → REJECT_TENANT",),
        tenant_expression="target_tenant_id vs checkpoint.tenant_id",
        durable_key_scope="n/a — logical binding",
    ),
    R3MatrixRow(
        row_id="task_store.get_by_token",
        surface_hint="SQLiteTaskCheckpointStore.get_by_token",
        verdict=R3RecoveryPathClass.DURABLE_TENANT_AUTHORITATIVE,
        evidence=("SQL filters tenant_id — wrong tenant yields None",),
        tenant_expression="tenant_id query parameter",
        durable_key_scope="(tenant_id, task_id, resume_token)",
    ),
    R3MatrixRow(
        row_id="agent_store.load",
        surface_hint="SQLiteAgentCheckpointStore.load",
        verdict=R3RecoveryPathClass.CALLER_TENANT_VALIDATED_AGAINST_DURABLE,
        evidence=("CheckpointStreamIdentityConflictError on payload/lookup mismatch",),
        tenant_expression="lookup tenant_id vs payload tenant_id",
        durable_key_scope="(run_id, tenant_id)",
    ),
)

RECOVERY_RESUME_MATRIX: Final[tuple[R3MatrixRow, ...]] = (
    R3MatrixRow(
        row_id="evaluate_checkpoint_resume_eligibility",
        surface_hint="checkpoint_resume_validation.py",
        verdict=R3RecoveryPathClass.CALLER_TENANT_VALIDATED_AGAINST_DURABLE,
        evidence=("REJECT_TENANT adversarial unit EE-B3-C",),
        tenant_expression="target_tenant_id",
        durable_key_scope="checkpoint.tenant_id",
    ),
    R3MatrixRow(
        row_id="governed_resume_checkpoint_task",
        surface_hint="applications/_shared/task_control.py",
        verdict=R3RecoveryPathClass.CALLER_TENANT_VALIDATED_AGAINST_DURABLE,
        evidence=(
            "assert_checkpoint_resume_materialization_eligible(..., target_tenant_id=tenant_id)",
            "validate_task_control_principal_tenant_authority(..., task_tenant_id=checkpoint.tenant_id)",
        ),
        tenant_expression="caller tenant_id validated vs checkpoint",
        durable_key_scope="get_by_token scoped lookup",
    ),
    R3MatrixRow(
        row_id="partial_recovery.validate",
        surface_hint="fan_out_partial_recovery.py",
        verdict=R3RecoveryPathClass.CALLER_TENANT_VALIDATED_AGAINST_DURABLE,
        evidence=("PartialRecoveryErrorCode.TENANT_MISMATCH",),
        tenant_expression="tenant_id parameter vs checkpoint.tenant_id",
        durable_key_scope="checkpoint durable record",
    ),
)

RETRY_REPLAY_MATRIX: Final[tuple[R3MatrixRow, ...]] = (
    R3MatrixRow(
        row_id="retry.transition_for_retry",
        surface_hint="execution/retry/service.py",
        verdict=R3ChainVerdict.PROVEN_CURRENT_HEAD,
        evidence=("tenant_id passed to attempt lifecycle transition",),
        tenant_expression="explicit tenant_id argument",
        durable_key_scope="attempt lifecycle durable store",
    ),
    R3MatrixRow(
        row_id="execution_reconstruction.tenant_scope",
        surface_hint="runtime_inspection/adapters/execution_reconstruction.py",
        verdict=R3ChainVerdict.PROVEN_CURRENT_HEAD,
        evidence=("execution reconstruction tenant mismatch",),
        tenant_expression="reconstruction scope tenant",
        durable_key_scope="historical execution record",
    ),
    R3MatrixRow(
        row_id="decision_checkpoint.tenant_match",
        surface_hint="contracts/decision_checkpoint.py",
        verdict=R3ChainVerdict.PROVEN_CURRENT_HEAD,
        evidence=("decision checkpoint revision proposal tenant_id does not match lifecycle",),
        tenant_expression="lifecycle tenant vs proposal tenant",
        durable_key_scope="decision checkpoint artifact",
    ),
)

DURABLE_KEY_ISOLATION_MATRIX: Final[tuple[R3MatrixRow, ...]] = (
    R3MatrixRow(
        row_id="task_checkpoints.composite",
        surface_hint="task_checkpoints table",
        verdict=R3ChainVerdict.PROVEN_CURRENT_HEAD,
        evidence=("idx on (tenant_id, task_id)", "UNIQUE (tenant_id, task_id, checkpoint_revision)"),
        tenant_expression="tenant_id column",
        durable_key_scope="tenant_id + task_id + revision",
    ),
    R3MatrixRow(
        row_id="agent_run_checkpoints.composite",
        surface_hint="agent_run_checkpoints",
        verdict=R3ChainVerdict.PROVEN_CURRENT_HEAD,
        evidence=("WHERE run_id = ? AND tenant_id = ?",),
        tenant_expression="tenant_id column",
        durable_key_scope="run_id + tenant_id",
    ),
    R3MatrixRow(
        row_id="compensation_queue.composite",
        surface_hint="compensation_queue_store",
        verdict=R3ChainVerdict.PROVEN_CURRENT_HEAD,
        evidence=("PRIMARY KEY (tenant_id, idempotency_key)",),
        tenant_expression="job.tenant_id",
        durable_key_scope="tenant_id + idempotency_key",
    ),
    R3MatrixRow(
        row_id="checkpoint_id.alone_not_tenant_isolation",
        surface_hint="task_checkpoints.checkpoint_id",
        verdict=R3ChainVerdict.NOT_APPLICABLE,
        evidence=("tenant isolation enforced via tenant_id column + stream keys, not checkpoint_id alone",),
        tenant_expression="checkpoint_id is not tenant authority",
        durable_key_scope="supplemented by tenant_id scopes",
    ),
)

RECOVERY_TENANT_AUTHORITY_MATRIX: Final[tuple[R3MatrixRow, ...]] = (
    R3MatrixRow(
        row_id="checkpoint_resume_validation",
        surface_hint="validate_checkpoint_identity_binding",
        verdict=R3RecoveryAuthorityRole.TENANT_VALIDATOR,
        evidence=("compares durable checkpoint.tenant_id to caller target",),
    ),
    R3MatrixRow(
        row_id="task_checkpoint_model",
        surface_hint="TaskCheckpoint",
        verdict=R3RecoveryAuthorityRole.TENANT_CARRIER,
        evidence=("tenant_id field on durable checkpoint artifact",),
    ),
    R3MatrixRow(
        row_id="task_snapshot_parser",
        surface_hint="_parse_checkpoint_snapshot_task",
        verdict=R3RecoveryAuthorityRole.TENANT_RESTORER_FROM_DURABLE_TRUTH,
        evidence=("restores tenant from snapshot; mismatch vs checkpoint → REJECT_TENANT",),
    ),
    R3MatrixRow(
        row_id="task_control.principal_authority",
        surface_hint="validate_task_control_principal_tenant_authority",
        verdict=R3RecoveryAuthorityRole.TENANT_VALIDATOR,
        evidence=("principal must match checkpoint.tenant_id for resume",),
    ),
)

MISSING_DEFAULT_GLOBAL_MATRIX: Final[tuple[R3MatrixRow, ...]] = (
    R3MatrixRow(
        row_id="cross_tenant_resume_request",
        surface_hint="evaluate_checkpoint_resume_eligibility",
        verdict=R3SubstitutionClass.ILLEGAL_INPUT_REJECTED,
        evidence=("checkpoint A + target tenant B → REJECT_TENANT",),
    ),
    R3MatrixRow(
        row_id="missing_snapshot_tenant",
        surface_hint="_parse_checkpoint_snapshot_task",
        verdict=R3SubstitutionClass.ILLEGAL_INPUT_REJECTED,
        evidence=("task_snapshot missing required tenant_id → REJECT_MALFORMED",),
    ),
    R3MatrixRow(
        row_id="blank_target_tenant_resume",
        surface_hint="validate_checkpoint_identity_binding",
        verdict=R3SubstitutionClass.ILLEGAL_INPUT_REJECTED,
        evidence=("empty target_tenant_id != checkpoint tenant → REJECT_TENANT",),
    ),
    R3MatrixRow(
        row_id="checkpoint_revision_schema",
        surface_hint="decision_checkpoint lifecycle",
        verdict=R3SubstitutionClass.LEGAL_NON_TENANT,
        evidence=("non-tenant schema versioning — no default tenant substitution",),
    ),
)

CROSS_TENANT_RECOVERY_MATRIX: Final[tuple[R3MatrixRow, ...]] = (
    R3MatrixRow(
        row_id="checkpoint_a_recovery_b",
        surface_hint="validate_checkpoint_identity_binding",
        verdict=R3CrossTenantClass.CROSS_TENANT_REJECTED,
        evidence=("REJECT_TENANT",),
    ),
    R3MatrixRow(
        row_id="state_a_restore_b_lookup",
        surface_hint="SQLiteTaskCheckpointStore.get_by_token",
        verdict=R3CrossTenantClass.CROSS_TENANT_REJECTED,
        evidence=("wrong tenant → no row / invalid_resume_token path",),
    ),
    R3MatrixRow(
        row_id="retry_record_a_worker_b",
        surface_hint="retry transition tenant_id parameter",
        verdict=R3CrossTenantClass.SAME_TENANT_PROVEN,
        evidence=("retry requires explicit tenant_id aligned with durable attempt scope",),
    ),
    R3MatrixRow(
        row_id="replay_a_caller_b",
        surface_hint="execution_reconstruction adapter",
        verdict=R3CrossTenantClass.CROSS_TENANT_REJECTED,
        evidence=("tenant mismatch fail closed on reconstruction",),
    ),
    R3MatrixRow(
        row_id="idempotency_lookup_a_vs_b",
        surface_hint="compensation_queue_store",
        verdict=R3CrossTenantClass.CROSS_TENANT_REJECTED,
        evidence=("lookup keyed by (tenant_id, idempotency_key)",),
    ),
    R3MatrixRow(
        row_id="partial_recovery_a_request_b",
        surface_hint="PartialRecoveryCoordinator",
        verdict=R3CrossTenantClass.CROSS_TENANT_REJECTED,
        evidence=("TENANT_MISMATCH",),
    ),
    R3MatrixRow(
        row_id="agent_checkpoint_load_a_as_b",
        surface_hint="SQLiteAgentCheckpointStore",
        verdict=R3CrossTenantClass.CROSS_TENANT_REJECTED,
        evidence=("CheckpointStreamIdentityConflictError",),
    ),
)
