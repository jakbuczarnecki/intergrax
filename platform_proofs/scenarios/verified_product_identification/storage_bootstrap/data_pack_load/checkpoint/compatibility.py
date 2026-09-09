"""Run identity, compatibility, and resume decision helpers."""

from __future__ import annotations

from datetime import UTC, datetime

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.manifest import (
    DataPackManifest,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.batching import (
    compute_bootstrap_plan,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.checkpoint.contracts import (
    VPI_STORAGE_BOOTSTRAP_CHECKPOINT_SCHEMA_VERSION,
    VPI_STORAGE_BOOTSTRAP_ORDERING_POLICY,
    BootstrapCheckpointCompatibility,
    BootstrapCheckpointState,
    BootstrapResumeDecision,
    BootstrapRunIdentity,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.checkpoint.errors import (
    CheckpointCorrupt,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.contracts import (
    BootstrapBatchPhase,
    BootstrapBatchSize,
    BootstrapPlan,
    BootstrapRequest,
    ResumeMode,
)


def utc_now_iso() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat()


def build_run_identity(*, manifest: DataPackManifest, request: BootstrapRequest) -> BootstrapRunIdentity:
    return BootstrapRunIdentity(
        data_pack_content_identity=manifest.content_identity,
        data_pack_version=manifest.data_pack_version,
        record_count=manifest.record_count,
        batch_size=request.batch_size.value,
        relational_target=request.relational_target,
        vector_target=request.vector_target,
        verification_mode=request.verification_mode,
        ordering_policy=VPI_STORAGE_BOOTSTRAP_ORDERING_POLICY,
        state_schema_version=VPI_STORAGE_BOOTSTRAP_CHECKPOINT_SCHEMA_VERSION,
        source_dataset_sha256=manifest.source_dataset.dataset_sha256,
        embedding_model_identity=manifest.embedding_identity.resolved_model_identity(),
    )


def initial_checkpoint_state(
    *,
    run_identity: BootstrapRunIdentity,
    plan: BootstrapPlan,
    updated_at_utc: str,
) -> BootstrapCheckpointState:
    return BootstrapCheckpointState(
        schema_version=VPI_STORAGE_BOOTSTRAP_CHECKPOINT_SCHEMA_VERSION,
        run_identity=run_identity,
        batch_size=plan.batch_size,
        total_records=plan.record_count,
        batch_count=plan.batch_count,
        last_committed_batch_number=None,
        last_committed_global_row_index=None,
        last_committed_identity=None,
        committed_record_count=0,
        batch_phases=tuple(BootstrapBatchPhase.PENDING for _ in range(plan.batch_count)),
        state_revision=1,
        updated_at_utc=updated_at_utc,
    )


def records_through_batch(*, batch_number: int, batch_size: int, total_records: int) -> int:
    if batch_number < 0:
        raise ValueError("batch_number must be >= 0")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if total_records < 0:
        raise ValueError("total_records must be >= 0")
    committed = min((batch_number + 1) * batch_size, total_records)
    return committed


def validate_contiguous_committed_prefix(batch_phases: tuple[BootstrapBatchPhase, ...]) -> None:
    in_pending_suffix = False
    for batch_number, phase in enumerate(batch_phases):
        if in_pending_suffix and phase is BootstrapBatchPhase.COMMITTED:
            raise CheckpointCorrupt(
                f"non-contiguous committed batch history at batch {batch_number}"
            )
        if phase is not BootstrapBatchPhase.COMMITTED:
            in_pending_suffix = True


def validate_checkpoint_internal_consistency(state: BootstrapCheckpointState) -> None:
    validate_contiguous_committed_prefix(state.batch_phases)
    if state.last_committed_batch_number is None:
        committed_prefix = 0
    else:
        committed_prefix = state.last_committed_batch_number + 1
        for batch_number in range(committed_prefix):
            if state.batch_phases[batch_number] is not BootstrapBatchPhase.COMMITTED:
                raise CheckpointCorrupt(
                    f"batch {batch_number} marked pending while last_committed_batch_number "
                    f"is {state.last_committed_batch_number}"
                )
    expected_records = records_through_batch(
        batch_number=committed_prefix - 1,
        batch_size=state.batch_size,
        total_records=state.total_records,
    ) if committed_prefix > 0 else 0
    if state.committed_record_count != expected_records:
        raise CheckpointCorrupt(
            f"committed_record_count {state.committed_record_count} != expected {expected_records}"
        )


def validate_checkpoint_compatibility(
    *,
    run_identity: BootstrapRunIdentity,
    checkpoint: BootstrapCheckpointState,
    manifest: DataPackManifest,
) -> BootstrapCheckpointCompatibility:
    try:
        validate_checkpoint_internal_consistency(checkpoint)
    except CheckpointCorrupt as exc:
        return BootstrapCheckpointCompatibility(is_compatible=False, reason=str(exc))

    current_identity = build_run_identity(manifest=manifest, request=_request_from_run_identity(run_identity))
    checkpoint_identity = checkpoint.run_identity
    mismatches: list[str] = []
    if current_identity.data_pack_content_identity != checkpoint_identity.data_pack_content_identity:
        mismatches.append("data_pack_content_identity")
    if current_identity.data_pack_version != checkpoint_identity.data_pack_version:
        mismatches.append("data_pack_version")
    if current_identity.record_count != checkpoint_identity.record_count:
        mismatches.append("record_count")
    if current_identity.batch_size != checkpoint_identity.batch_size:
        mismatches.append("batch_size")
    if current_identity.relational_target != checkpoint_identity.relational_target:
        mismatches.append("relational_target")
    if current_identity.vector_target != checkpoint_identity.vector_target:
        mismatches.append("vector_target")
    if current_identity.verification_mode != checkpoint_identity.verification_mode:
        mismatches.append("verification_mode")
    if current_identity.ordering_policy != checkpoint_identity.ordering_policy:
        mismatches.append("ordering_policy")
    if current_identity.state_schema_version != checkpoint_identity.state_schema_version:
        mismatches.append("state_schema_version")
    if current_identity.source_dataset_sha256 != checkpoint_identity.source_dataset_sha256:
        mismatches.append("source_dataset_sha256")
    if current_identity.embedding_model_identity != checkpoint_identity.embedding_model_identity:
        mismatches.append("embedding_model_identity")
    if manifest.record_count != checkpoint.total_records:
        mismatches.append("manifest.record_count")
    plan = compute_bootstrap_plan(
        record_count=manifest.record_count,
        batch_size=run_identity.batch_size,
        relational_target=run_identity.relational_target,
        vector_target=run_identity.vector_target,
    )
    if plan.batch_count != checkpoint.batch_count:
        mismatches.append("batch_count")
    if mismatches:
        return BootstrapCheckpointCompatibility(
            is_compatible=False,
            reason=f"checkpoint incompatible fields: {', '.join(sorted(mismatches))}",
        )
    return BootstrapCheckpointCompatibility(is_compatible=True)


def compute_resume_decision(checkpoint: BootstrapCheckpointState) -> BootstrapResumeDecision:
    validate_checkpoint_internal_consistency(checkpoint)
    if checkpoint.last_committed_batch_number is None:
        return BootstrapResumeDecision(
            start_batch_number=0,
            committed_batch_count=0,
            committed_record_count=0,
        )
    start_batch = checkpoint.last_committed_batch_number + 1
    return BootstrapResumeDecision(
        start_batch_number=start_batch,
        committed_batch_count=start_batch,
        committed_record_count=checkpoint.committed_record_count,
    )


def advance_checkpoint_after_batch(
    state: BootstrapCheckpointState,
    *,
    batch_number: int,
    last_global_row_index: int,
    last_identity: str,
    updated_at_utc: str,
) -> BootstrapCheckpointState:
    if batch_number < 0 or batch_number >= state.batch_count:
        raise ValueError("batch_number out of range")
    new_phases = list(state.batch_phases)
    new_phases[batch_number] = BootstrapBatchPhase.COMMITTED
    committed_record_count = records_through_batch(
        batch_number=batch_number,
        batch_size=state.batch_size,
        total_records=state.total_records,
    )
    return BootstrapCheckpointState(
        schema_version=state.schema_version,
        run_identity=state.run_identity,
        batch_size=state.batch_size,
        total_records=state.total_records,
        batch_count=state.batch_count,
        last_committed_batch_number=batch_number,
        last_committed_global_row_index=last_global_row_index,
        last_committed_identity=last_identity,
        committed_record_count=committed_record_count,
        batch_phases=tuple(new_phases),
        state_revision=state.state_revision + 1,
        updated_at_utc=updated_at_utc,
        current_batch_phase=BootstrapBatchPhase.COMMITTED,
    )


def _request_from_run_identity(run_identity: BootstrapRunIdentity) -> BootstrapRequest:
    from pathlib import Path

    return BootstrapRequest(
        artifact_root=Path("/checkpoint-compat-placeholder"),
        relational_target=run_identity.relational_target,
        vector_target=run_identity.vector_target,
        batch_size=BootstrapBatchSize(run_identity.batch_size),
        resume_mode=ResumeMode.RESUME,
        verification_mode=run_identity.verification_mode,
    )
