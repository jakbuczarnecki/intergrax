"""Provider-neutral Data Pack storage bootstrap application service.

Cross-store consistency model (PostgreSQL + Qdrant, MySQL + Qdrant, etc.):

- No distributed ACID transaction across relational and vector backends.
- Application coordinates at logical batch boundaries.
- Checkpoint advances only after relational write AND vector write AND verification succeed.
- If vector write fails after relational success, batch remains NOT COMMITTED.
- Retry is idempotent via adapter upsert / insert-if-absent semantics.

Execution semantics:

- AT-LEAST-ONCE batch execution.
- Idempotent storage adapters.
- Durable logical-batch commit checkpoint.
- Effectively-once canonical storage results — not distributed exactly-once.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.status import (
    DataPackStatus,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.manifest import (
    DataPackManifest,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.batching import (
    compute_bootstrap_plan,
    iter_record_batches,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.checkpoint.compatibility import (
    advance_checkpoint_after_batch,
    build_run_identity,
    compute_resume_decision,
    initial_checkpoint_state,
    utc_now_iso,
    validate_checkpoint_compatibility,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.checkpoint.contracts import (
    BootstrapCheckpointState,
    BootstrapResumeDecision,
    BootstrapRunIdentity,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.checkpoint.errors import (
    BootstrapCheckpointError,
    CheckpointAlreadyExists,
    CheckpointConcurrentModification,
    CheckpointCorrupt,
    CheckpointNotFound,
    CheckpointPersistenceError,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.checkpoint.ports import (
    BootstrapCheckpointStorePort,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.contracts import (
    BootstrapBatchPhase,
    BootstrapFinalStatus,
    BootstrapPlan,
    BootstrapProgress,
    BootstrapRequest,
    BootstrapResult,
    RelationalBatch,
    RelationalLoadRecord,
    ResumeMode,
    VectorBatch,
    VectorLoadRecord,
    VerificationMode,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.errors import (
    BootstrapFailure,
    BootstrapFailureCategory,
    StorageBootstrapIdentityError,
    StorageBootstrapIntegrityError,
    StorageBootstrapPreconditionError,
    StorageBootstrapWriteError,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.mapping import (
    identity_key,
    paired_load_records,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.ports import (
    DataPackBootstrapReaderPort,
    RelationalStorageLoadPort,
    VectorStorageLoadPort,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.reader.errors import (
    DataPackReaderError,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.progress import (
    BootstrapProgressSinkPort,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.verification import (
    assert_batch_identity_parity,
    assert_verification_complete,
)


@dataclass(frozen=True, slots=True)
class StorageBootstrapDependencies:
    reader: DataPackBootstrapReaderPort
    relational: RelationalStorageLoadPort
    vector: VectorStorageLoadPort
    checkpoint_store: BootstrapCheckpointStorePort


@dataclass(slots=True)
class StorageBootstrapService:
    dependencies: StorageBootstrapDependencies

    def run(
        self,
        request: BootstrapRequest,
        *,
        progress_sink: BootstrapProgressSinkPort | None = None,
    ) -> BootstrapResult:
        started = time.perf_counter()
        self._validate_request(request)
        try:
            manifest = self.dependencies.reader.read_manifest()
        except DataPackReaderError as exc:
            return self._failed_result(
                request=request,
                record_count=0,
                failure=BootstrapFailure(
                    category=BootstrapFailureCategory.PRECONDITION_FAILED,
                    detail=str(exc),
                ),
            )
        if manifest.status is not DataPackStatus.READY:
            return self._failed_result(
                request=request,
                record_count=0,
                failure=BootstrapFailure(
                    category=BootstrapFailureCategory.PRECONDITION_FAILED,
                    detail=f"data pack status must be READY (got {manifest.status.value})",
                ),
            )

        record_count = manifest.record_count
        plan = compute_bootstrap_plan(
            record_count=record_count,
            batch_size=request.batch_size.value,
            relational_target=request.relational_target,
            vector_target=request.vector_target,
        )

        if request.plan_only:
            return BootstrapResult(
                status=BootstrapFinalStatus.SUCCESS,
                plan=plan,
                total_expected_records=record_count,
                total_relational_written=0,
                total_vectors_written=0,
                committed_batches=0,
                failed_batches=0,
                last_committed_global_row_index=None,
                failure=None,
            )

        run_identity = build_run_identity(manifest=manifest, request=request)
        checkpoint_state, resume_context = self._prepare_checkpoint(
            request=request,
            manifest=manifest,
            run_identity=run_identity,
            plan=plan,
        )
        if isinstance(resume_context, BootstrapFailure):
            return self._failed_result(
                request=request,
                record_count=record_count,
                failure=resume_context,
            )

        start_batch_number = resume_context.start_batch_number
        previously_committed_batches = resume_context.committed_batch_count
        previously_committed_records = resume_context.committed_record_count
        resumed_from_batch = start_batch_number if previously_committed_batches > 0 else None

        if resumed_from_batch is not None:
            self._emit_progress(
                progress_sink,
                phase=BootstrapBatchPhase.RESUMED,
                batch_number=resumed_from_batch,
                records_processed=previously_committed_records,
                total_records=record_count,
                elapsed_seconds=time.perf_counter() - started,
                last_identity=checkpoint_state.last_committed_identity,
            )

        total_relational_written = 0
        total_vectors_written = 0
        committed_batches = 0
        failed_batches = 0
        last_committed_global_row_index = checkpoint_state.last_committed_global_row_index
        terminal_failure: BootstrapFailure | None = None

        records_streamed = 0
        try:
            batch_iterator = iter_record_batches(
                self.dependencies.reader.iter_paired_records(),
                batch_size=request.batch_size.value,
            )
        except DataPackReaderError as exc:
            return self._failed_result(
                request=request,
                record_count=record_count,
                failure=BootstrapFailure(
                    category=BootstrapFailureCategory.PRECONDITION_FAILED,
                    detail=str(exc),
                ),
            )

        try:
            for batch_number, batch_pairs in batch_iterator:
                records_streamed += len(batch_pairs)
                if batch_number < start_batch_number:
                    continue

                relational_records: list[RelationalLoadRecord] = []
                vector_records: list[VectorLoadRecord] = []
                try:
                    for pair in batch_pairs:
                        relational_record, vector_record = paired_load_records(pair)
                        relational_records.append(relational_record)
                        vector_records.append(vector_record)
                except StorageBootstrapIdentityError as exc:
                    failed_batches += 1
                    terminal_failure = BootstrapFailure(
                        category=BootstrapFailureCategory.IDENTITY_MISMATCH,
                        detail=str(exc),
                        batch_number=batch_number,
                    )
                    break

                relational_batch = RelationalBatch(
                    batch_number=batch_number,
                    target=request.relational_target,
                    records=tuple(relational_records),
                )
                vector_batch = VectorBatch(
                    batch_number=batch_number,
                    target=request.vector_target,
                    records=tuple(vector_records),
                )
                assert_batch_identity_parity(relational_batch, vector_batch)

                last_identity = identity_key(relational_batch.records[-1].source_ref)
                records_processed_before_batch = (
                    previously_committed_records + committed_batches * request.batch_size.value
                )
                self._emit_progress(
                    progress_sink,
                    phase=BootstrapBatchPhase.RELATIONAL_WRITING,
                    batch_number=batch_number,
                    records_processed=records_processed_before_batch,
                    total_records=record_count,
                    elapsed_seconds=time.perf_counter() - started,
                    last_identity=last_identity,
                )

                try:
                    relational_result = self.dependencies.relational.write_batch(relational_batch)
                except StorageBootstrapWriteError as exc:
                    failed_batches += 1
                    terminal_failure = BootstrapFailure(
                        category=BootstrapFailureCategory.RELATIONAL_WRITE_FAILED,
                        detail=str(exc),
                        batch_number=batch_number,
                    )
                    break

                if not relational_result.is_complete_success:
                    failed_batches += 1
                    terminal_failure = BootstrapFailure(
                        category=BootstrapFailureCategory.RELATIONAL_WRITE_FAILED,
                        detail="relational batch write incomplete",
                        batch_number=batch_number,
                        first_failed_identity=relational_result.first_failed_identity,
                    )
                    break

                self._emit_progress(
                    progress_sink,
                    phase=BootstrapBatchPhase.VECTOR_WRITING,
                    batch_number=batch_number,
                    records_processed=records_processed_before_batch,
                    total_records=record_count,
                    elapsed_seconds=time.perf_counter() - started,
                    last_identity=last_identity,
                )

                try:
                    vector_result = self.dependencies.vector.write_batch(vector_batch)
                except StorageBootstrapWriteError as exc:
                    failed_batches += 1
                    terminal_failure = BootstrapFailure(
                        category=BootstrapFailureCategory.VECTOR_WRITE_FAILED,
                        detail=str(exc),
                        batch_number=batch_number,
                    )
                    break

                if not vector_result.is_complete_success:
                    failed_batches += 1
                    terminal_failure = BootstrapFailure(
                        category=BootstrapFailureCategory.VECTOR_WRITE_FAILED,
                        detail="vector batch write incomplete",
                        batch_number=batch_number,
                        first_failed_identity=vector_result.first_failed_identity,
                    )
                    break

                if request.verification_mode is VerificationMode.STRICT:
                    self._emit_progress(
                        progress_sink,
                        phase=BootstrapBatchPhase.VERIFYING,
                        batch_number=batch_number,
                        records_processed=records_processed_before_batch,
                        total_records=record_count,
                        elapsed_seconds=time.perf_counter() - started,
                        last_identity=last_identity,
                    )
                    try:
                        relational_verify = self.dependencies.relational.verify_batch(relational_batch)
                        vector_verify = self.dependencies.vector.verify_batch(vector_batch)
                        assert_verification_complete(
                            batch_number=batch_number,
                            relational_result=relational_verify,
                            vector_result=vector_verify,
                        )
                    except StorageBootstrapIntegrityError as exc:
                        failed_batches += 1
                        terminal_failure = BootstrapFailure(
                            category=BootstrapFailureCategory.INTEGRITY_FAILED,
                            detail=str(exc),
                            batch_number=batch_number,
                        )
                        break

                next_checkpoint = advance_checkpoint_after_batch(
                    checkpoint_state,
                    batch_number=batch_number,
                    last_global_row_index=relational_batch.records[-1].global_row_index,
                    last_identity=last_identity,
                    updated_at_utc=utc_now_iso(),
                )
                try:
                    checkpoint_state = self.dependencies.checkpoint_store.commit_batch(
                        run_identity=run_identity,
                        expected_revision=checkpoint_state.state_revision,
                        checkpoint=next_checkpoint,
                    )
                except (
                    CheckpointConcurrentModification,
                    CheckpointPersistenceError,
                    CheckpointCorrupt,
                    CheckpointNotFound,
                ) as exc:
                    failed_batches += 1
                    terminal_failure = BootstrapFailure(
                        category=BootstrapFailureCategory.CHECKPOINT_FAILED,
                        detail=str(exc),
                        batch_number=batch_number,
                    )
                    break

                committed_batches += 1
                total_relational_written += relational_result.successful_count
                total_vectors_written += vector_result.successful_count
                last_committed_global_row_index = relational_batch.records[-1].global_row_index

                self._emit_progress(
                    progress_sink,
                    phase=BootstrapBatchPhase.COMMITTED,
                    batch_number=batch_number,
                    records_processed=previously_committed_records + total_relational_written,
                    total_records=record_count,
                    elapsed_seconds=time.perf_counter() - started,
                    last_identity=last_identity,
                )
        except (DataPackReaderError, StorageBootstrapIdentityError) as exc:
            terminal_failure = BootstrapFailure(
                category=(
                    BootstrapFailureCategory.IDENTITY_MISMATCH
                    if isinstance(exc, StorageBootstrapIdentityError)
                    else BootstrapFailureCategory.PRECONDITION_FAILED
                ),
                detail=str(exc),
            )

        if terminal_failure is None and records_streamed != record_count:
            terminal_failure = BootstrapFailure(
                category=BootstrapFailureCategory.PRECONDITION_FAILED,
                detail=(
                    "DATA_PACK_RECORD_COUNT_MISMATCH: "
                    f"manifest declares {record_count}, streamed {records_streamed}"
                ),
            )

        status = self._resolve_final_status(
            record_count=record_count,
            committed_batches=committed_batches,
            failed_batches=failed_batches,
            previously_committed_records=previously_committed_records,
            final_committed_record_count=checkpoint_state.committed_record_count,
            terminal_failure=terminal_failure,
        )

        return BootstrapResult(
            status=status,
            plan=plan,
            total_expected_records=record_count,
            total_relational_written=total_relational_written,
            total_vectors_written=total_vectors_written,
            committed_batches=committed_batches,
            failed_batches=failed_batches,
            last_committed_global_row_index=last_committed_global_row_index,
            failure=terminal_failure,
            resumed_from_batch=resumed_from_batch,
            previously_committed_batches=previously_committed_batches,
            previously_committed_records=previously_committed_records,
        )

    def plan(self, request: BootstrapRequest) -> BootstrapResult:
        plan_request = BootstrapRequest(
            artifact_root=request.artifact_root,
            relational_target=request.relational_target,
            vector_target=request.vector_target,
            batch_size=request.batch_size,
            resume_mode=request.resume_mode,
            verification_mode=request.verification_mode,
            plan_only=True,
        )
        return self.run(plan_request)

    def _prepare_checkpoint(
        self,
        *,
        request: BootstrapRequest,
        manifest: DataPackManifest,
        run_identity: BootstrapRunIdentity,
        plan: BootstrapPlan,
    ) -> tuple[BootstrapCheckpointState, BootstrapResumeDecision | BootstrapFailure]:
        if request.resume_mode is ResumeMode.FRESH:
            existing = self.dependencies.checkpoint_store.load(run_identity)
            if existing is not None:
                return existing, BootstrapFailure(
                    category=BootstrapFailureCategory.CHECKPOINT_FAILED,
                    detail="checkpoint already exists for FRESH run",
                )
            initial_state = initial_checkpoint_state(
                run_identity=run_identity,
                plan=plan,
                updated_at_utc=utc_now_iso(),
            )
            try:
                checkpoint_state = self.dependencies.checkpoint_store.initialize(
                    run_identity,
                    initial_state,
                )
            except CheckpointAlreadyExists as exc:
                return initial_state, BootstrapFailure(
                    category=BootstrapFailureCategory.CHECKPOINT_FAILED,
                    detail=str(exc),
                )
            except (CheckpointPersistenceError, CheckpointCorrupt) as exc:
                return initial_state, BootstrapFailure(
                    category=BootstrapFailureCategory.CHECKPOINT_FAILED,
                    detail=str(exc),
                )
            return checkpoint_state, BootstrapResumeDecision(
                start_batch_number=0,
                committed_batch_count=0,
                committed_record_count=0,
            )

        loaded = self.dependencies.checkpoint_store.load(run_identity)
        if loaded is None:
            return initial_checkpoint_state(
                run_identity=run_identity,
                plan=plan,
                updated_at_utc=utc_now_iso(),
            ), BootstrapFailure(
                category=BootstrapFailureCategory.CHECKPOINT_FAILED,
                detail="checkpoint not found for RESUME run",
            )
        try:
            compatibility = validate_checkpoint_compatibility(
                run_identity=run_identity,
                checkpoint=loaded,
                manifest=manifest,
            )
        except BootstrapCheckpointError as exc:
            return loaded, BootstrapFailure(
                category=BootstrapFailureCategory.CHECKPOINT_FAILED,
                detail=str(exc),
            )
        if not compatibility.is_compatible:
            return loaded, BootstrapFailure(
                category=BootstrapFailureCategory.CHECKPOINT_FAILED,
                detail=compatibility.reason or "checkpoint incompatible",
            )
        try:
            resume_decision = compute_resume_decision(loaded)
        except BootstrapCheckpointError as exc:
            return loaded, BootstrapFailure(
                category=BootstrapFailureCategory.CHECKPOINT_FAILED,
                detail=str(exc),
            )
        return loaded, resume_decision

    def _validate_request(self, request: BootstrapRequest) -> None:
        if request.batch_size.value <= 0:
            raise StorageBootstrapPreconditionError("batch_size must be > 0")

    def _failed_result(
        self,
        *,
        request: BootstrapRequest,
        record_count: int,
        failure: BootstrapFailure,
    ) -> BootstrapResult:
        plan = compute_bootstrap_plan(
            record_count=record_count,
            batch_size=request.batch_size.value,
            relational_target=request.relational_target,
            vector_target=request.vector_target,
        )
        return BootstrapResult(
            status=BootstrapFinalStatus.FAILED,
            plan=plan,
            total_expected_records=record_count,
            total_relational_written=0,
            total_vectors_written=0,
            committed_batches=0,
            failed_batches=0,
            last_committed_global_row_index=None,
            failure=failure,
        )

    @staticmethod
    def _resolve_final_status(
        *,
        record_count: int,
        committed_batches: int,
        failed_batches: int,
        previously_committed_records: int,
        final_committed_record_count: int,
        terminal_failure: BootstrapFailure | None,
    ) -> BootstrapFinalStatus:
        if record_count == 0:
            return BootstrapFinalStatus.SUCCESS
        if terminal_failure is not None:
            if final_committed_record_count == 0:
                return BootstrapFinalStatus.FAILED
            if final_committed_record_count < record_count:
                return BootstrapFinalStatus.PARTIAL
            return BootstrapFinalStatus.SUCCESS
        if final_committed_record_count == record_count and failed_batches == 0:
            return BootstrapFinalStatus.SUCCESS
        if committed_batches > 0 or previously_committed_records > 0:
            return BootstrapFinalStatus.PARTIAL
        return BootstrapFinalStatus.FAILED

    @staticmethod
    def _emit_progress(
        progress_sink: BootstrapProgressSinkPort | None,
        *,
        phase: BootstrapBatchPhase,
        batch_number: int,
        records_processed: int,
        total_records: int,
        elapsed_seconds: float,
        last_identity: str | None,
    ) -> None:
        if progress_sink is None:
            return
        progress_sink.emit(
            BootstrapProgress(
                phase=phase,
                batch_number=batch_number,
                records_processed=records_processed,
                total_records=total_records,
                elapsed_seconds=elapsed_seconds,
                last_identity=last_identity,
            )
        )
