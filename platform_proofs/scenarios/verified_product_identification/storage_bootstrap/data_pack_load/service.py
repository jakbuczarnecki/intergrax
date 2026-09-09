"""Provider-neutral Data Pack storage bootstrap application service.

Cross-store consistency model (PostgreSQL + Qdrant, MySQL + Qdrant, etc.):

- No distributed ACID transaction across relational and vector backends.
- Application coordinates at logical batch boundaries.
- Checkpoint advances only after relational write AND vector write AND verification succeed.
- If vector write fails after relational success, batch remains NOT COMMITTED.
- Retry is idempotent via adapter upsert / insert-if-absent semantics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.status import (
    DataPackStatus,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.batching import (
    compute_bootstrap_plan,
    iter_record_batches,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.contracts import (
    BootstrapBatchPhase,
    BootstrapFinalStatus,
    BootstrapProgress,
    BootstrapRequest,
    BootstrapResult,
    RelationalBatch,
    RelationalLoadRecord,
    RelationalTargetId,
    VectorBatch,
    VectorLoadRecord,
    VectorTargetId,
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
    PairedDataPackRecord,
    RelationalStorageLoadPort,
    VectorStorageLoadPort,
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
        manifest = self.dependencies.reader.read_manifest()
        if manifest.status is not DataPackStatus.READY:
            return self._failed_result(
                request=request,
                record_count=0,
                failure=BootstrapFailure(
                    category=BootstrapFailureCategory.PRECONDITION_FAILED,
                    detail=f"data pack status must be READY (got {manifest.status.value})",
                ),
            )

        paired_records = self._collect_paired_records()
        record_count = len(paired_records)
        plan = compute_bootstrap_plan(
            record_count=record_count,
            batch_size=request.batch_size.value,
            relational_target=request.relational_target,
            vector_target=request.vector_target,
        )

        if request.plan_only:
            return BootstrapResult(
                status=BootstrapFinalStatus.SUCCESS if record_count >= 0 else BootstrapFinalStatus.FAILED,
                plan=plan,
                total_expected_records=record_count,
                total_relational_written=0,
                total_vectors_written=0,
                committed_batches=0,
                failed_batches=0,
                last_committed_global_row_index=None,
                failure=None,
            )

        total_relational_written = 0
        total_vectors_written = 0
        committed_batches = 0
        failed_batches = 0
        last_committed_global_row_index: int | None = None
        terminal_failure: BootstrapFailure | None = None

        for batch_number, batch_pairs in iter_record_batches(
            paired_records,
            batch_size=request.batch_size.value,
        ):
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
            self._emit_progress(
                progress_sink,
                phase=BootstrapBatchPhase.RELATIONAL_WRITING,
                batch_number=batch_number,
                records_processed=committed_batches * request.batch_size.value,
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
                records_processed=committed_batches * request.batch_size.value,
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
                    records_processed=committed_batches * request.batch_size.value,
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

            committed_batches += 1
            total_relational_written += relational_result.successful_count
            total_vectors_written += vector_result.successful_count
            last_committed_global_row_index = relational_batch.records[-1].global_row_index

            self._emit_progress(
                progress_sink,
                phase=BootstrapBatchPhase.COMMITTED,
                batch_number=batch_number,
                records_processed=total_relational_written,
                total_records=record_count,
                elapsed_seconds=time.perf_counter() - started,
                last_identity=last_identity,
            )

        status = self._resolve_final_status(
            record_count=record_count,
            committed_batches=committed_batches,
            failed_batches=failed_batches,
            total_relational_written=total_relational_written,
            total_vectors_written=total_vectors_written,
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

    def _validate_request(self, request: BootstrapRequest) -> None:
        if request.batch_size.value <= 0:
            raise StorageBootstrapPreconditionError("batch_size must be > 0")

    def _collect_paired_records(self) -> tuple[PairedDataPackRecord, ...]:
        records = list(self.dependencies.reader.iter_paired_records())
        return tuple(sorted(records, key=lambda pair: pair.relational.global_row_index))

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
        total_relational_written: int,
        total_vectors_written: int,
        terminal_failure: BootstrapFailure | None,
    ) -> BootstrapFinalStatus:
        if terminal_failure is not None and committed_batches == 0:
            return BootstrapFinalStatus.FAILED
        if (
            committed_batches > 0
            and failed_batches > 0
            and total_relational_written < record_count
        ):
            return BootstrapFinalStatus.PARTIAL
        if (
            committed_batches > 0
            and failed_batches == 0
            and total_relational_written == record_count
            and total_vectors_written == record_count
        ):
            return BootstrapFinalStatus.SUCCESS
        if terminal_failure is not None:
            return BootstrapFinalStatus.PARTIAL if committed_batches > 0 else BootstrapFinalStatus.FAILED
        if record_count == 0:
            return BootstrapFinalStatus.SUCCESS
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
