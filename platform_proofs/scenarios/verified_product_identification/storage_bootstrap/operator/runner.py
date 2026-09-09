"""Operator runner composing existing storage bootstrap components."""

from __future__ import annotations

import signal
import subprocess
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass

from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.batching import (
    compute_bootstrap_plan,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.checkpoint.compatibility import (
    build_run_identity,
    compute_resume_decision,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.checkpoint.filesystem_store import (
    FilesystemBootstrapCheckpointStore,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.contracts import (
    BootstrapFinalStatus,
    BootstrapResult,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.operator.composition import (
    StorageLoadOperatorRuntime,
    build_storage_load_operator_runtime,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.operator.config import (
    OperatorRunMode,
    StorageLoadOperatorConfig,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.operator.errors import (
    StorageLoadOperatorError,
    StorageLoadOperatorEvidenceError,
    StorageLoadOperatorLockError,
    StorageLoadOperatorPreconditionError,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.operator.evidence import (
    OperatorEvidenceWriter,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.operator.exit_codes import (
    StorageLoadOperatorExitCode,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.operator.lock import (
    OperatorLock,
    build_operator_lock_metadata,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.operator.preconditions import (
    resolve_operator_preconditions,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.operator.progress import (
    JsonlOperatorProgressSink,
)


@dataclass(frozen=True, slots=True)
class StorageLoadOperatorOutcome:
    exit_code: StorageLoadOperatorExitCode
    result: BootstrapResult | None
    message: str | None = None


def _resolve_git_sha() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    return completed.stdout.strip() or None


def _map_result_exit_code(result: BootstrapResult) -> StorageLoadOperatorExitCode:
    if result.status is BootstrapFinalStatus.SUCCESS:
        return StorageLoadOperatorExitCode.SUCCESS
    return StorageLoadOperatorExitCode.LOAD_FAILED


@dataclass(slots=True)
class StorageLoadOperatorRunner:
    runtime_builder: Callable[[StorageLoadOperatorConfig], StorageLoadOperatorRuntime] | None = None
    validate_providers: bool = True

    def run(self, config: StorageLoadOperatorConfig) -> StorageLoadOperatorOutcome:
        started = time.perf_counter()
        run_id = uuid.uuid4().hex
        try:
            evidence = OperatorEvidenceWriter.initialize(
                evidence_root=config.evidence_root,
                run_id=run_id,
                git_sha=_resolve_git_sha(),
            )
        except StorageLoadOperatorEvidenceError as exc:
            return StorageLoadOperatorOutcome(
                exit_code=StorageLoadOperatorExitCode.PRECONDITION_ERROR,
                result=None,
                message=str(exc),
            )
        try:
            resolved = resolve_operator_preconditions(
                config,
                validate_providers=self.validate_providers,
            )
        except StorageLoadOperatorPreconditionError as exc:
            return StorageLoadOperatorOutcome(
                exit_code=StorageLoadOperatorExitCode.PRECONDITION_ERROR,
                result=None,
                message=str(exc),
            )

        evidence.write_run_metadata(config=config, manifest=resolved.manifest)

        if config.run_mode is OperatorRunMode.PLAN:
            runtime = self._build_runtime(config)
            try:
                result = runtime.service.run(resolved.bootstrap_request)
            finally:
                runtime.close()
            elapsed = time.perf_counter() - started
            exit_code = _map_result_exit_code(result)
            evidence.write_final_report(
                result=result,
                exit_code=exit_code,
                elapsed_seconds=elapsed,
            )
            return StorageLoadOperatorOutcome(exit_code=exit_code, result=result)

        lock = OperatorLock(
            checkpoint_root=config.checkpoint_root,
            metadata=build_operator_lock_metadata(
                artifact_content_identity=resolved.manifest.content_identity,
                run_id=run_id,
            ),
        )
        try:
            lock.acquire()
        except StorageLoadOperatorLockError as exc:
            return StorageLoadOperatorOutcome(
                exit_code=StorageLoadOperatorExitCode.PRECONDITION_ERROR,
                result=None,
                message=str(exc),
            )

        runtime = self._build_runtime(config)
        progress_sink = JsonlOperatorProgressSink(progress_path=evidence.progress_path)
        interrupted = False
        previous_sigterm = signal.getsignal(signal.SIGTERM)

        def _handle_sigterm(signum: int, _frame: object) -> None:
            raise KeyboardInterrupt(f"received signal {signum}")

        signal.signal(signal.SIGTERM, _handle_sigterm)
        try:
            try:
                result = runtime.service.run(
                    resolved.bootstrap_request,
                    progress_sink=progress_sink,
                )
            except KeyboardInterrupt:
                interrupted = True
                result = self._interrupted_result(
                    config=config,
                    resolved=resolved,
                )
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm)
            runtime.close()
            lock.release()

        elapsed = time.perf_counter() - started
        if interrupted:
            evidence.write_final_report(
                result=result,
                exit_code=StorageLoadOperatorExitCode.INTERRUPTED,
                elapsed_seconds=elapsed,
                interrupted=True,
            )
            return StorageLoadOperatorOutcome(
                exit_code=StorageLoadOperatorExitCode.INTERRUPTED,
                result=result,
                message="operator interrupted",
            )

        exit_code = _map_result_exit_code(result)
        evidence.write_final_report(
            result=result,
            exit_code=exit_code,
            elapsed_seconds=elapsed,
        )
        return StorageLoadOperatorOutcome(exit_code=exit_code, result=result)

    def _build_runtime(self, config: StorageLoadOperatorConfig) -> StorageLoadOperatorRuntime:
        if self.runtime_builder is not None:
            return self.runtime_builder(config)
        return build_storage_load_operator_runtime(config)

    @staticmethod
    def _interrupted_result(
        *,
        config: StorageLoadOperatorConfig,
        resolved: object,
    ) -> BootstrapResult:
        from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.operator.preconditions import (
            ResolvedOperatorPreconditions,
        )

        assert isinstance(resolved, ResolvedOperatorPreconditions)
        plan = compute_bootstrap_plan(
            record_count=resolved.manifest.record_count,
            batch_size=config.batch_size.value,
            relational_target=config.relational_target,
            vector_target=config.vector_target,
        )
        checkpoint_store = FilesystemBootstrapCheckpointStore(config.checkpoint_root)
        run_identity = build_run_identity(
            manifest=resolved.manifest,
            request=resolved.bootstrap_request,
        )
        checkpoint = checkpoint_store.load(run_identity)
        if checkpoint is None:
            committed_batches = 0
            committed_records = 0
            last_committed_global_row_index = None
        else:
            resume_decision = compute_resume_decision(checkpoint)
            committed_batches = resume_decision.committed_batch_count
            committed_records = resume_decision.committed_record_count
            last_committed_global_row_index = checkpoint.last_committed_global_row_index
        return BootstrapResult(
            status=BootstrapFinalStatus.PARTIAL,
            plan=plan,
            total_expected_records=resolved.manifest.record_count,
            total_relational_written=0,
            total_vectors_written=0,
            committed_batches=0,
            failed_batches=0,
            last_committed_global_row_index=last_committed_global_row_index,
            failure=None,
            previously_committed_batches=committed_batches,
            previously_committed_records=committed_records,
        )


def run_storage_load_operator(
    config: StorageLoadOperatorConfig,
) -> StorageLoadOperatorOutcome:
    try:
        return StorageLoadOperatorRunner().run(config)
    except StorageLoadOperatorError as exc:
        return StorageLoadOperatorOutcome(
            exit_code=StorageLoadOperatorExitCode.PRECONDITION_ERROR,
            result=None,
            message=str(exc) or exc.__class__.__name__,
        )
