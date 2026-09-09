"""Typed operator evidence writer for VPI full storage load runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.manifest import (
    DataPackManifest,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.contracts import (
    BootstrapFinalStatus,
    BootstrapResult,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.operator.config import (
    OPERATOR_SCHEMA_VERSION,
    PRODUCTION_COMPOSITION,
    StorageLoadOperatorConfig,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.operator.errors import (
    StorageLoadOperatorEvidenceError,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.operator.exit_codes import (
    StorageLoadOperatorExitCode,
)


@dataclass(frozen=True, slots=True)
class OperatorRunMetadata:
    operator_schema_version: str
    run_id: str
    started_at: str
    git_sha: str | None
    artifact_root: str
    data_pack_content_identity: str
    data_pack_version: str
    record_count: int
    batch_size: int
    resume_mode: str
    verification_mode: str
    relational_target: str
    vector_target: str
    checkpoint_root: str
    evidence_root: str
    composition: str

    def to_json(self) -> str:
        return json.dumps(
            {
                "operator_schema_version": self.operator_schema_version,
                "run_id": self.run_id,
                "started_at": self.started_at,
                "git_sha": self.git_sha,
                "artifact_root": self.artifact_root,
                "data_pack_content_identity": self.data_pack_content_identity,
                "data_pack_version": self.data_pack_version,
                "record_count": self.record_count,
                "batch_size": self.batch_size,
                "resume_mode": self.resume_mode,
                "verification_mode": self.verification_mode,
                "relational_target": self.relational_target,
                "vector_target": self.vector_target,
                "checkpoint_root": self.checkpoint_root,
                "evidence_root": self.evidence_root,
                "composition": self.composition,
            },
            indent=2,
            sort_keys=True,
        )


@dataclass(frozen=True, slots=True)
class OperatorFinalReport:
    status: str
    exit_code: int
    total_expected_records: int
    previously_committed_records: int
    records_handled_this_invocation: int
    total_relational_written: int
    total_vectors_written: int
    committed_batches: int
    failed_batches: int
    last_committed_global_row_index: int | None
    elapsed_seconds: float
    checkpoint_status: str
    failure_category: str | None = None
    failure_detail: str | None = None

    def to_json(self) -> str:
        return json.dumps(
            {
                "status": self.status,
                "exit_code": self.exit_code,
                "total_expected_records": self.total_expected_records,
                "previously_committed_records": self.previously_committed_records,
                "records_handled_this_invocation": self.records_handled_this_invocation,
                "total_relational_written": self.total_relational_written,
                "total_vectors_written": self.total_vectors_written,
                "committed_batches": self.committed_batches,
                "failed_batches": self.failed_batches,
                "last_committed_global_row_index": self.last_committed_global_row_index,
                "elapsed_seconds": round(self.elapsed_seconds, 3),
                "checkpoint_status": self.checkpoint_status,
                "failure_category": self.failure_category,
                "failure_detail": self.failure_detail,
            },
            indent=2,
            sort_keys=True,
        )


@dataclass(slots=True)
class OperatorEvidenceWriter:
    evidence_root: Path
    run_id: str
    started_at: str
    git_sha: str | None

    @classmethod
    def initialize(
        cls,
        *,
        evidence_root: Path,
        run_id: str,
        git_sha: str | None,
    ) -> OperatorEvidenceWriter:
        try:
            evidence_root.mkdir(parents=True, exist_ok=True)
            probe = evidence_root / ".write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
        except OSError as exc:
            raise StorageLoadOperatorEvidenceError(
                f"evidence root is not writable: {evidence_root}"
            ) from exc
        return cls(
            evidence_root=evidence_root,
            run_id=run_id,
            started_at=datetime.now(tz=UTC).replace(microsecond=0).isoformat(),
            git_sha=git_sha,
        )

    @property
    def run_path(self) -> Path:
        return self.evidence_root / "run.json"

    @property
    def progress_path(self) -> Path:
        return self.evidence_root / "progress.jsonl"

    @property
    def final_report_path(self) -> Path:
        return self.evidence_root / "final-report.json"

    def write_run_metadata(
        self,
        *,
        config: StorageLoadOperatorConfig,
        manifest: DataPackManifest,
    ) -> None:
        metadata = OperatorRunMetadata(
            operator_schema_version=OPERATOR_SCHEMA_VERSION,
            run_id=self.run_id,
            started_at=self.started_at,
            git_sha=self.git_sha,
            artifact_root=str(config.artifact_root.resolve()),
            data_pack_content_identity=manifest.content_identity,
            data_pack_version=manifest.data_pack_version,
            record_count=manifest.record_count,
            batch_size=config.batch_size.value,
            resume_mode=config.resume_mode.value,
            verification_mode=config.verification_mode.value,
            relational_target=str(config.relational_target),
            vector_target=str(config.vector_target),
            checkpoint_root=str(config.checkpoint_root.resolve()),
            evidence_root=str(config.evidence_root.resolve()),
            composition=PRODUCTION_COMPOSITION,
        )
        self.run_path.write_text(metadata.to_json() + "\n", encoding="utf-8")

    def write_final_report(
        self,
        *,
        result: BootstrapResult,
        exit_code: StorageLoadOperatorExitCode,
        elapsed_seconds: float,
        interrupted: bool = False,
    ) -> None:
        failure_category = None
        failure_detail = None
        if result.failure is not None:
            failure_category = result.failure.category.value
            failure_detail = result.failure.detail
        checkpoint_status = "complete" if result.status is BootstrapFinalStatus.SUCCESS else "incomplete"
        if interrupted:
            checkpoint_status = "interrupted"
        report = OperatorFinalReport(
            status=result.status.value,
            exit_code=int(exit_code),
            total_expected_records=result.total_expected_records,
            previously_committed_records=result.previously_committed_records,
            records_handled_this_invocation=result.total_relational_written,
            total_relational_written=result.total_relational_written,
            total_vectors_written=result.total_vectors_written,
            committed_batches=result.committed_batches,
            failed_batches=result.failed_batches,
            last_committed_global_row_index=result.last_committed_global_row_index,
            elapsed_seconds=elapsed_seconds,
            checkpoint_status=checkpoint_status,
            failure_category=failure_category,
            failure_detail=failure_detail,
        )
        self.final_report_path.write_text(report.to_json() + "\n", encoding="utf-8")
