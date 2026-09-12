# © Artur Czarnecki. All rights reserved.

"""Artifact validation contract for R6 multi-model summary sessions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from testing_support.decision_e2e.local_qualification_session.artifact_finalization_contract import (
    QualificationArtifactFinalizationContract,
)

MATRIX_ARTIFACT_SCHEMA_VERSION = "r6-summary-v1"
MATRIX_ARTIFACT_PRODUCER = "MultiModelQualificationArtifactProvider"


class MatrixArtifactStatus(StrEnum):
    COMPLETE = "COMPLETE"
    INCOMPLETE = "INCOMPLETE"
    INVALID = "INVALID"


@dataclass(frozen=True, slots=True)
class MatrixArtifactValidationResult:
    status: MatrixArtifactStatus
    missing: tuple[str, ...]
    manifest_valid: bool
    checksum_valid: bool


REQUIRED_MATRIX_SUMMARY_ARTIFACTS: tuple[str, ...] = (
    "runs.json",
    "summary.json",
    "analysis.json",
    "local_model_profile.json",
    "artifact-manifest.txt",
    "checksum.json",
    "final-report.md",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


class QualificationArtifactProvider:
    """Persistence, checksum, and manifest contract for matrix summary output."""

    @staticmethod
    def required_artifact_names() -> tuple[str, ...]:
        return REQUIRED_MATRIX_SUMMARY_ARTIFACTS

    @staticmethod
    def write_checksum_json(
        output_dir: Path,
        *,
        task_id: str,
        repository_head_sha: str,
        source_fingerprint: str,
        matrix_version: str,
    ) -> None:
        file_digests = {
            name: _sha256_file(output_dir / name)
            for name in REQUIRED_MATRIX_SUMMARY_ARTIFACTS
            if name != "checksum.json" and (output_dir / name).is_file()
        }
        payload = {
            "schema_version": MATRIX_ARTIFACT_SCHEMA_VERSION,
            "producer": MATRIX_ARTIFACT_PRODUCER,
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "task_id": task_id,
            "matrix_version": matrix_version,
            "source_identity": {
                "repository_head_sha": repository_head_sha,
                "semantic_fingerprint": source_fingerprint,
            },
            "artifacts": file_digests,
        }
        (output_dir / "checksum.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def write_manifest(output_dir: Path, files: tuple[str, ...]) -> None:
        lines = [f"{name} sha256:{_sha256_file(output_dir / name)}" for name in sorted(files)]
        body = "\n".join(lines) + ("\n" if lines else "")
        self_digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
        manifest_body = body + f"artifact-manifest.txt sha256:{self_digest}\n"
        (output_dir / "artifact-manifest.txt").write_text(manifest_body, encoding="utf-8")

    @classmethod
    def validate_session(cls, output_dir: Path) -> MatrixArtifactValidationResult:
        missing = tuple(
            name
            for name in REQUIRED_MATRIX_SUMMARY_ARTIFACTS
            if not (output_dir / name).is_file()
        )
        manifest_valid = False
        checksum_valid = False
        if "artifact-manifest.txt" not in missing:
            try:
                QualificationArtifactFinalizationContract.validate_manifest_checksums(
                    output_dir
                )
                manifest_valid = True
            except ValueError:
                manifest_valid = False
        checksum_path = output_dir / "checksum.json"
        if checksum_path.is_file():
            try:
                payload = json.loads(checksum_path.read_text(encoding="utf-8"))
                checksum_valid = (
                    payload.get("schema_version") == MATRIX_ARTIFACT_SCHEMA_VERSION
                    and payload.get("producer") == MATRIX_ARTIFACT_PRODUCER
                    and isinstance(payload.get("artifacts"), dict)
                )
            except (json.JSONDecodeError, OSError):
                checksum_valid = False
        if missing or not manifest_valid or not checksum_valid:
            status = MatrixArtifactStatus.INCOMPLETE if missing else MatrixArtifactStatus.INVALID
        else:
            status = MatrixArtifactStatus.COMPLETE
        return MatrixArtifactValidationResult(
            status=status,
            missing=missing,
            manifest_valid=manifest_valid,
            checksum_valid=checksum_valid,
        )


__all__ = [
    "MATRIX_ARTIFACT_PRODUCER",
    "MATRIX_ARTIFACT_SCHEMA_VERSION",
    "MatrixArtifactStatus",
    "MatrixArtifactValidationResult",
    "QualificationArtifactProvider",
    "REQUIRED_MATRIX_SUMMARY_ARTIFACTS",
]
