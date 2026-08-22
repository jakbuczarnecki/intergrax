# © Artur Czarnecki. All rights reserved.

"""Canonical publication of verified Platform Proof artifacts (PP-PUBLISH-1)."""

from __future__ import annotations

import shutil
import uuid
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from scripts.proof.intergrax_platform_proof_artifact_verifier import (
    ArtifactVerificationStatus,
    ProofArtifactVerificationSummary,
    _check_regular_file,
    _resolve_artifact_path,
)
from scripts.proof.intergrax_platform_proof_evidence_verifier import (
    EvidenceVerificationResult,
)
from scripts.proof.intergrax_platform_proof_execution import ProofExecutionSpec
from scripts.proof.intergrax_proof_contracts import (
    EvidenceVerificationStatus,
    ProofRunResult,
    ProofStatus,
)

CANONICAL_OUTPUT_DIR_NAME = "output"


class PublicationStatus(StrEnum):
    PUBLISHED = "PUBLISHED"
    SKIPPED = "SKIPPED"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class PublicationResult:
    status: PublicationStatus
    published_output_path: Path | None
    diagnostic_code: str
    diagnostic_summary: str


def _path_within_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def canonical_proof_output_directory(spec: ProofExecutionSpec) -> Path:
    """Return ``<package_root>/output`` for descriptor-backed proofs."""
    if spec.package_root is None:
        raise ValueError("descriptor-backed proof requires package_root")
    package_root = spec.package_root.resolve()
    output = (package_root / CANONICAL_OUTPUT_DIR_NAME).resolve()
    if not _path_within_root(output, package_root):
        raise ValueError("canonical output path escapes package root")
    return output


def _non_publishable_transport_diagnostic(result: ProofRunResult) -> bool:
    return result.diagnostic_summary in {
        "timeout",
        "missing_executable",
        "missing_proof_artifact_directory",
    }


def should_publish_canonical_output(
    *,
    transport_result: ProofRunResult,
    execution_spec: ProofExecutionSpec,
    artifact_summary: ProofArtifactVerificationSummary | None,
    evidence_verification: EvidenceVerificationResult | None,
) -> bool:
    if execution_spec.package_root is None:
        return False
    if not execution_spec.expected_artifacts:
        return False
    if _non_publishable_transport_diagnostic(transport_result):
        return False
    if artifact_summary is None or not artifact_summary.passed:
        return False
    if execution_spec.evidence_required:
        if evidence_verification is None:
            return False
        if evidence_verification.status != EvidenceVerificationStatus.PASS:
            return False
    return True


def _cleanup_publish_temp_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def _install_staged_canonical_output(
    *,
    package_root: Path,
    canonical_output: Path,
    staging_output: Path,
) -> PublicationResult:
    """Swap staged NEW into canonical output; rollback OLD on install failure."""
    backup_root = package_root / f".proof-publish-backup-{uuid.uuid4().hex}"
    if not _path_within_root(backup_root, package_root):
        return PublicationResult(
            status=PublicationStatus.FAILED,
            published_output_path=None,
            diagnostic_code="output_path_escape",
            diagnostic_summary="backup path escapes package root",
        )

    if not canonical_output.exists():
        try:
            staging_output.rename(canonical_output)
        except OSError as exc:
            return PublicationResult(
                status=PublicationStatus.FAILED,
                published_output_path=None,
                diagnostic_code="canonical_output_publish_failed",
                diagnostic_summary=str(exc),
            )
        return PublicationResult(
            status=PublicationStatus.PUBLISHED,
            published_output_path=canonical_output,
            diagnostic_code="canonical_output_published",
            diagnostic_summary="canonical_output_published",
        )

    try:
        canonical_output.rename(backup_root)
    except OSError as exc:
        return PublicationResult(
            status=PublicationStatus.FAILED,
            published_output_path=None,
            diagnostic_code="canonical_output_publish_failed",
            diagnostic_summary=str(exc),
        )

    try:
        staging_output.rename(canonical_output)
    except OSError as exc:
        try:
            if canonical_output.exists():
                shutil.rmtree(canonical_output, ignore_errors=True)
            backup_root.rename(canonical_output)
        except OSError as rollback_exc:
            return PublicationResult(
                status=PublicationStatus.FAILED,
                published_output_path=None,
                diagnostic_code="canonical_output_publish_rollback_failed",
                diagnostic_summary=f"install_failed={exc}; rollback_failed={rollback_exc}",
            )
        return PublicationResult(
            status=PublicationStatus.FAILED,
            published_output_path=None,
            diagnostic_code="canonical_output_publish_failed",
            diagnostic_summary=str(exc),
        )

    _cleanup_publish_temp_directory(backup_root)
    return PublicationResult(
        status=PublicationStatus.PUBLISHED,
        published_output_path=canonical_output,
        diagnostic_code="canonical_output_published",
        diagnostic_summary="canonical_output_published",
    )


def _copy_declared_artifact(
    *,
    candidate_directory: Path,
    staging_output: Path,
    relative_path: str,
) -> tuple[bool, str]:
    resolved, path_error = _resolve_artifact_path(candidate_directory, relative_path)
    if path_error is not None or resolved is None:
        return False, path_error or "invalid_artifact_path"
    file_ok, file_code, file_summary = _check_regular_file(resolved)
    if not file_ok:
        return False, file_summary or file_code
    destination = staging_output / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    shutil.copy2(resolved, destination)
    if destination.read_bytes() != resolved.read_bytes():
        return False, "artifact_copy_mismatch"
    return True, ""


def publish_verified_proof_artifacts(
    *,
    candidate_directory: Path,
    spec: ProofExecutionSpec,
    artifact_summary: ProofArtifactVerificationSummary,
) -> PublicationResult:
    """Copy verified declared artifacts into ``package_root/output`` atomically-ish."""
    if spec.package_root is None:
        return PublicationResult(
            status=PublicationStatus.FAILED,
            published_output_path=None,
            diagnostic_code="missing_package_root",
            diagnostic_summary="descriptor-backed proof requires package_root",
        )

    package_root = spec.package_root.resolve()
    canonical_output = canonical_proof_output_directory(spec)
    if not _path_within_root(canonical_output, package_root):
        return PublicationResult(
            status=PublicationStatus.FAILED,
            published_output_path=None,
            diagnostic_code="output_path_escape",
            diagnostic_summary="canonical output path escapes package root",
        )

    publishable = [
        result
        for result in artifact_summary.results
        if result.status == ArtifactVerificationStatus.PASS and result.resolved_path is not None
    ]
    if not publishable:
        return PublicationResult(
            status=PublicationStatus.FAILED,
            published_output_path=None,
            diagnostic_code="no_publishable_artifacts",
            diagnostic_summary="no verified artifacts available for publication",
        )

    staging_root = package_root / f".proof-publish-{uuid.uuid4().hex}"
    staging_output = staging_root / CANONICAL_OUTPUT_DIR_NAME
    try:
        staging_output.mkdir(parents=True, exist_ok=True)
        for result in publishable:
            relative = result.relative_path.strip().replace("\\", "/")
            if ".." in relative.split("/") or relative.startswith("/"):
                return PublicationResult(
                    status=PublicationStatus.FAILED,
                    published_output_path=None,
                    diagnostic_code="artifact_path_outside_root",
                    diagnostic_summary="declared artifact path escapes output root",
                )
            copied, error = _copy_declared_artifact(
                candidate_directory=candidate_directory,
                staging_output=staging_output,
                relative_path=relative,
            )
            if not copied:
                return PublicationResult(
                    status=PublicationStatus.FAILED,
                    published_output_path=None,
                    diagnostic_code="canonical_output_publish_failed",
                    diagnostic_summary=error or "canonical_output_publish_failed",
                )

        canonical_output.parent.mkdir(parents=True, exist_ok=True)
        return _install_staged_canonical_output(
            package_root=package_root,
            canonical_output=canonical_output,
            staging_output=staging_output,
        )
    except OSError as exc:
        return PublicationResult(
            status=PublicationStatus.FAILED,
            published_output_path=None,
            diagnostic_code="canonical_output_publish_failed",
            diagnostic_summary=str(exc),
        )
    finally:
        _cleanup_publish_temp_directory(staging_root)


def apply_canonical_publication(
    result: ProofRunResult,
    *,
    transport_result: ProofRunResult,
    execution_spec: ProofExecutionSpec | None,
    candidate_directory: Path | None,
    artifact_summary: ProofArtifactVerificationSummary | None,
    evidence_verification: EvidenceVerificationResult | None,
) -> ProofRunResult:
    if execution_spec is None or candidate_directory is None:
        return result
    if not should_publish_canonical_output(
        transport_result=transport_result,
        execution_spec=execution_spec,
        artifact_summary=artifact_summary,
        evidence_verification=evidence_verification,
    ):
        return result
    assert artifact_summary is not None
    publication = publish_verified_proof_artifacts(
        candidate_directory=candidate_directory,
        spec=execution_spec,
        artifact_summary=artifact_summary,
    )
    if publication.status != PublicationStatus.PUBLISHED:
        return result.model_copy(
            update={
                "status": ProofStatus.FAIL,
                "diagnostic_summary": publication.diagnostic_code,
            }
        )
    return result
