# © Artur Czarnecki. All rights reserved.

"""Generic verification of declared Platform Proof artifacts (PP-SUITE-4).

Structural artifact contract only — semantic evidence validation remains in
``intergrax_platform_proof_evidence_verifier`` (PP-SUITE-3); HTML report
semantics belong to PP-SUITE-5.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from scripts.proof.intergrax_platform_proof_descriptor import ExpectedArtifactKind
from scripts.proof.intergrax_platform_proof_execution import ProofExecutionSpec
from scripts.proof.intergrax_proof_contracts import (
    ArtifactVerificationStatus as ProofArtifactVerificationStatus,
    ProofRunResult,
    ProofStatus,
)


class ArtifactVerificationStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    MISSING = "MISSING"
    INVALID = "INVALID"
    OPTIONAL_MISSING = "OPTIONAL_MISSING"


@dataclass(frozen=True, slots=True)
class ArtifactVerificationResult:
    kind: ExpectedArtifactKind
    relative_path: str
    required: bool
    status: ArtifactVerificationStatus
    resolved_path: Path | None
    diagnostic_code: str
    diagnostic_summary: str


@dataclass(frozen=True, slots=True)
class ProofArtifactVerificationSummary:
    passed: bool
    results: tuple[ArtifactVerificationResult, ...]


def _path_within_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _bounded_message(prefix: str, detail: str, *, limit: int = 200) -> str:
    text = f"{prefix}: {detail}"
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _resolve_artifact_path(
    proof_artifact_directory: Path,
    relative_path: str,
) -> tuple[Path | None, str | None]:
    """Resolve artifact path under proof artifact root; return (path, error_code)."""
    normalized = relative_path.strip().replace("\\", "/")
    if not normalized:
        return None, "invalid_artifact_path"
    if normalized.startswith("/") or _is_windows_absolute(normalized):
        return None, "artifact_path_outside_root"
    if ".." in normalized.split("/"):
        return None, "artifact_path_outside_root"

    root = proof_artifact_directory.resolve()
    candidate = (root / normalized).resolve()
    if not _path_within_root(candidate, root):
        return None, "artifact_path_outside_root"
    return candidate, None


def _is_windows_absolute(path: str) -> bool:
    return len(path) >= 2 and path[1] == ":"


def _check_regular_file(path: Path) -> tuple[bool, str, str]:
    """Return (ok, diagnostic_code, diagnostic_summary)."""
    try:
        if path.is_symlink():
            return False, "invalid_required_artifact", f"{path.name} must be a regular file"
        if path.is_dir():
            return False, "invalid_required_artifact", f"{path.name} is a directory, file expected"
        if not path.is_file():
            return False, "missing_required_artifact", f"{path.name} is missing"
        size = path.stat().st_size
        if size == 0:
            return False, "invalid_required_artifact", f"{path.name} is empty"
    except OSError as exc:
        return False, "invalid_required_artifact", _bounded_message(
            f"{path.name} stat failed",
            str(exc),
        )
    return True, "", ""


def _validate_domain_result_json(path: Path) -> tuple[bool, str]:
    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return False, f"{path.name} is not valid UTF-8"
    except OSError as exc:
        return False, _bounded_message(f"{path.name} read failed", str(exc))
    try:
        json.loads(raw)
    except json.JSONDecodeError:
        return False, f"{path.name} is not valid JSON"
    return True, ""


def _validate_report_html(path: Path) -> tuple[bool, str]:
    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return False, f"{path.name} is not valid UTF-8"
    except OSError as exc:
        return False, _bounded_message(f"{path.name} read failed", str(exc))
    if not raw.strip():
        return False, f"{path.name} is empty"
    return True, ""


def _verify_single_artifact(
    *,
    proof_artifact_directory: Path,
    kind: ExpectedArtifactKind,
    relative_path: str,
    required: bool,
) -> ArtifactVerificationResult:
    resolved, path_error = _resolve_artifact_path(proof_artifact_directory, relative_path)
    if path_error is not None:
        status = (
            ArtifactVerificationStatus.FAIL
            if required
            else ArtifactVerificationStatus.INVALID
        )
        return ArtifactVerificationResult(
            kind=kind,
            relative_path=relative_path,
            required=required,
            status=status,
            resolved_path=resolved,
            diagnostic_code=path_error,
            diagnostic_summary=path_error,
        )

    assert resolved is not None
    file_ok, file_code, file_summary = _check_regular_file(resolved)
    if not file_ok:
        if not required:
            if file_code == "missing_required_artifact":
                return ArtifactVerificationResult(
                    kind=kind,
                    relative_path=relative_path,
                    required=required,
                    status=ArtifactVerificationStatus.OPTIONAL_MISSING,
                    resolved_path=None,
                    diagnostic_code="optional_artifact_missing",
                    diagnostic_summary="optional artifact absent",
                )
        status = (
            ArtifactVerificationStatus.MISSING
            if file_code == "missing_required_artifact"
            else ArtifactVerificationStatus.INVALID
        )
        return ArtifactVerificationResult(
            kind=kind,
            relative_path=relative_path,
            required=required,
            status=status,
            resolved_path=resolved,
            diagnostic_code=(
                "missing_required_artifact"
                if file_code == "missing_required_artifact"
                else "invalid_required_artifact"
            ),
            diagnostic_summary=file_summary,
        )

    if kind == ExpectedArtifactKind.DOMAIN_RESULT_JSON:
        json_ok, json_summary = _validate_domain_result_json(resolved)
        if not json_ok:
            return ArtifactVerificationResult(
                kind=kind,
                relative_path=relative_path,
                required=required,
                status=ArtifactVerificationStatus.INVALID,
                resolved_path=resolved,
                diagnostic_code="invalid_required_artifact",
                diagnostic_summary=json_summary,
            )
    elif kind == ExpectedArtifactKind.REPORT_HTML:
        html_ok, html_summary = _validate_report_html(resolved)
        if not html_ok:
            return ArtifactVerificationResult(
                kind=kind,
                relative_path=relative_path,
                required=required,
                status=ArtifactVerificationStatus.INVALID,
                resolved_path=resolved,
                diagnostic_code="invalid_required_artifact",
                diagnostic_summary=html_summary,
            )
    elif kind == ExpectedArtifactKind.EVIDENCE_JSON:
        pass
    elif kind == ExpectedArtifactKind.OTHER:
        pass

    return ArtifactVerificationResult(
        kind=kind,
        relative_path=relative_path,
        required=required,
        status=ArtifactVerificationStatus.PASS,
        resolved_path=resolved,
        diagnostic_code="artifact_verified",
        diagnostic_summary="artifact_verified",
    )


def verify_platform_proof_artifacts(
    *,
    proof_artifact_directory: Path,
    spec: ProofExecutionSpec,
) -> ProofArtifactVerificationSummary:
    """Verify declared artifacts in descriptor declaration order."""
    results: list[ArtifactVerificationResult] = []
    for artifact in spec.expected_artifacts:
        results.append(
            _verify_single_artifact(
                proof_artifact_directory=proof_artifact_directory,
                kind=artifact.kind,
                relative_path=artifact.relative_path,
                required=artifact.required,
            )
        )

    passed = all(
        result.status
        in {
            ArtifactVerificationStatus.PASS,
            ArtifactVerificationStatus.OPTIONAL_MISSING,
        }
        for result in results
    )
    return ProofArtifactVerificationSummary(passed=passed, results=tuple(results))


def _first_failure(
    summary: ProofArtifactVerificationSummary,
) -> ArtifactVerificationResult | None:
    for result in summary.results:
        if result.required and result.status not in {
            ArtifactVerificationStatus.PASS,
        }:
            return result
        if not result.required and result.status == ArtifactVerificationStatus.INVALID:
            return result
        if not result.required and result.status == ArtifactVerificationStatus.FAIL:
            return result
    return None


def apply_artifact_verification(
    subprocess_result: ProofRunResult,
    summary: ProofArtifactVerificationSummary,
) -> ProofRunResult:
    failure = _first_failure(summary)
    overall_status = (
        ProofArtifactVerificationStatus.PASS
        if summary.passed
        else ProofArtifactVerificationStatus.FAIL
    )

    artifact_updates = {
        "artifact_verification_status": overall_status,
        "artifact_diagnostic": (
            failure.diagnostic_summary if failure is not None else "artifacts_verified"
        ),
    }

    if summary.passed:
        return subprocess_result.model_copy(update=artifact_updates)

    assert failure is not None
    diagnostic = failure.diagnostic_summary
    if (
        subprocess_result.status == ProofStatus.FAIL
        and subprocess_result.diagnostic_summary
        and failure.diagnostic_code.startswith("missing")
    ):
        diagnostic = failure.diagnostic_summary
    elif (
        subprocess_result.status == ProofStatus.FAIL
        and subprocess_result.diagnostic_summary
        and failure.diagnostic_code == "invalid_required_artifact"
    ):
        diagnostic = failure.diagnostic_summary

    return subprocess_result.model_copy(
        update={
            **artifact_updates,
            "status": ProofStatus.FAIL,
            "diagnostic_summary": diagnostic,
        }
    )
