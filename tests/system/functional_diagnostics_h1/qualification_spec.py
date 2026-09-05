# © Artur Czarnecki. All rights reserved.

"""Explicit qualification policy descriptors for DIAG-FUNCTIONAL-H1 family."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tests.system.functional_diagnostics_h1.models import (
    H1_K_QUALIFICATION_ID,
    H1_K_R2_QUALIFICATION_ID,
    H1_K_R3_QUALIFICATION_ID,
    H1_K_R4_QUALIFICATION_ID,
    H1_QUALIFICATION_ID,
    H1_R1_QUALIFICATION_ID,
    H1_R2_QUALIFICATION_ID,
    H1_R3_QUALIFICATION_ID,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True, slots=True)
class DiagnosticHealthQualificationSpec:
    qualification_id: str
    artifact_directory: Path
    closure_doc_path: Path
    requires_clean_repository: bool
    requires_origin_development_match: bool
    requires_stable_head: bool
    historical: bool
    requires_closure_doc_at_run: bool

    @property
    def artifact_report_json(self) -> Path:
        return self.artifact_directory / "qualification-report.json"

    @property
    def artifact_inventory_json(self) -> Path:
        return self.artifact_directory / "test-inventory.json"

    @property
    def artifact_human_report_md(self) -> Path:
        return self.artifact_directory / "qualification-report.md"

    def requires_repository_preconditions(self) -> bool:
        return (
            self.requires_clean_repository
            or self.requires_origin_development_match
            or self.requires_stable_head
        )


def _spec(
    qualification_id: str,
    artifact_suffix: str,
    closure_doc_name: str,
    *,
    historical: bool,
    requires_preconditions: bool,
    requires_closure_doc_at_run: bool,
) -> DiagnosticHealthQualificationSpec:
    return DiagnosticHealthQualificationSpec(
        qualification_id=qualification_id,
        artifact_directory=Path(f".tmp/session/diag-functional-h1{artifact_suffix}"),
        closure_doc_path=Path(
            f"docs/project/maintainers/qualification/{closure_doc_name}"
        ),
        requires_clean_repository=requires_preconditions,
        requires_origin_development_match=requires_preconditions,
        requires_stable_head=requires_preconditions,
        historical=historical,
        requires_closure_doc_at_run=requires_closure_doc_at_run,
    )


QUALIFICATION_SPECS: tuple[DiagnosticHealthQualificationSpec, ...] = (
    _spec(
        H1_QUALIFICATION_ID,
        "",
        "DIAG_FUNCTIONAL_H1_TEST_SUITE_HEALTH_QUALIFICATION.md",
        historical=True,
        requires_preconditions=False,
        requires_closure_doc_at_run=True,
    ),
    _spec(
        H1_R1_QUALIFICATION_ID,
        "-r1",
        "DIAG_FUNCTIONAL_H1_R1_TEST_SUITE_HEALTH_QUALIFICATION.md",
        historical=True,
        requires_preconditions=True,
        requires_closure_doc_at_run=True,
    ),
    _spec(
        H1_R2_QUALIFICATION_ID,
        "-r2",
        "DIAG_FUNCTIONAL_H1_R2_TEST_SUITE_HEALTH_QUALIFICATION.md",
        historical=True,
        requires_preconditions=True,
        requires_closure_doc_at_run=True,
    ),
    _spec(
        H1_R3_QUALIFICATION_ID,
        "-r3",
        "DIAG_FUNCTIONAL_H1_R3_TEST_SUITE_HEALTH_QUALIFICATION.md",
        historical=False,
        requires_preconditions=True,
        requires_closure_doc_at_run=False,
    ),
)

@dataclass(frozen=True, slots=True)
class LocalIntegrationQualificationSpec:
    qualification_id: str
    artifact_directory: Path
    canonical_run_count: int
    requires_clean_repository: bool
    requires_origin_development_match: bool

    @property
    def artifact_report_json(self) -> Path:
        return self.artifact_directory / "qualification-report.json"

    @property
    def artifact_human_report_md(self) -> Path:
        return self.artifact_directory / "qualification-report.md"


H1_K_R1_QUALIFICATION_SPEC = LocalIntegrationQualificationSpec(
    qualification_id=H1_K_QUALIFICATION_ID,
    artifact_directory=Path(".tmp/session/diag-h1-k-qualification-r1"),
    canonical_run_count=3,
    requires_clean_repository=True,
    requires_origin_development_match=True,
)

H1_K_R2_QUALIFICATION_SPEC = LocalIntegrationQualificationSpec(
    qualification_id=H1_K_R2_QUALIFICATION_ID,
    artifact_directory=Path(".tmp/session/diag-h1-k-qualification-r2"),
    canonical_run_count=3,
    requires_clean_repository=True,
    requires_origin_development_match=True,
)

H1_K_R3_QUALIFICATION_SPEC = LocalIntegrationQualificationSpec(
    qualification_id=H1_K_R3_QUALIFICATION_ID,
    artifact_directory=Path(".tmp/session/diag-h1-k-qualification-r3"),
    canonical_run_count=3,
    requires_clean_repository=True,
    requires_origin_development_match=True,
)

H1_K_R4_QUALIFICATION_SPEC = LocalIntegrationQualificationSpec(
    qualification_id=H1_K_R4_QUALIFICATION_ID,
    artifact_directory=Path(".tmp/session/diag-h1-k-qualification-r4"),
    canonical_run_count=3,
    requires_clean_repository=True,
    requires_origin_development_match=True,
)

H1_K_QUALIFICATION_SPEC = H1_K_R1_QUALIFICATION_SPEC

LOCAL_INTEGRATION_QUALIFICATION_SPECS: tuple[LocalIntegrationQualificationSpec, ...] = (
    H1_K_R1_QUALIFICATION_SPEC,
    H1_K_R2_QUALIFICATION_SPEC,
    H1_K_R3_QUALIFICATION_SPEC,
    H1_K_R4_QUALIFICATION_SPEC,
)

_LOCAL_INTEGRATION_SPEC_BY_ID: dict[str, LocalIntegrationQualificationSpec] = {
    spec.qualification_id: spec for spec in LOCAL_INTEGRATION_QUALIFICATION_SPECS
}

_QUALIFICATION_SPEC_BY_ID: dict[str, DiagnosticHealthQualificationSpec] = {
    spec.qualification_id: spec for spec in QUALIFICATION_SPECS
}


def resolve_qualification_spec(qualification_id: str) -> DiagnosticHealthQualificationSpec:
    spec = _QUALIFICATION_SPEC_BY_ID.get(qualification_id)
    if spec is None:
        msg = f"unknown qualification_id: {qualification_id}"
        raise ValueError(msg)
    return spec


def resolve_local_integration_qualification_spec(
    qualification_id: str,
) -> LocalIntegrationQualificationSpec:
    spec = _LOCAL_INTEGRATION_SPEC_BY_ID.get(qualification_id)
    if spec is None:
        msg = f"unknown qualification_id: {qualification_id}"
        raise ValueError(msg)
    return spec


def validate_local_integration_qualification_registry() -> tuple[str, ...]:
    violations: list[str] = []
    qualification_ids = [spec.qualification_id for spec in LOCAL_INTEGRATION_QUALIFICATION_SPECS]
    if len(qualification_ids) != len(set(qualification_ids)):
        violations.append("duplicate_qualification_ids")
    artifact_directories = [
        spec.artifact_directory.as_posix()
        for spec in LOCAL_INTEGRATION_QUALIFICATION_SPECS
    ]
    if len(artifact_directories) != len(set(artifact_directories)):
        violations.append("duplicate_artifact_directories")
    return tuple(violations)


def compose_qualification_spec(
    qualification_id: str,
    artifact_suffix: str,
    closure_doc_name: str,
    *,
    historical: bool,
    requires_preconditions: bool,
    requires_closure_doc_at_run: bool,
) -> DiagnosticHealthQualificationSpec:
    """Explicit composition helper for future qualifications (e.g. H1-R99 proof)."""
    return _spec(
        qualification_id,
        artifact_suffix,
        closure_doc_name,
        historical=historical,
        requires_preconditions=requires_preconditions,
        requires_closure_doc_at_run=requires_closure_doc_at_run,
    )


def health_qualification_runner_path() -> Path:
    return _REPO_ROOT / "tests/system/functional_diagnostics_h1/runner.py"
