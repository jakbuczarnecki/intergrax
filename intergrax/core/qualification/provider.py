# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Provider-scoped qualification contracts (PROVIDER-QUAL-2)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from intergrax.core.qualification.evidence import QualificationEvidence
from intergrax.core.qualification.status import QualificationStatus
from intergrax.core.qualification.validity import (
    QualificationRunId,
    _require_aware_instant,
    _require_non_empty_text,
    validate_qualification_run_id,
)


class ProviderQualificationEvidenceKind(StrEnum):
    """Auditable provider qualification evidence categories."""

    SUITE_EXECUTION = "suite_execution"
    LIVE_BACKEND = "live_backend"
    REPRODUCIBILITY = "reproducibility"
    LIMITATION = "limitation"
    SOURCE_ANCHOR = "source_anchor"


def _validate_limitations(value: tuple[str, ...]) -> None:
    if not isinstance(value, tuple):
        raise TypeError("limitations must be a tuple")
    for index, item in enumerate(value):
        if type(item) is not str:
            raise TypeError(f"limitations[{index}] must be str, got {type(item).__name__}")
        _require_non_empty_text(item, field_name=f"limitations[{index}]")


def _require_bool(value: object, *, field_name: str) -> None:
    if type(value) is not bool:
        raise TypeError(f"{field_name} must be bool, got {type(value).__name__}")


def _validate_evidence(
    value: tuple[QualificationEvidence[ProviderQualificationEvidenceKind], ...],
) -> None:
    if not isinstance(value, tuple):
        raise TypeError("evidence must be a tuple")
    for index, item in enumerate(value):
        if not isinstance(item, QualificationEvidence):
            raise TypeError(f"evidence[{index}] must be QualificationEvidence")
        if not isinstance(item.kind, ProviderQualificationEvidenceKind):
            raise TypeError(
                f"evidence[{index}].kind must be ProviderQualificationEvidenceKind, "
                f"got {type(item.kind).__name__}"
            )


def _validate_optional_text(value: str | None, *, field_name: str) -> None:
    if value is not None:
        _require_non_empty_text(value, field_name=field_name)


@dataclass(frozen=True, slots=True)
class ProviderQualificationSubject:
    """Identity for one provider capability in one environment under one suite version."""

    provider_id: str
    provider_version: str
    capability_id: str
    domain: str
    intergrax_revision: str
    qualification_suite_id: str
    qualification_suite_version: str
    environment_id: str
    adapter_identity: str | None = None
    package_name: str | None = None
    package_version: str | None = None
    entry_point_group: str | None = None
    entry_point_name: str | None = None
    host_registration_path: str | None = None
    delivery_source: str | None = None
    integration_kind: str | None = None

    def __post_init__(self) -> None:
        _require_non_empty_text(self.provider_id, field_name="provider_id")
        _require_non_empty_text(self.provider_version, field_name="provider_version")
        _require_non_empty_text(self.capability_id, field_name="capability_id")
        _require_non_empty_text(self.domain, field_name="domain")
        _require_non_empty_text(self.intergrax_revision, field_name="intergrax_revision")
        _require_non_empty_text(self.qualification_suite_id, field_name="qualification_suite_id")
        _require_non_empty_text(
            self.qualification_suite_version,
            field_name="qualification_suite_version",
        )
        _require_non_empty_text(self.environment_id, field_name="environment_id")
        _validate_optional_text(self.adapter_identity, field_name="adapter_identity")
        _validate_optional_text(self.package_name, field_name="package_name")
        _validate_optional_text(self.package_version, field_name="package_version")
        _validate_optional_text(self.entry_point_group, field_name="entry_point_group")
        _validate_optional_text(self.entry_point_name, field_name="entry_point_name")
        _validate_optional_text(self.host_registration_path, field_name="host_registration_path")
        _validate_optional_text(self.delivery_source, field_name="delivery_source")
        _validate_optional_text(self.integration_kind, field_name="integration_kind")


@dataclass(frozen=True, slots=True)
class ProviderQualificationExecutor:
    """Executor-neutral metadata for a qualification run."""

    executor_kind: str
    executor_id: str
    executor_version: str | None = None

    def __post_init__(self) -> None:
        _require_non_empty_text(self.executor_kind, field_name="executor_kind")
        _require_non_empty_text(self.executor_id, field_name="executor_id")
        _validate_optional_text(self.executor_version, field_name="executor_version")


@dataclass(frozen=True, slots=True)
class ProviderQualificationResultSummary:
    """Structured pass/fail counts for a provider qualification run."""

    passed: int
    failed: int
    skipped: int
    label: str | None = None

    def __post_init__(self) -> None:
        for field_name, count in (
            ("passed", self.passed),
            ("failed", self.failed),
            ("skipped", self.skipped),
        ):
            if type(count) is not int:
                raise TypeError(f"{field_name} must be int, got {type(count).__name__}")
            if count < 0:
                raise ValueError(f"{field_name} must be >= 0")
        _validate_optional_text(self.label, field_name="label")


@dataclass(frozen=True, slots=True)
class ProviderQualificationEnvironmentMetadata:
    """Bounded environment facts for provider qualification admission semantics."""

    real_backend: bool
    mocks: bool
    sqlite_substitution: bool
    bounded_environment: str | None = None

    def __post_init__(self) -> None:
        _require_bool(self.real_backend, field_name="real_backend")
        _require_bool(self.mocks, field_name="mocks")
        _require_bool(self.sqlite_substitution, field_name="sqlite_substitution")
        _validate_optional_text(self.bounded_environment, field_name="bounded_environment")


@dataclass(frozen=True, slots=True)
class ProviderQualificationRun:
    """Immutable audit record for one executed provider qualification attempt."""

    qualification_run_id: QualificationRunId
    subject: ProviderQualificationSubject
    status: QualificationStatus
    executed_at: datetime
    executor: ProviderQualificationExecutor
    result_summary: ProviderQualificationResultSummary
    evidence: tuple[QualificationEvidence[ProviderQualificationEvidenceKind], ...]
    reproducibility: str | None
    limitations: tuple[str, ...]
    source_revision: str
    environment_metadata: ProviderQualificationEnvironmentMetadata

    def __post_init__(self) -> None:
        validate_qualification_run_id(self.qualification_run_id)
        if not isinstance(self.subject, ProviderQualificationSubject):
            raise TypeError("subject must be ProviderQualificationSubject")
        if not isinstance(self.status, QualificationStatus):
            raise TypeError("status must be QualificationStatus")
        _require_aware_instant(self.executed_at, field_name="executed_at")
        if not isinstance(self.executor, ProviderQualificationExecutor):
            raise TypeError("executor must be ProviderQualificationExecutor")
        if not isinstance(self.result_summary, ProviderQualificationResultSummary):
            raise TypeError("result_summary must be ProviderQualificationResultSummary")
        _validate_evidence(self.evidence)
        _validate_optional_text(self.reproducibility, field_name="reproducibility")
        _validate_limitations(self.limitations)
        _require_non_empty_text(self.source_revision, field_name="source_revision")
        if not isinstance(self.environment_metadata, ProviderQualificationEnvironmentMetadata):
            raise TypeError(
                "environment_metadata must be ProviderQualificationEnvironmentMetadata"
            )
