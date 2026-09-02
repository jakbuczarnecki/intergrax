# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Typed domain qualification suite contracts (PROVIDER-QUAL-7)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from intergrax.core.qualification.evidence import QualificationEvidence
from intergrax.core.qualification.provider import (
    ProviderQualificationEnvironmentMetadata,
    ProviderQualificationEvidenceKind,
    ProviderQualificationResultSummary,
)
from intergrax.core.qualification.status import QualificationStatus
from intergrax.core.qualification.validity import _require_non_empty_text


@dataclass(frozen=True, slots=True)
class ProviderQualificationSuiteIdentity:
    """Canonical identity for one domain-owned qualification suite implementation."""

    domain: str
    capability_id: str
    qualification_suite_id: str
    qualification_suite_version: str

    def __post_init__(self) -> None:
        _require_non_empty_text(self.domain, field_name="domain")
        _require_non_empty_text(self.capability_id, field_name="capability_id")
        _require_non_empty_text(
            self.qualification_suite_id,
            field_name="qualification_suite_id",
        )
        _require_non_empty_text(
            self.qualification_suite_version,
            field_name="qualification_suite_version",
        )


@dataclass(frozen=True, slots=True)
class ProviderQualificationSuiteOutcome:
    """Structured suite result before platform run construction."""

    status: QualificationStatus
    result_summary: ProviderQualificationResultSummary
    evidence: tuple[QualificationEvidence[ProviderQualificationEvidenceKind], ...]
    environment_metadata: ProviderQualificationEnvironmentMetadata
    limitations: tuple[str, ...]
    reproducibility: str | None = None


@runtime_checkable
class ProviderQualificationSuite(Protocol):
    """Domain-owned semantic qualification suite against a typed materialized capability."""

    @property
    def identity(self) -> ProviderQualificationSuiteIdentity:
        """Return the suite identity implemented by this binding."""

    def execute(self, capability: object) -> ProviderQualificationSuiteOutcome:
        """Execute semantic qualification checks against ``capability``."""


@runtime_checkable
class ProviderQualificationMaterializationHandle(Protocol):
    """Lifecycle owner for materialized provider resources used during qualification."""

    def close(self) -> None:
        """Release provider-owned resources."""


@runtime_checkable
class ProviderQualificationDomainBinding(Protocol):
    """
    Domain extension point: materialize a provider capability and run a typed suite.

  Provider-neutral qualification core depends on this protocol instead of vendor dispatch.
    """

    @property
    def suite(self) -> ProviderQualificationSuite:
        """Return the domain qualification suite executed by this binding."""

    def materialize(
        self,
        profile: object,
        *,
        resolved_provider_id: str,
    ) -> tuple[object, ProviderQualificationMaterializationHandle]:
        """Materialize the typed domain capability for ``resolved_provider_id``."""

    def validate_resolved_provider(
        self,
        subject: object,
        *,
        resolved_provider_id: str,
    ) -> None:
        """Fail closed when resolved provider facts do not match the qualification subject."""
