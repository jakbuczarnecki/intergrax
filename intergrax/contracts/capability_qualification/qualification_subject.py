# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical capability qualification subject (AW-7C-P2)."""

from __future__ import annotations

import hashlib
from enum import StrEnum
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from intergrax.contracts.capability_acquisition.acquisition_outcome import (
    CapabilityAcquisitionOutcome,
)
from intergrax.contracts.capability_acquisition.acquisition_result import (
    CapabilityAcquisitionResult,
)
from intergrax.contracts.capability_catalog._validation import require_non_empty_text
from intergrax.integrations.contracts.scoped_integration_adaptation import (
    ScopedIntegrationAdaptationArtifact,
    derive_scoped_integration_adaptation_scope_fingerprint,
)

SCHEMA_CAPABILITY_QUALIFICATION_SUBJECT_V1: Final = "capability_qualification_subject.v1"
_NON_EMPTY = Field(min_length=1)


class CapabilityQualificationSubjectKind(StrEnum):
    ACQUIRED_CAPABILITY = "ACQUIRED_CAPABILITY"
    SCOPED_INTEGRATION_ADAPTATION = "SCOPED_INTEGRATION_ADAPTATION"


class CapabilityQualificationSubjectSourceKind(StrEnum):
    ACQUISITION_HANDOFF = "ACQUISITION_HANDOFF"
    INTEGRATION_ADAPTATION_ARTIFACT = "INTEGRATION_ADAPTATION_ARTIFACT"


class CapabilityQualificationSubjectSource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_kind: CapabilityQualificationSubjectSourceKind
    source_reference: str = _NON_EMPTY


class CapabilityQualificationAcquisitionLineage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    acquisition_request_id: str = _NON_EMPTY
    gap_id: str = _NON_EMPTY
    strategy_id: str = _NON_EMPTY
    artifact_reference: str | None = None
    domain_handoff_reference: str | None = None
    correlation_id: str | None = None
    causation_id: str | None = None

    @model_validator(mode="after")
    def _subject_reference_present(self) -> CapabilityQualificationAcquisitionLineage:
        if not self.artifact_reference and not self.domain_handoff_reference:
            raise ValueError(
                "acquisition lineage requires artifact_reference or domain_handoff_reference",
            )
        return self


class CapabilityQualificationAdaptationLineage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_id: str = _NON_EMPTY
    artifact_fingerprint: str = _NON_EMPTY
    integration_category: str = _NON_EMPTY
    provider_id: str = _NON_EMPTY
    resource_scope: str = _NON_EMPTY
    strategy_id: str = _NON_EMPTY
    candidate_id: str = _NON_EMPTY
    candidate_revision: str = _NON_EMPTY
    correlation_id: str | None = None
    causation_id: str | None = None


class CapabilityQualificationSubject(BaseModel):
    """Immutable qualification semantic subject — identity only, not authority."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["capability_qualification_subject.v1"] = (
        SCHEMA_CAPABILITY_QUALIFICATION_SUBJECT_V1
    )
    subject_kind: CapabilityQualificationSubjectKind
    subject_id: str = _NON_EMPTY
    subject_integrity_fingerprint: str = _NON_EMPTY
    tenant_id: str | None = None
    scope_fingerprint: str | None = None
    source: CapabilityQualificationSubjectSource
    acquisition_lineage: CapabilityQualificationAcquisitionLineage | None = None
    adaptation_lineage: CapabilityQualificationAdaptationLineage | None = None

    @model_validator(mode="after")
    def _kind_lineage_consistent(self) -> CapabilityQualificationSubject:
        if self.subject_kind is CapabilityQualificationSubjectKind.ACQUIRED_CAPABILITY:
            if self.tenant_id is not None:
                raise ValueError("ACQUIRED_CAPABILITY subject must not carry tenant_id")
            if self.scope_fingerprint is not None:
                raise ValueError(
                    "ACQUIRED_CAPABILITY subject must not carry scope_fingerprint",
                )
            if self.acquisition_lineage is None:
                raise ValueError("ACQUIRED_CAPABILITY requires acquisition_lineage")
            if self.adaptation_lineage is not None:
                raise ValueError(
                    "ACQUIRED_CAPABILITY must not carry adaptation_lineage",
                )
            if (
                self.source.source_kind
                is not CapabilityQualificationSubjectSourceKind.ACQUISITION_HANDOFF
            ):
                raise ValueError("ACQUIRED_CAPABILITY requires ACQUISITION_HANDOFF source")
            return self
        if (
            self.subject_kind
            is CapabilityQualificationSubjectKind.SCOPED_INTEGRATION_ADAPTATION
        ):
            if not self.tenant_id:
                raise ValueError(
                    "SCOPED_INTEGRATION_ADAPTATION requires tenant_id",
                )
            if not self.scope_fingerprint:
                raise ValueError(
                    "SCOPED_INTEGRATION_ADAPTATION requires scope_fingerprint",
                )
            if self.adaptation_lineage is None:
                raise ValueError(
                    "SCOPED_INTEGRATION_ADAPTATION requires adaptation_lineage",
                )
            if self.acquisition_lineage is not None:
                raise ValueError(
                    "SCOPED_INTEGRATION_ADAPTATION must not carry acquisition_lineage",
                )
            if (
                self.source.source_kind
                is not CapabilityQualificationSubjectSourceKind.INTEGRATION_ADAPTATION_ARTIFACT
            ):
                raise ValueError(
                    "SCOPED_INTEGRATION_ADAPTATION requires INTEGRATION_ADAPTATION_ARTIFACT source",
                )
            return self
        raise ValueError(f"unsupported subject_kind: {self.subject_kind}")


def derive_acquisition_subject_integrity_fingerprint(
    *,
    acquisition_request_id: str,
    gap_id: str,
    strategy_id: str,
    artifact_reference: str | None,
    domain_handoff_reference: str | None,
    correlation_id: str | None,
    causation_id: str | None,
) -> str:
    parts = [
        "kind=ACQUIRED_CAPABILITY",
        f"request_id={require_non_empty_text(acquisition_request_id, label='request_id')}",
        f"gap_id={require_non_empty_text(gap_id, label='gap_id')}",
        f"strategy_id={require_non_empty_text(strategy_id, label='strategy_id')}",
        "outcome=SUCCEEDED",
        f"artifact_reference={artifact_reference or ''}",
        f"domain_handoff_reference={domain_handoff_reference or ''}",
        f"correlation_id={correlation_id or ''}",
        f"causation_id={causation_id or ''}",
    ]
    canonical = "\n".join(parts)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def project_acquisition_qualification_subject(
    *,
    acquisition_result: CapabilityAcquisitionResult,
) -> CapabilityQualificationSubject:
    if acquisition_result.outcome is not CapabilityAcquisitionOutcome.SUCCEEDED:
        raise ValueError("projection requires acquisition outcome SUCCEEDED")
    strategy_id = acquisition_result.strategy_id
    if strategy_id is None:
        raise ValueError("SUCCEEDED acquisition requires strategy_id")
    evidence = acquisition_result.evidence
    if evidence is None:
        raise ValueError("SUCCEEDED acquisition requires evidence")
    artifact_reference = evidence.artifact_reference
    domain_handoff_reference = evidence.domain_handoff_reference
    fingerprint = derive_acquisition_subject_integrity_fingerprint(
        acquisition_request_id=acquisition_result.request_id,
        gap_id=acquisition_result.gap_id,
        strategy_id=strategy_id,
        artifact_reference=artifact_reference,
        domain_handoff_reference=domain_handoff_reference,
        correlation_id=acquisition_result.correlation_id,
        causation_id=acquisition_result.causation_id,
    )
    source_ref = artifact_reference or domain_handoff_reference
    if source_ref is None:
        raise ValueError("acquisition evidence must name a subject reference")
    return CapabilityQualificationSubject(
        subject_kind=CapabilityQualificationSubjectKind.ACQUIRED_CAPABILITY,
        subject_id=acquisition_result.request_id,
        subject_integrity_fingerprint=fingerprint,
        source=CapabilityQualificationSubjectSource(
            source_kind=CapabilityQualificationSubjectSourceKind.ACQUISITION_HANDOFF,
            source_reference=source_ref,
        ),
        acquisition_lineage=CapabilityQualificationAcquisitionLineage(
            acquisition_request_id=acquisition_result.request_id,
            gap_id=acquisition_result.gap_id,
            strategy_id=strategy_id,
            artifact_reference=artifact_reference,
            domain_handoff_reference=domain_handoff_reference,
            correlation_id=acquisition_result.correlation_id,
            causation_id=acquisition_result.causation_id,
        ),
    )


def project_adaptation_qualification_subject(
    artifact: ScopedIntegrationAdaptationArtifact,
) -> CapabilityQualificationSubject:
    scope_fp = derive_scoped_integration_adaptation_scope_fingerprint(artifact.scope)
    if artifact.scope_fingerprint != scope_fp:
        raise ValueError("artifact scope_fingerprint mismatch")
    if artifact.artifact_fingerprint != artifact.derived_artifact_fingerprint():
        raise ValueError("artifact fingerprint mismatch")
    return CapabilityQualificationSubject(
        subject_kind=CapabilityQualificationSubjectKind.SCOPED_INTEGRATION_ADAPTATION,
        subject_id=artifact.artifact_id,
        subject_integrity_fingerprint=artifact.artifact_fingerprint,
        tenant_id=artifact.tenant_id,
        scope_fingerprint=artifact.scope_fingerprint,
        source=CapabilityQualificationSubjectSource(
            source_kind=CapabilityQualificationSubjectSourceKind.INTEGRATION_ADAPTATION_ARTIFACT,
            source_reference=artifact.artifact_id,
        ),
        adaptation_lineage=CapabilityQualificationAdaptationLineage(
            artifact_id=artifact.artifact_id,
            artifact_fingerprint=artifact.artifact_fingerprint,
            integration_category=artifact.integration_category.value,
            provider_id=artifact.provider_id,
            resource_scope=artifact.resource_scope,
            strategy_id=artifact.strategy_id,
            candidate_id=artifact.candidate_id,
            candidate_revision=artifact.candidate_revision,
            correlation_id=artifact.correlation_id,
            causation_id=artifact.causation_id,
        ),
    )


def subjects_semantically_equal(
    left: CapabilityQualificationSubject,
    right: CapabilityQualificationSubject,
) -> bool:
    return (
        left.subject_kind == right.subject_kind
        and left.subject_id == right.subject_id
        and left.subject_integrity_fingerprint == right.subject_integrity_fingerprint
        and left.tenant_id == right.tenant_id
        and left.scope_fingerprint == right.scope_fingerprint
    )


__all__ = [
    "SCHEMA_CAPABILITY_QUALIFICATION_SUBJECT_V1",
    "CapabilityQualificationAcquisitionLineage",
    "CapabilityQualificationAdaptationLineage",
    "CapabilityQualificationSubject",
    "CapabilityQualificationSubjectKind",
    "CapabilityQualificationSubjectSource",
    "CapabilityQualificationSubjectSourceKind",
    "derive_acquisition_subject_integrity_fingerprint",
    "project_acquisition_qualification_subject",
    "project_adaptation_qualification_subject",
    "subjects_semantically_equal",
]
