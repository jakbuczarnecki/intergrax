# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Platform reconciliation evidence — durable truth after probe execution (ERL)."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from intergrax.contracts.enterprise_reliability.evidence import ExternalEffectEvidenceVerdict

SCHEMA_EXTERNAL_EFFECT_EVIDENCE_V1: Final = "external_effect_evidence.v1"
SCHEMA_EXTERNAL_EFFECT_EVIDENCE_OPERATION_LINK_V1: Final = (
    "external_effect_evidence_operation_link.v1"
)


class ExternalEffectEvidenceSourceKind(StrEnum):
    """Where platform-trusted information originated — no provider product names."""

    RECONCILIATION_PROBE = "reconciliation_probe"


class ExternalEffectEvidenceType(StrEnum):
    """Kind of check performed — orthogonal to business domain."""

    RECONCILIATION_PROBE_READ = "reconciliation_probe_read"


class ExternalEffectEvidenceConfidence(StrEnum):
    """Whether the check supports a definitive operational decision."""

    DEFINITIVE = "definitive"
    INCONCLUSIVE = "inconclusive"


class ExternalEffectEvidenceCheckResult(StrEnum):
    """Normalized outcome of the verification read."""

    CONFIRMED_SUCCESS = "confirmed_success"
    CONFIRMED_FAILURE = "confirmed_failure"
    INCONCLUSIVE = "inconclusive"


class ExternalEffectEvidenceOperationLink(BaseModel):
    """Bind evidence to one external-effect operation episode."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_EXTERNAL_EFFECT_EVIDENCE_OPERATION_LINK_V1
    tenant_id: str = Field(min_length=1, max_length=256)
    correlation_id: str = Field(min_length=1, max_length=256)
    contract_id: str = Field(min_length=1, max_length=256)
    probe_ref: str = Field(min_length=1, max_length=256)
    plugin_id: str = Field(min_length=1, max_length=256)
    attempt_index: int = Field(default=1, ge=1, le=256)


class ExternalEffectEvidence(BaseModel):
    """
    Platform record of what was learned from reconciliation.

    Integrations store provider payloads behind ``evidence_ref``; core never
    embeds domain-specific evidence types.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_EXTERNAL_EFFECT_EVIDENCE_V1
    source_kind: ExternalEffectEvidenceSourceKind
    evidence_type: ExternalEffectEvidenceType
    confidence: ExternalEffectEvidenceConfidence
    check_result: ExternalEffectEvidenceCheckResult
    verdict: ExternalEffectEvidenceVerdict
    evidence_ref: str = Field(min_length=1, max_length=512)
    operation_link: ExternalEffectEvidenceOperationLink
    obtained_at: datetime
    rationale: str = Field(default="", max_length=512)


class ExternalEffectEvidenceError(ValueError):
    """Evidence cannot be applied to the current uncertainty episode."""


def check_result_from_verdict(
    verdict: ExternalEffectEvidenceVerdict,
) -> ExternalEffectEvidenceCheckResult:
    if verdict is ExternalEffectEvidenceVerdict.DEFINITIVE_SUCCESS:
        return ExternalEffectEvidenceCheckResult.CONFIRMED_SUCCESS
    if verdict is ExternalEffectEvidenceVerdict.DEFINITIVE_FAILURE:
        return ExternalEffectEvidenceCheckResult.CONFIRMED_FAILURE
    return ExternalEffectEvidenceCheckResult.INCONCLUSIVE


def confidence_from_verdict(
    verdict: ExternalEffectEvidenceVerdict,
) -> ExternalEffectEvidenceConfidence:
    if verdict is ExternalEffectEvidenceVerdict.INSUFFICIENT:
        return ExternalEffectEvidenceConfidence.INCONCLUSIVE
    return ExternalEffectEvidenceConfidence.DEFINITIVE


__all__ = [
    "ExternalEffectEvidence",
    "ExternalEffectEvidenceCheckResult",
    "ExternalEffectEvidenceConfidence",
    "ExternalEffectEvidenceError",
    "ExternalEffectEvidenceOperationLink",
    "ExternalEffectEvidenceSourceKind",
    "ExternalEffectEvidenceType",
    "SCHEMA_EXTERNAL_EFFECT_EVIDENCE_OPERATION_LINK_V1",
    "SCHEMA_EXTERNAL_EFFECT_EVIDENCE_V1",
    "check_result_from_verdict",
    "confidence_from_verdict",
]
