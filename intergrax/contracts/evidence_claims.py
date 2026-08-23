# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Universal evidence-backed claim and challenge contracts (GAP-1A).

Platform owns typed structure and structural invariants. Applications and domains
own claim meaning, ``claim_kind`` vocabulary, and optional ``defect_code`` values.

These contracts represent the system's evidence-backed position — not metaphysical
truth. ``ClaimResolution.SUPPORTED`` means evidence satisfies application-defined
acceptance criteria, not universal correctness.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Final, Literal, NewType
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_EVIDENCE_BACKED_CLAIM_V1: Final = "evidence_backed_claim.v1"
SCHEMA_EVIDENCE_CHALLENGE_V1: Final = "evidence_challenge.v1"
SCHEMA_EVIDENCE_CLAIM_SET_V1: Final = "evidence_claim_set.v1"

_EVIDENCE_REF_MAX_LENGTH: Final = 256
_CLAIM_KIND_MAX_LENGTH: Final = 128
_DEFECT_CODE_MAX_LENGTH: Final = 128
_STATEMENT_MAX_LENGTH: Final = 4096
_DESCRIPTION_MAX_LENGTH: Final = 2048

_EVIDENCE_CLAIM_ID_PREFIX: Final = "eclaim_"
_EVIDENCE_CHALLENGE_ID_PREFIX: Final = "echlg_"
_CANONICAL_SUFFIX = re.compile(r"^[0-9a-f]{32}$")
_SAFE_BOUNDED_ID_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9._:-]*$")

_NON_EMPTY = Field(min_length=1)

EvidenceClaimId = NewType("EvidenceClaimId", str)
EvidenceChallengeId = NewType("EvidenceChallengeId", str)
EvidenceReferenceId = NewType("EvidenceReferenceId", str)
ClaimKind = NewType("ClaimKind", str)
DefectCode = NewType("DefectCode", str)


class ClaimResolution(StrEnum):
    """Business-neutral resolution of an evidence-backed claim."""

    PENDING = "pending"
    SUPPORTED = "supported"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class ChallengeResolution(StrEnum):
    """Business-neutral lifecycle state for a challenge."""

    OPEN = "open"
    SATISFIED = "satisfied"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


class ChallengeDefectFamily(StrEnum):
    """Closed platform taxonomy for structural challenge families."""

    MISSING_EVIDENCE = "missing_evidence"
    CONTRADICTORY_EVIDENCE = "contradictory_evidence"
    ADMISSIBILITY_FAILURE = "admissibility_failure"
    UNSUPPORTED_INFERENCE = "unsupported_inference"
    UNADDRESSED_ALTERNATIVE = "unaddressed_alternative"
    OTHER = "other"


def _validate_canonical_entity_id(
    value: object,
    *,
    prefix: str,
    label: str,
) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be str, got {type(value).__name__}")
    if not value or not value.strip():
        raise ValueError(f"{label} must be non-empty and not whitespace-only")
    if value != value.strip():
        raise ValueError(f"{label} must not contain leading or trailing whitespace")
    if not value.startswith(prefix):
        raise ValueError(f"{label} must start with {prefix!r}")
    suffix = value[len(prefix):]
    if not _CANONICAL_SUFFIX.fullmatch(suffix):
        raise ValueError(f"{label} suffix must match [0-9a-f]{{32}}")
    return value


def validate_evidence_claim_id(value: object) -> EvidenceClaimId:
    return EvidenceClaimId(
        _validate_canonical_entity_id(
            value,
            prefix=_EVIDENCE_CLAIM_ID_PREFIX,
            label="EvidenceClaimId",
        )
    )


def validate_evidence_challenge_id(value: object) -> EvidenceChallengeId:
    return EvidenceChallengeId(
        _validate_canonical_entity_id(
            value,
            prefix=_EVIDENCE_CHALLENGE_ID_PREFIX,
            label="EvidenceChallengeId",
        )
    )


def mint_evidence_claim_id() -> EvidenceClaimId:
    return EvidenceClaimId(f"{_EVIDENCE_CLAIM_ID_PREFIX}{uuid4().hex}")


def mint_evidence_challenge_id() -> EvidenceChallengeId:
    return EvidenceChallengeId(f"{_EVIDENCE_CHALLENGE_ID_PREFIX}{uuid4().hex}")


def validate_evidence_reference_id(value: object) -> EvidenceReferenceId:
    if type(value) is not str:
        raise TypeError(
            f"EvidenceReferenceId must be str, got {type(value).__name__}"
        )
    identifier = value.strip()
    if not identifier:
        raise ValueError("EvidenceReferenceId must be non-empty")
    if len(identifier) > _EVIDENCE_REF_MAX_LENGTH:
        raise ValueError(
            f"EvidenceReferenceId must be at most {_EVIDENCE_REF_MAX_LENGTH} characters"
        )
    if any(character.isspace() or ord(character) < 32 for character in identifier):
        raise ValueError(
            "EvidenceReferenceId must not contain whitespace or control characters"
        )
    if not _SAFE_BOUNDED_ID_RE.match(identifier):
        raise ValueError("EvidenceReferenceId has invalid characters")
    return EvidenceReferenceId(identifier)


def validate_claim_kind(value: object) -> ClaimKind:
    if type(value) is not str:
        raise TypeError(f"ClaimKind must be str, got {type(value).__name__}")
    identifier = value.strip()
    if not identifier:
        raise ValueError("ClaimKind must be non-empty")
    if len(identifier) > _CLAIM_KIND_MAX_LENGTH:
        raise ValueError(
            f"ClaimKind must be at most {_CLAIM_KIND_MAX_LENGTH} characters"
        )
    if any(character.isspace() or ord(character) < 32 for character in identifier):
        raise ValueError("ClaimKind must not contain whitespace or control characters")
    if not _SAFE_BOUNDED_ID_RE.match(identifier):
        raise ValueError("ClaimKind has invalid characters")
    return ClaimKind(identifier)


def validate_defect_code(value: object) -> DefectCode:
    if type(value) is not str:
        raise TypeError(f"DefectCode must be str, got {type(value).__name__}")
    identifier = value.strip()
    if not identifier:
        raise ValueError("DefectCode must be non-empty")
    if len(identifier) > _DEFECT_CODE_MAX_LENGTH:
        raise ValueError(
            f"DefectCode must be at most {_DEFECT_CODE_MAX_LENGTH} characters"
        )
    if any(character.isspace() or ord(character) < 32 for character in identifier):
        raise ValueError("DefectCode must not contain whitespace or control characters")
    if not _SAFE_BOUNDED_ID_RE.match(identifier):
        raise ValueError("DefectCode has invalid characters")
    return DefectCode(identifier)


def _normalize_evidence_reference_collection(
    value: object,
    *,
    field_name: str,
) -> tuple[EvidenceReferenceId, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        raise ValueError(f"{field_name} must be a sequence")
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a sequence")
    normalized: list[EvidenceReferenceId] = []
    seen: set[EvidenceReferenceId] = set()
    for item in value:
        evidence_id = validate_evidence_reference_id(item)
        if evidence_id in seen:
            raise ValueError(f"{field_name} must not contain duplicates")
        seen.add(evidence_id)
        normalized.append(evidence_id)
    return tuple(sorted(normalized, key=str))


class EvidenceBackedClaim(BaseModel):
    """Immutable evidence-backed claim — domain-neutral structural contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["evidence_backed_claim.v1"] = SCHEMA_EVIDENCE_BACKED_CLAIM_V1
    claim_id: EvidenceClaimId
    statement: str = _NON_EMPTY
    claim_kind: ClaimKind
    supporting_evidence_ids: tuple[EvidenceReferenceId, ...] = ()
    contradicting_evidence_ids: tuple[EvidenceReferenceId, ...] = ()
    resolution: ClaimResolution = ClaimResolution.PENDING
    supersedes_claim_id: EvidenceClaimId | None = None

    @field_validator("claim_id", mode="before")
    @classmethod
    def _validate_claim_id(cls, value: object) -> EvidenceClaimId:
        return validate_evidence_claim_id(value)

    @field_validator("supersedes_claim_id", mode="before")
    @classmethod
    def _validate_supersedes_claim_id(cls, value: object) -> EvidenceClaimId | None:
        if value is None:
            return None
        return validate_evidence_claim_id(value)

    @field_validator("claim_kind", mode="before")
    @classmethod
    def _validate_claim_kind(cls, value: object) -> ClaimKind:
        return validate_claim_kind(value)

    @field_validator("statement")
    @classmethod
    def _validate_statement(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("statement must be non-empty")
        if len(normalized) > _STATEMENT_MAX_LENGTH:
            raise ValueError(
                f"statement must be at most {_STATEMENT_MAX_LENGTH} characters"
            )
        return normalized

    @field_validator("supporting_evidence_ids", mode="before")
    @classmethod
    def _validate_supporting_evidence_ids(cls, value: object) -> tuple[EvidenceReferenceId, ...]:
        return _normalize_evidence_reference_collection(
            value,
            field_name="supporting_evidence_ids",
        )

    @field_validator("contradicting_evidence_ids", mode="before")
    @classmethod
    def _validate_contradicting_evidence_ids(
        cls,
        value: object,
    ) -> tuple[EvidenceReferenceId, ...]:
        return _normalize_evidence_reference_collection(
            value,
            field_name="contradicting_evidence_ids",
        )

    @model_validator(mode="after")
    def _evidence_collections_disjoint(self) -> EvidenceBackedClaim:
        overlap = set(self.supporting_evidence_ids) & set(self.contradicting_evidence_ids)
        if overlap:
            raise ValueError(
                "supporting_evidence_ids and contradicting_evidence_ids must be disjoint"
            )
        if self.supersedes_claim_id is not None and self.supersedes_claim_id == self.claim_id:
            raise ValueError("claim must not supersede itself")
        return self


class EvidenceChallenge(BaseModel):
    """Immutable challenge against a specific evidence-backed claim."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["evidence_challenge.v1"] = SCHEMA_EVIDENCE_CHALLENGE_V1
    challenge_id: EvidenceChallengeId
    claim_id: EvidenceClaimId
    defect_family: ChallengeDefectFamily
    defect_code: DefectCode | None = None
    evidence_ids: tuple[EvidenceReferenceId, ...] = ()
    description: str = ""
    resolution: ChallengeResolution = ChallengeResolution.OPEN

    @field_validator("challenge_id", mode="before")
    @classmethod
    def _validate_challenge_id(cls, value: object) -> EvidenceChallengeId:
        return validate_evidence_challenge_id(value)

    @field_validator("claim_id", mode="before")
    @classmethod
    def _validate_target_claim_id(cls, value: object) -> EvidenceClaimId:
        return validate_evidence_claim_id(value)

    @field_validator("defect_code", mode="before")
    @classmethod
    def _validate_optional_defect_code(cls, value: object) -> DefectCode | None:
        if value is None:
            return None
        return validate_defect_code(value)

    @field_validator("evidence_ids", mode="before")
    @classmethod
    def _validate_evidence_ids(cls, value: object) -> tuple[EvidenceReferenceId, ...]:
        return _normalize_evidence_reference_collection(
            value,
            field_name="evidence_ids",
        )

    @field_validator("description")
    @classmethod
    def _validate_description(cls, value: str) -> str:
        normalized = value.strip()
        if len(normalized) > _DESCRIPTION_MAX_LENGTH:
            raise ValueError(
                f"description must be at most {_DESCRIPTION_MAX_LENGTH} characters"
            )
        return normalized


class EvidenceClaimSet(BaseModel):
    """Aggregate for referential integrity across claims and challenges."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["evidence_claim_set.v1"] = SCHEMA_EVIDENCE_CLAIM_SET_V1
    claims: tuple[EvidenceBackedClaim, ...] = ()
    challenges: tuple[EvidenceChallenge, ...] = ()

    @model_validator(mode="after")
    def _validate_referential_integrity(self) -> EvidenceClaimSet:
        claim_ids: list[EvidenceClaimId] = []
        seen_claim_ids: set[EvidenceClaimId] = set()
        for claim in self.claims:
            if claim.claim_id in seen_claim_ids:
                raise ValueError("claims must have unique claim_id values")
            seen_claim_ids.add(claim.claim_id)
            claim_ids.append(claim.claim_id)

        claim_id_set = set(claim_ids)
        seen_challenge_ids: set[EvidenceChallengeId] = set()
        for challenge in self.challenges:
            if challenge.challenge_id in seen_challenge_ids:
                raise ValueError("challenges must have unique challenge_id values")
            seen_challenge_ids.add(challenge.challenge_id)
            if challenge.claim_id not in claim_id_set:
                raise ValueError("challenge claim_id must reference an existing claim")

        for claim in self.claims:
            if (
                claim.supersedes_claim_id is not None
                and claim.supersedes_claim_id not in claim_id_set
            ):
                raise ValueError(
                    "supersedes_claim_id must reference an existing claim in the set"
                )

        return self
