# © Artur Czarnecki. All rights reserved.

"""TENANT-X-R3 typed verdicts and matrix rows."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final


class R3ChainVerdict(StrEnum):
    PROVEN_CURRENT_HEAD = "PROVEN_CURRENT_HEAD"
    MISSING = "MISSING"
    INVALID = "INVALID"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class R3FamilyBinding(StrEnum):
    TENANT_BOUND = "TENANT_BOUND"
    NON_TENANT = "NON_TENANT"
    UNKNOWN = "UNKNOWN"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class R3RecoveryPathClass(StrEnum):
    DURABLE_TENANT_AUTHORITATIVE = "DURABLE_TENANT_AUTHORITATIVE"
    CALLER_TENANT_VALIDATED_AGAINST_DURABLE = "CALLER_TENANT_VALIDATED_AGAINST_DURABLE"
    CALLER_TENANT_AUTHORITATIVE = "CALLER_TENANT_AUTHORITATIVE"
    NO_TENANT_CHECK = "NO_TENANT_CHECK"
    UNKNOWN = "UNKNOWN"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class R3CrossTenantClass(StrEnum):
    SAME_TENANT_PROVEN = "SAME_TENANT_PROVEN"
    CROSS_TENANT_REJECTED = "CROSS_TENANT_REJECTED"
    CROSS_TENANT_ALLOWED = "CROSS_TENANT_ALLOWED"
    UNKNOWN = "UNKNOWN"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class R3RecoveryAuthorityRole(StrEnum):
    TENANT_CARRIER = "TENANT_CARRIER"
    TENANT_VALIDATOR = "TENANT_VALIDATOR"
    TENANT_RESTORER_FROM_DURABLE_TRUTH = "TENANT_RESTORER_FROM_DURABLE_TRUTH"
    TENANT_AUTHORITY = "TENANT_AUTHORITY"
    UNKNOWN = "UNKNOWN"


class R3SubstitutionClass(StrEnum):
    ILLEGAL_SUBSTITUTION_PRESENT = "ILLEGAL_SUBSTITUTION_PRESENT"
    ILLEGAL_INPUT_REJECTED = "ILLEGAL_INPUT_REJECTED"
    LEGAL_NON_TENANT = "LEGAL_NON_TENANT"
    UNKNOWN = "UNKNOWN"


class R3HistoricalEvidenceClass(StrEnum):
    CURRENT_HEAD_STRUCTURALLY_VALIDATED = "CURRENT_HEAD_STRUCTURALLY_VALIDATED"
    HISTORICAL_SUPPORT_ONLY = "HISTORICAL_SUPPORT_ONLY"
    STALE = "STALE"


@dataclass(frozen=True, slots=True)
class R3RecoveryFamilyRow:
    family_id: str
    family_label: str
    binding: R3FamilyBinding
    authoritative_tenant_source: str
    evidence: tuple[str, ...]
    notes: str = ""


@dataclass(frozen=True, slots=True)
class R3ChainTransition:
    transition_id: str
    source_symbol: str
    target_symbol: str
    tenant_source: str
    tenant_target: str
    relation: str
    mismatch_behavior: str
    fail_closed: bool
    verdict: R3ChainVerdict
    evidence: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class R3MatrixRow:
    row_id: str
    surface_hint: str
    verdict: (
        R3ChainVerdict
        | R3RecoveryPathClass
        | R3CrossTenantClass
        | R3RecoveryAuthorityRole
        | R3SubstitutionClass
    )
    evidence: tuple[str, ...]
    tenant_expression: str = ""
    durable_key_scope: str = ""


R3_EXPECTED_BASELINE_HEAD: Final[str] = "9bc53dfad65dc20d327d7c7340953264c5d69c7a"
