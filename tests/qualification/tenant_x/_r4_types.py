# © Artur Czarnecki. All rights reserved.

"""TENANT-X-R4 typed verdicts and matrix rows."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final


class R4ChainVerdict(StrEnum):
    PROVEN_CURRENT_HEAD = "PROVEN_CURRENT_HEAD"
    MISSING = "MISSING"
    INVALID = "INVALID"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class R4FamilyBinding(StrEnum):
    TENANT_BOUND = "TENANT_BOUND"
    NON_TENANT = "NON_TENANT"
    UNKNOWN = "UNKNOWN"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class R4CrossTenantClass(StrEnum):
    SAME_TENANT_PROVEN = "SAME_TENANT_PROVEN"
    CROSS_TENANT_REJECTED = "CROSS_TENANT_REJECTED"
    CROSS_TENANT_ALLOWED = "CROSS_TENANT_ALLOWED"
    UNKNOWN = "UNKNOWN"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class R4ObservabilityRole(StrEnum):
    CARRIER = "CARRIER"
    VALIDATOR = "VALIDATOR"
    DERIVED_OBSERVATION = "DERIVED_OBSERVATION"
    NON_TENANT = "NON_TENANT"
    ILLEGAL_AUTHORITY = "ILLEGAL_AUTHORITY"
    UNKNOWN = "UNKNOWN"


class R4ExtensionTenantClass(StrEnum):
    PRESERVED = "PRESERVED"
    REJECTED = "REJECTED"
    ALLOWED = "ALLOWED"
    UNKNOWN = "UNKNOWN"


class R4SubstitutionClass(StrEnum):
    ILLEGAL_SUBSTITUTION_PRESENT = "ILLEGAL_SUBSTITUTION_PRESENT"
    ILLEGAL_INPUT_REJECTED = "ILLEGAL_INPUT_REJECTED"
    LEGAL_NON_TENANT = "LEGAL_NON_TENANT"
    UNKNOWN = "UNKNOWN"


class R4ContextTenantSourceClass(StrEnum):
    CANONICAL_EXECUTION_TENANT = "CANONICAL_EXECUTION_TENANT"
    VALIDATED_METADATA_TENANT = "VALIDATED_METADATA_TENANT"
    AMBIENT_RUNTIME_TENANT = "AMBIENT_RUNTIME_TENANT"
    UNTRUSTED_CALLER_METADATA = "UNTRUSTED_CALLER_METADATA"
    UNKNOWN = "UNKNOWN"


class R4EventTenantSourceClass(StrEnum):
    DIRECT_FROM_CANONICAL_EXECUTION = "DIRECT_FROM_CANONICAL_EXECUTION"
    VALIDATED_FROM_BOUND_CONTEXT = "VALIDATED_FROM_BOUND_CONTEXT"
    RESTORED_FROM_TYPED_ASYNC_IDENTITY = "RESTORED_FROM_TYPED_ASYNC_IDENTITY"
    NON_TENANT_EVENT = "NON_TENANT_EVENT"
    UNKNOWN = "UNKNOWN"


class R4AttributionTenantOptionalClass(StrEnum):
    LEGAL_NON_TENANT = "LEGAL_NON_TENANT"
    TENANT_REQUIRED_BUT_OPTIONAL = "TENANT_REQUIRED_BUT_OPTIONAL"
    LEGACY_COMPATIBILITY = "LEGACY_COMPATIBILITY"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class R4FamilyRow:
    family_id: str
    family_label: str
    binding: R4FamilyBinding
    authoritative_tenant_source: str
    evidence: tuple[str, ...]
    notes: str = ""


@dataclass(frozen=True, slots=True)
class R4ChainTransition:
    chain_id: str
    transition_id: str
    source_symbol: str
    target_symbol: str
    tenant_source: str
    tenant_target: str
    relation: str
    mismatch_behavior: str
    fail_closed: bool
    verdict: R4ChainVerdict
    evidence: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class R4MatrixRow:
    row_id: str
    surface_hint: str
    verdict: (
        R4ChainVerdict
        | R4CrossTenantClass
        | R4ObservabilityRole
        | R4ExtensionTenantClass
        | R4SubstitutionClass
        | R4ContextTenantSourceClass
        | R4EventTenantSourceClass
        | R4AttributionTenantOptionalClass
    )
    evidence: tuple[str, ...]
    tenant_expression: str = ""


R4_EXPECTED_BASELINE_HEAD: Final[str] = "5d9f6887f21f5313bedbcc38eced5cf8b306cc92"
