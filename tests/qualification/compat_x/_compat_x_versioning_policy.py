# © Artur Czarnecki. All rights reserved.

"""COMPAT-X-R1 cross-platform evolution rules (qualification policy — not runtime authority)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

COMPAT_X_VERSIONING_POLICY_OWNER: Final[str] = (
    "tests/qualification/compat_x/_compat_x_versioning_policy.py (COMPAT-X evolution rules)"
)

PLATFORM_POLICY_CANON: Final[tuple[str, ...]] = (
    "For a compatibility-relevant surface, the owning domain controls version identity.",
    "COMPAT-X defines the cross-platform evolution rules.",
    "COMPAT-X does not own individual version values.",
    "breaking semantic/structural change => new contract/schema version",
    "unknown compatibility impact => fail closed / require explicit classification",
    "old-version acceptance => only through explicit reader/migration/compatibility policy",
    "additive field compatibility is family-policy/evidence based — not globally assumed",
)


class VersionObligation(StrEnum):
    EXPLICIT_VERSION_REQUIRED = "EXPLICIT_VERSION_REQUIRED"
    VERSION_INHERITED_FROM_CANONICAL_ENVELOPE = "VERSION_INHERITED_FROM_CANONICAL_ENVELOPE"
    INTERNAL_NON_VERSIONED_ALLOWED = "INTERNAL_NON_VERSIONED_ALLOWED"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    UNCLASSIFIED = "UNCLASSIFIED"


class VersionIdentityScheme(StrEnum):
    SCHEMA_ID_GENERATION = "SCHEMA_ID_GENERATION"
    INTEGER_GENERATION = "INTEGER_GENERATION"
    SEMANTIC_VERSION = "SEMANTIC_VERSION"
    EXTERNALLY_DEFINED_VERSION = "EXTERNALLY_DEFINED_VERSION"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    UNCLASSIFIED = "UNCLASSIFIED"


class CompatibilityChangeClass(StrEnum):
    REPRESENTATION_PRESERVING = "REPRESENTATION_PRESERVING"
    ADDITIVE_BACKWARD_COMPATIBLE = "ADDITIVE_BACKWARD_COMPATIBLE"
    BREAKING_STRUCTURAL = "BREAKING_STRUCTURAL"
    BREAKING_SEMANTIC = "BREAKING_SEMANTIC"
    REMOVAL_OR_RENAME = "REMOVAL_OR_RENAME"
    UNKNOWN = "UNKNOWN"


class VersionFieldRole(StrEnum):
    SCHEMA_CONTRACT_EVOLUTION = "SCHEMA_CONTRACT_EVOLUTION"
    OPTIMISTIC_CONCURRENCY = "OPTIMISTIC_CONCURRENCY"
    BUSINESS_ENTITY_REVISION = "BUSINESS_ENTITY_REVISION"
    NON_SCHEMA_DOMAIN_VERSION = "NON_SCHEMA_DOMAIN_VERSION"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class LaterStageDeferral(StrEnum):
    R2_PERSISTED_MIGRATION = "R2_PERSISTED_MIGRATION"
    R3_EVENT_EVOLUTION = "R3_EVENT_EVOLUTION"
    R4_PLUGIN_PROVIDER = "R4_PLUGIN_PROVIDER"
    R5_DEPRECATION = "R5_DEPRECATION"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class R1ComplianceState(StrEnum):
    COMPLIANT = "COMPLIANT"
    BLOCKED_R2_VALIDATION = "BLOCKED_R2_VALIDATION"
    BLOCKED_LATER_STAGE_ONLY = "BLOCKED_LATER_STAGE_ONLY"


@dataclass(frozen=True, slots=True)
class ChangeClassificationVerdict:
    change_class: CompatibilityChangeClass
    requires_new_version_identity: bool
    qualification_passes: bool
    reason: str


@dataclass(frozen=True, slots=True)
class VersionAuthorityModel:
    """Distinguishes policy owner from domain version truth."""

    cross_platform_policy_owner: str
    domain_version_owner_path: str
    current_version_source: str


_SCHEMA_VERSION_FIELD_NAMES: Final[frozenset[str]] = frozenset(
    {"schema_version", "contract_version", "payload_schema_version"}
)

_OPTIMISTIC_CONCURRENCY_FIELDS: Final[frozenset[str]] = frozenset(
    {"_version", "state_version", "etag", "revision_token"}
)

_BUSINESS_REVISION_FIELDS: Final[frozenset[str]] = frozenset(
    {"version", "artifact_version", "policy_version", "revision"}
)


def classify_version_field_role(field_name: str) -> VersionFieldRole:
    if field_name in _SCHEMA_VERSION_FIELD_NAMES:
        return VersionFieldRole.SCHEMA_CONTRACT_EVOLUTION
    if field_name in _OPTIMISTIC_CONCURRENCY_FIELDS:
        return VersionFieldRole.OPTIMISTIC_CONCURRENCY
    if field_name in _BUSINESS_REVISION_FIELDS:
        return VersionFieldRole.BUSINESS_ENTITY_REVISION
    if field_name.endswith("_version"):
        return VersionFieldRole.NON_SCHEMA_DOMAIN_VERSION
    return VersionFieldRole.NOT_APPLICABLE


def infer_version_identity_scheme(current_version: str) -> VersionIdentityScheme:
    stripped = current_version.strip()
    if not stripped or stripped == "unknown":
        return VersionIdentityScheme.UNCLASSIFIED
    if stripped.isdigit():
        return VersionIdentityScheme.INTEGER_GENERATION
    if stripped.count(".") == 2 and all(part.isdigit() for part in stripped.split(".")):
        return VersionIdentityScheme.SEMANTIC_VERSION
    if ".v" in stripped or stripped.endswith(".v1") or ".v" in stripped.lower():
        return VersionIdentityScheme.SCHEMA_ID_GENERATION
    if "." in stripped and not stripped[0].isdigit():
        return VersionIdentityScheme.SCHEMA_ID_GENERATION
    return VersionIdentityScheme.EXTERNALLY_DEFINED_VERSION


def evaluate_change_classification(change_class: CompatibilityChangeClass) -> ChangeClassificationVerdict:
    if change_class == CompatibilityChangeClass.UNKNOWN:
        return ChangeClassificationVerdict(
            change_class=change_class,
            requires_new_version_identity=True,
            qualification_passes=False,
            reason="unknown compatibility impact must fail closed",
        )
    if change_class == CompatibilityChangeClass.REPRESENTATION_PRESERVING:
        return ChangeClassificationVerdict(
            change_class=change_class,
            requires_new_version_identity=False,
            qualification_passes=True,
            reason="wire representation unchanged",
        )
    if change_class == CompatibilityChangeClass.ADDITIVE_BACKWARD_COMPATIBLE:
        return ChangeClassificationVerdict(
            change_class=change_class,
            requires_new_version_identity=False,
            qualification_passes=True,
            reason="additive only when family policy explicitly supports reader compatibility",
        )
    if change_class in {
        CompatibilityChangeClass.BREAKING_STRUCTURAL,
        CompatibilityChangeClass.BREAKING_SEMANTIC,
        CompatibilityChangeClass.REMOVAL_OR_RENAME,
    }:
        return ChangeClassificationVerdict(
            change_class=change_class,
            requires_new_version_identity=True,
            qualification_passes=True,
            reason="breaking change requires new version identity",
        )
    return ChangeClassificationVerdict(
        change_class=change_class,
        requires_new_version_identity=True,
        qualification_passes=False,
        reason="unhandled change class",
    )


def evaluate_declared_change_with_bump(
    change_class: CompatibilityChangeClass,
    declared_version_bump: bool,
) -> ChangeClassificationVerdict:
    base = evaluate_change_classification(change_class)
    if not base.qualification_passes:
        return base
    if base.requires_new_version_identity and not declared_version_bump:
        return ChangeClassificationVerdict(
            change_class=change_class,
            requires_new_version_identity=True,
            qualification_passes=False,
            reason="breaking change declared without required version identity bump",
        )
    if not base.requires_new_version_identity and declared_version_bump:
        return ChangeClassificationVerdict(
            change_class=change_class,
            requires_new_version_identity=False,
            qualification_passes=True,
            reason="optional bump allowed",
        )
    return base


@dataclass(frozen=True, slots=True)
class ShimCanonicalAuthorityProbe:
    claims_canonical_current_version: bool
    adapter_supported_old_version: str | None
    canonical_version_owner_path: str


def evaluate_shim_canonical_authority(probe: ShimCanonicalAuthorityProbe) -> bool:
    """Return True when qualification passes (shim is not canonical version owner)."""
    if probe.claims_canonical_current_version:
        return False
    if (
        probe.adapter_supported_old_version is not None
        and probe.adapter_supported_old_version == probe.canonical_version_owner_path
    ):
        return False
    return True
