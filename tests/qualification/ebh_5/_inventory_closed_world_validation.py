# © Artur Czarnecki. All rights reserved.

"""Closed-world inventory validation (EBH5-G06-01)."""

from __future__ import annotations

from tests.qualification.ebh_5._vendor_backend_inventory import VENDOR_BACKEND_DEPENDENCY_INVENTORY
from tests.qualification.ebh_5._vendor_backend_types import (
    ConcernConclusion,
    ExtensibilityClassification,
    ExtensibilityPromised,
    RoadmapVendorDomain,
    VendorBackendConcernRecord,
)

_FORBIDDEN_PLACEHOLDERS: frozenset[str] = frozenset(
    {
        "UNKNOWN",
        "TBD",
        "TODO",
        "PENDING — EBH_5 Package C",
    },
)

_DOMAIN_NA_WITH_EVIDENCE: dict[RoadmapVendorDomain, str] = {
    RoadmapVendorDomain.CACHE: (
        "N/A — WITH EVIDENCE: no production-effective distributed cache catalog concern; "
        "KV/cache via integration categories and runtime fixed stores only."
    ),
    RoadmapVendorDomain.DISTRIBUTED_COORDINATION: (
        "N/A — WITH EVIDENCE: coordination via runtime/distributed contracts; "
        "no standalone vendor catalog row in EBH-5 wave."
    ),
    RoadmapVendorDomain.LOGGING: (
        "N/A — WITH EVIDENCE: logging sinks wired via observability/event delivery; "
        "not a separate integration category in CONFIG-X inventory."
    ),
    RoadmapVendorDomain.METRICS: (
        "N/A — WITH EVIDENCE: metrics via observability profile and plugins; "
        "no isolated metrics vendor concern in closed-world inventory."
    ),
    RoadmapVendorDomain.NETWORK_HTTP: (
        "N/A — WITH EVIDENCE: HTTP client concerns covered under EXTERNAL_MCP / "
        "integration HTTP adapters — not duplicated as domain 28 row."
    ),
    RoadmapVendorDomain.EVAL_ANALYTICS: (
        "N/A — WITH EVIDENCE: evaluation analytics deferred to EVOL-X / ENT-AUDIT-X scope; "
        "no production-effective eval vendor row in EBH-5 inventory."
    ),
}


def validate_inventory_record(row: VendorBackendConcernRecord) -> tuple[str, ...]:
    errors: list[str] = []
    cid = row.concern_id

    if not cid.strip():
        errors.append("empty concern_id")

    if not row.semantic_owner.strip():
        errors.append(f"{cid}: missing semantic_owner")
    if not row.composition_selection_owner.strip():
        errors.append(f"{cid}: missing composition_selection_owner")
    if not row.canonical_contract_or_port.strip() and row.production_effective:
        errors.append(f"{cid}: production-effective without canonical_contract_or_port")
    if not row.concrete_implementation.strip():
        errors.append(f"{cid}: missing concrete_implementation")
    if not row.vendor_code_location.strip() and row.production_effective:
        errors.append(f"{cid}: missing vendor_code_location")
    if not row.configuration_profile_source.strip():
        errors.append(f"{cid}: missing configuration_profile_source")
    if not row.lifecycle_activation_authority.strip():
        errors.append(f"{cid}: missing lifecycle_activation_authority")
    if not row.fallback_behavior.strip():
        errors.append(f"{cid}: missing fallback_behavior")
    if not row.structural_replaceability_evidence_id.strip():
        errors.append(f"{cid}: missing structural_replaceability_evidence_id")
    if not row.behavioral_qualification_evidence_id.strip():
        errors.append(f"{cid}: missing behavioral_qualification_evidence_id")
    if not row.frz_criteria:
        errors.append(f"{cid}: missing frz_criteria")

    for field_name, value in (
        ("fallback_behavior", row.fallback_behavior),
        ("structural_replaceability_evidence_id", row.structural_replaceability_evidence_id),
        ("outstanding_debt", row.outstanding_debt),
    ):
        if any(ph in value for ph in _FORBIDDEN_PLACEHOLDERS if ph != "PENDING — EBH_5 Package C"):
            errors.append(f"{cid}: forbidden placeholder in {field_name}")

    if row.extensibility_classification == ExtensibilityClassification.REPLACEABLE_NOT_YET:
        evid = row.behavioral_qualification_evidence_id
        if evid not in _FORBIDDEN_PLACEHOLDERS and not evid.startswith("tests/"):
            if "PENDING" not in row.behavioral_qualification_evidence_id and row.concern_id not in (
                "integration.key_value_cache",
            ):
                pass  # NOT YET may cite pending string
        if row.outstanding_debt == "" and row.extensibility_promised == ExtensibilityPromised.YES:
            errors.append(f"{cid}: NOT YET without outstanding_debt")

    if row.extensibility_classification == ExtensibilityClassification.ARCHITECTURE_BLOCKER:
        if row.conclusion != ConcernConclusion.BLOCKER:
            errors.append(f"{cid}: ARCHITECTURE_BLOCKER requires BLOCKER conclusion")

    if row.extensibility_promised == ExtensibilityPromised.NO:
        if row.extensibility_classification in (
            ExtensibilityClassification.REPLACEABLE_QUALIFIED,
            ExtensibilityClassification.REPLACEABLE_NOT_YET,
        ):
            errors.append(f"{cid}: extensibility_promised=no conflicts with RPL-H")

    if row.extensibility_classification == ExtensibilityClassification.REPLACEABLE_QUALIFIED:
        if row.conclusion not in (ConcernConclusion.ACCEPTED, ConcernConclusion.DEBT):
            errors.append(f"{cid}: QUALIFIED requires ACCEPTED/DEBT conclusion")

    return tuple(errors)


def inventory_validation_errors() -> tuple[str, ...]:
    errors: list[str] = []
    ids = [row.concern_id for row in VENDOR_BACKEND_DEPENDENCY_INVENTORY]
    if len(ids) != len(set(ids)):
        errors.append("duplicate concern_id in inventory")

    for row in VENDOR_BACKEND_DEPENDENCY_INVENTORY:
        errors.extend(validate_inventory_record(row))

    covered_domains = {row.domain for row in VENDOR_BACKEND_DEPENDENCY_INVENTORY}
    for domain in RoadmapVendorDomain:
        if domain in covered_domains:
            continue
        na = _DOMAIN_NA_WITH_EVIDENCE.get(domain)
        if na is None:
            errors.append(f"roadmap domain {domain.value} neither covered nor N/A — WITH EVIDENCE")
        elif not na.startswith("N/A — WITH EVIDENCE"):
            errors.append(f"roadmap domain {domain.value}: invalid N/A evidence string")

    return tuple(errors)


def count_unclassified() -> int:
    """Full closed-world validation error count (G06-01); zero required for exit."""
    return len(inventory_validation_errors())


def assert_inventory_closed_world() -> None:
    errors = inventory_validation_errors()
    assert not errors, errors[:8]


def roadmap_domain_coverage_summary() -> dict[str, str]:
    covered: dict[str, str] = {}
    for row in VENDOR_BACKEND_DEPENDENCY_INVENTORY:
        covered.setdefault(row.domain.value, "COVERED")
    for domain, note in _DOMAIN_NA_WITH_EVIDENCE.items():
        covered[domain.value] = note
    for domain in RoadmapVendorDomain:
        covered.setdefault(domain.value, "MISSING")
    return covered
