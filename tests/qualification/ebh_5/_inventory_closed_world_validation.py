# © Artur Czarnecki. All rights reserved.

"""Closed-world inventory validation (EBH5-G06-01)."""

from __future__ import annotations

from typing import Final

from tests.qualification.ebh_5._behavioral_proof_registry import QUALIFIED_CONCERN_BEHAVIORAL_PROOF_NODE
from tests.qualification.ebh_5._roadmap_domain_na_evidence import (
    roadmap_domain_na_dispositions,
    roadmap_domain_na_validation_errors,
)
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
    },
)

_PENDING_BEHAVIORAL_EVIDENCE: Final[str] = "PENDING — EBH_5 Package C"

_DOMAIN_NA_WITH_EVIDENCE: dict[RoadmapVendorDomain, str] = {
    disp.domain: f"N/A — WITH EVIDENCE: {disp.rationale}" for disp in roadmap_domain_na_dispositions()
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
        if any(ph in value for ph in _FORBIDDEN_PLACEHOLDERS):
            errors.append(f"{cid}: forbidden placeholder in {field_name}")

    behavioral = row.behavioral_qualification_evidence_id.strip()
    if any(ph in behavioral for ph in _FORBIDDEN_PLACEHOLDERS):
        errors.append(f"{cid}: forbidden placeholder in behavioral_qualification_evidence_id")

    if row.extensibility_classification == ExtensibilityClassification.REPLACEABLE_NOT_YET:
        if behavioral == "N/A" and row.outstanding_debt.strip():
            pass  # debt-tracked NOT YET without Package C L3 (e.g. OD-EBH5-06)
        elif behavioral == _PENDING_BEHAVIORAL_EVIDENCE:
            if not row.outstanding_debt.strip():
                errors.append(f"{cid}: NOT YET PENDING behavioral requires outstanding_debt")
        elif behavioral.startswith("tests/"):
            errors.append(f"{cid}: NOT YET must not cite tests/ proof without QUALIFIED classification")
        elif "PENDING" not in behavioral.upper():
            errors.append(
                f"{cid}: NOT YET behavioral evidence must be {_PENDING_BEHAVIORAL_EVIDENCE!r} "
                "or contain explicit PENDING marker with debt",
            )
        elif not row.outstanding_debt.strip():
            errors.append(f"{cid}: NOT YET custom PENDING behavioral requires outstanding_debt")

    if row.extensibility_classification == ExtensibilityClassification.REPLACEABLE_QUALIFIED:
        expected = QUALIFIED_CONCERN_BEHAVIORAL_PROOF_NODE.get(cid)
        if expected is None:
            errors.append(f"{cid}: QUALIFIED without behavioral proof registry entry")
        elif behavioral != expected:
            errors.append(f"{cid}: QUALIFIED behavioral must match registry node")

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

    errors.extend(roadmap_domain_na_validation_errors())

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
