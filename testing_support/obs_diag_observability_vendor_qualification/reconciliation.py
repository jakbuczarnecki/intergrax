# © Artur Czarnecki. All rights reserved.

"""EC3 qualification integrity checks."""

from __future__ import annotations

from testing_support.obs_diag_observability_vendor_qualification.descriptor import (
    ObservabilityQualifiedPathRow,
    ObservabilityVendorQualificationEvidence,
    ObservabilityVendorQualificationRow,
    ObservabilityVendorQualificationStatus,
    PlatformIsolationProofReference,
)


def observability_vendor_live_qualified_without_evidence(
    rows: tuple[ObservabilityVendorQualificationRow, ...],
) -> list[str]:
    violations: list[str] = []
    for row in rows:
        if row.qualification != ObservabilityVendorQualificationStatus.LIVE_QUALIFIED:
            continue
        if row.evidence.missing_live_categories():
            violations.append(row.provider_id)
    return violations


def observability_qualified_path_without_evidence(
    paths: tuple[ObservabilityQualifiedPathRow, ...],
) -> list[str]:
    violations: list[str] = []
    for path in paths:
        if path.qualification != ObservabilityVendorQualificationStatus.LIVE_QUALIFIED:
            continue
        missing = path.evidence.missing_live_categories()
        if path.platform_isolation is not None:
            missing = tuple(item for item in missing if item != "canonical_truth_isolation")
        if not path.privacy_required:
            missing = tuple(item for item in missing if item != "privacy")
        if missing:
            violations.append(path.path_id)
    return violations


def _vendor_evidence_references_platform_isolation(
    evidence: ObservabilityVendorQualificationEvidence,
) -> bool:
    for ref in (
        evidence.normal_delivery,
        evidence.failure_isolation,
        evidence.recovery,
        evidence.canonical_truth_isolation,
        evidence.privacy,
    ):
        if isinstance(ref, PlatformIsolationProofReference):
            return True
    return False


def observability_vendor_rows_borrowing_platform_canonical_isolation(
    rows: tuple[ObservabilityVendorQualificationRow, ...],
) -> list[str]:
    """Vendor rows must not attach platform isolation proof references to vendor evidence."""
    violations: list[str] = []
    for row in rows:
        if _vendor_evidence_references_platform_isolation(row.evidence):
            violations.append(row.provider_id)
    return violations
