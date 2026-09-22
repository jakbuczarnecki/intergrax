# © Artur Czarnecki. All rights reserved.

"""EC3 qualification integrity checks."""

from __future__ import annotations

from testing_support.obs_diag_observability_vendor_qualification.descriptor import (
    ObservabilityQualifiedPathRow,
    ObservabilityVendorQualificationRow,
    ObservabilityVendorQualificationStatus,
)
from testing_support.obs_diag_observability_vendor_qualification.inventory import (
    PLATFORM_CANONICAL_TRUTH_ISOLATION_EVIDENCE,
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
        if not path.privacy_required:
            missing = tuple(item for item in missing if item != "privacy")
        if missing:
            violations.append(path.path_id)
    return violations


def observability_vendor_rows_borrowing_platform_canonical_isolation(
    rows: tuple[ObservabilityVendorQualificationRow, ...],
) -> list[str]:
    """Platform OTLP isolation proof must not appear on vendor-specific evidence rows."""
    platform_ref = PLATFORM_CANONICAL_TRUTH_ISOLATION_EVIDENCE.canonical_truth_isolation
    violations: list[str] = []
    for row in rows:
        vendor_ref = row.evidence.canonical_truth_isolation
        if vendor_ref is not None and vendor_ref == platform_ref:
            violations.append(row.provider_id)
    return violations
