# © Artur Czarnecki. All rights reserved.

"""EC3 qualification integrity checks."""

from __future__ import annotations

from testing_support.obs_diag_observability_vendor_qualification.descriptor import (
    ObservabilityQualifiedPathRow,
    ObservabilityVendorQualificationRow,
    ObservabilityVendorQualificationStatus,
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
