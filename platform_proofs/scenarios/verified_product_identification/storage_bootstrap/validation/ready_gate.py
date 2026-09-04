"""READY gate aggregation — partial provider success is not READY."""

from __future__ import annotations

from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.results import (
    ValidationCheck,
    ValidationReport,
    ValidationStatus,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.manifest.model import (
    BootstrapState,
    VpiBootstrapManifest,
)


def evaluate_ready_gate(
    *,
    manifest: VpiBootstrapManifest,
    artifact_input_report: ValidationReport,
    catalog_report: ValidationReport,
    search_report: ValidationReport,
    checkpoint_complete: bool,
) -> ValidationReport:
    checks: list[ValidationCheck] = [
        *_report_checks("artifact_input", artifact_input_report),
        *_report_checks("catalog", catalog_report),
        *_report_checks("search", search_report),
        ValidationCheck(
            name="checkpoint_complete",
            status=ValidationStatus.PASS if checkpoint_complete else ValidationStatus.FAIL,
            detail=(
                f"checkpoint rows={manifest.checkpoint_rows_processed} "
                f"target={manifest.target_max_records}"
            ),
        ),
        ValidationCheck(
            name="catalog_counts_present",
            status=(
                ValidationStatus.PASS
                if manifest.catalog_source_offer_count > 0
                else ValidationStatus.FAIL
            ),
            detail=f"catalog_source_offer_count={manifest.catalog_source_offer_count}",
        ),
        ValidationCheck(
            name="search_points_present",
            status=ValidationStatus.PASS if manifest.search_point_count > 0 else ValidationStatus.FAIL,
            detail=f"search_point_count={manifest.search_point_count}",
        ),
    ]
    return ValidationReport.from_checks(tuple(checks))


def _report_checks(prefix: str, report: ValidationReport) -> tuple[ValidationCheck, ...]:
    return tuple(
        ValidationCheck(
            name=f"{prefix}.{check.name}",
            status=check.status,
            detail=check.detail,
        )
        for check in report.checks
    )
