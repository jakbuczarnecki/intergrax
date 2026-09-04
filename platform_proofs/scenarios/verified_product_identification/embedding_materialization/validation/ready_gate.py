"""READY gate for embedding artifact materialization."""

from __future__ import annotations

from platform_proofs.scenarios.verified_product_identification.embedding_materialization.manifest.model import (
    EmbeddingArtifactManifest,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.results import (
    ValidationCheck,
    ValidationReport,
    ValidationStatus,
)


def evaluate_artifact_ready_gate(
    *,
    manifest: EmbeddingArtifactManifest,
    artifact_report: ValidationReport,
    checkpoint_complete: bool,
) -> ValidationReport:
    checks: list[ValidationCheck] = list(artifact_report.checks)
    checks.append(
        ValidationCheck(
            name="checkpoint_complete",
            status=ValidationStatus.PASS if checkpoint_complete else ValidationStatus.FAIL,
            detail=(
                f"rows={manifest.checkpoint_rows_materialized} "
                f"target={manifest.target_max_records or manifest.dataset_record_count}"
            ),
        )
    )
    return ValidationReport.from_checks(tuple(checks))
