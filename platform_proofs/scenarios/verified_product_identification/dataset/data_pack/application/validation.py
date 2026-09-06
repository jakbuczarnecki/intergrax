"""Cross-artifact and per-artifact validation."""

from __future__ import annotations

import math

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.embedding import (
    EmbeddingDataPackRecord,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.errors import (
    VpiDataPackValidationError,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.identity import (
    semantic_text_hash,
    source_ref_key,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.relational import (
    RelationalDataPackRecord,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.results import (
    ValidationCheck,
    ValidationReport,
    ValidationStatus,
)


def validate_relational_records(
    records: tuple[RelationalDataPackRecord, ...],
    *,
    expected_count: int,
) -> ValidationReport:
    checks: list[ValidationCheck] = []
    if len(records) != expected_count:
        checks.append(
            ValidationCheck(
                name="relational_row_count",
                status=ValidationStatus.FAIL,
                detail=f"count={len(records)} expected={expected_count}",
            )
        )
    else:
        checks.append(
            ValidationCheck(
                name="relational_row_count",
                status=ValidationStatus.PASS,
                detail=f"count={len(records)}",
            )
        )
    refs = [source_ref_key(record.source_ref) for record in records]
    if len(set(refs)) != len(refs):
        checks.append(
            ValidationCheck(
                name="relational_duplicate_refs",
                status=ValidationStatus.FAIL,
                detail="duplicate source_record_ref detected",
            )
        )
    else:
        checks.append(
            ValidationCheck(
                name="relational_duplicate_refs",
                status=ValidationStatus.PASS,
                detail="all source_record_ref values unique",
            )
        )
    return ValidationReport.from_checks(tuple(checks))


def validate_embedding_records(
    records: tuple[EmbeddingDataPackRecord, ...],
    *,
    expected_count: int,
    expected_dimension: int,
) -> ValidationReport:
    checks: list[ValidationCheck] = []
    if len(records) != expected_count:
        checks.append(
            ValidationCheck(
                name="embedding_row_count",
                status=ValidationStatus.FAIL,
                detail=f"count={len(records)} expected={expected_count}",
            )
        )
    else:
        checks.append(
            ValidationCheck(
                name="embedding_row_count",
                status=ValidationStatus.PASS,
                detail=f"count={len(records)}",
            )
        )
    refs = [source_ref_key(record.source_ref) for record in records]
    if len(set(refs)) != len(refs):
        checks.append(
            ValidationCheck(
                name="embedding_duplicate_refs",
                status=ValidationStatus.FAIL,
                detail="duplicate source_record_ref detected",
            )
        )
    else:
        checks.append(
            ValidationCheck(
                name="embedding_duplicate_refs",
                status=ValidationStatus.PASS,
                detail="all source_record_ref values unique",
            )
        )

    dimension_ok = True
    finite_ok = True
    for record in records:
        if record.embedding_dimension != expected_dimension:
            dimension_ok = False
        for value in record.dense_embedding:
            if not math.isfinite(value):
                finite_ok = False
                break
    checks.append(
        ValidationCheck(
            name="embedding_dimension",
            status=ValidationStatus.PASS if dimension_ok else ValidationStatus.FAIL,
            detail=f"expected_dimension={expected_dimension}",
        )
    )
    checks.append(
        ValidationCheck(
            name="embedding_finite_vectors",
            status=ValidationStatus.PASS if finite_ok else ValidationStatus.FAIL,
            detail="all vector values finite",
        )
    )
    return ValidationReport.from_checks(tuple(checks))


def validate_cross_artifact_identity(
    relational_records: tuple[RelationalDataPackRecord, ...],
    embedding_records: tuple[EmbeddingDataPackRecord, ...],
    *,
    expected_refs: frozenset[tuple[str, str, str | None]] | None = None,
) -> ValidationReport:
    relational_refs = {source_ref_key(record.source_ref) for record in relational_records}
    embedding_refs = {source_ref_key(record.source_ref) for record in embedding_records}
    refs_equal = relational_refs == embedding_refs
    checks: list[ValidationCheck] = [
        ValidationCheck(
            name="relational_embedding_ref_equality",
            status=ValidationStatus.PASS if refs_equal else ValidationStatus.FAIL,
            detail=(
                f"relational={len(relational_refs)} embedding={len(embedding_refs)} "
                f"equal={refs_equal}"
            ),
        )
    ]
    if expected_refs is not None:
        checks.append(
            ValidationCheck(
                name="expected_sample_ref_equality",
                status=ValidationStatus.PASS if relational_refs == expected_refs else ValidationStatus.FAIL,
                detail=f"expected={len(expected_refs)} actual={len(relational_refs)}",
            )
        )
    return ValidationReport.from_checks(tuple(checks))


def validate_semantic_text_hashes(
    relational_records: tuple[RelationalDataPackRecord, ...],
    embedding_records: tuple[EmbeddingDataPackRecord, ...],
) -> ValidationReport:
    relational_by_ref = {source_ref_key(record.source_ref): record for record in relational_records}
    embedding_by_ref = {source_ref_key(record.source_ref): record for record in embedding_records}
    mismatches: list[str] = []
    for ref, relational_record in relational_by_ref.items():
        embedding_record = embedding_by_ref.get(ref)
        if embedding_record is None:
            mismatches.append(f"missing embedding for {ref}")
            continue
        expected_hash = semantic_text_hash(relational_record.semantic_text)
        if relational_record.semantic_text_hash != expected_hash:
            mismatches.append(f"relational hash mismatch for {ref}")
        if embedding_record.semantic_text_hash != expected_hash:
            mismatches.append(f"embedding hash mismatch for {ref}")
    status = ValidationStatus.PASS if not mismatches else ValidationStatus.FAIL
    detail = "all semantic_text_hash values match" if not mismatches else "; ".join(mismatches[:5])
    return ValidationReport.from_checks(
        (
            ValidationCheck(
                name="semantic_text_hash_validation",
                status=status,
                detail=detail,
            ),
        )
    )


def assert_validation_pass(report: ValidationReport, *, stage: str) -> None:
    if report.status is not ValidationStatus.PASS:
        failed = [check.name for check in report.checks if check.status is ValidationStatus.FAIL]
        raise VpiDataPackValidationError(f"{stage} failed: {', '.join(failed)}")
