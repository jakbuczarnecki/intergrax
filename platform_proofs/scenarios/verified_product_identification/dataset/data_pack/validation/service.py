"""Full Data Pack validation service — bounded-memory shard streaming."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.compatibility import (
    validate_shard_index_contract,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.build_state import (
    read_build_state_file,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.errors import (
    VpiDataPackBuildStateError,
    VpiDataPackFormatError,
    VpiDataPackIntegrityError,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.manifest import (
    DataPackManifest,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.paths import (
    resolve_data_pack_paths,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.ports import (
    DataPackReaderPort,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.shard_index import (
    ShardIndex,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.stores.filesystem_reader import (
    FilesystemDataPackReader,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.stores.parquet.embedding_codec import (
    read_embedding_parquet,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.stores.parquet.relational_codec import (
    read_relational_parquet,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.validation.contracts import (
    DataPackValidationFailureCategory,
    DataPackValidationPhase,
    DataPackValidationSummary,
    DataPackValidationVerdict,
    FullDataPackValidationReport,
    PhaseValidationResult,
    ShardValidationSummary,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.validation.global_validation import (
    DuplicatePartitionWriter,
    ShardBoundary,
    boundary_from_metrics,
    cleanup_scratch,
    finalize_duplicate_validation,
    validate_artifact_structure,
    validate_build_state_consistency,
    validate_checksums_extended,
    validate_finalization_semantics,
    validate_global_coverage,
    validate_manifest_identity,
    validate_shard_index_extended,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.validation.plan import (
    DataPackValidationExpectations,
    canonical_v1_validation_expectations,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.validation.shard_validation import (
    collect_shard_metrics,
    identity_lines_for_global_validation,
    validate_cross_shard_identity,
    validate_embedding_shard_records,
    validate_relational_shard_records,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.results import (
    ValidationCheck,
    ValidationStatus,
)

logger = logging.getLogger(__name__)


class DataPackValidationPreconditionError(RuntimeError):
    """Raised when validation cannot start due to missing artifact root or config."""


ProgressCallback = Callable[[str, int, int], None]


def _phase_result(
    phase: DataPackValidationPhase,
    checks: tuple[ValidationCheck, ...],
    *,
    failure_category: DataPackValidationFailureCategory | None,
    continue_validation: bool,
) -> PhaseValidationResult:
    return PhaseValidationResult(
        phase=phase,
        checks=checks,
        failure_category=failure_category,
        continue_validation=continue_validation,
    )


def _phase_passed(result: PhaseValidationResult) -> bool:
    return result.passed


def _emit_progress(
    callback: ProgressCallback | None,
    *,
    phase: str,
    current: int,
    total: int,
) -> None:
    if callback is not None:
        callback(phase, current, total)
    if current == 0 or current == total or current % max(1, total // 20) == 0:
        logger.info("validation progress phase=%s shard=%s/%s", phase, current, total)


def validate_full_data_pack(
    artifact_root: Path,
    *,
    expectations: DataPackValidationExpectations | None = None,
    reader: DataPackReaderPort | None = None,
    scratch_root: Path | None = None,
    keep_scratch: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> FullDataPackValidationReport:
    if not artifact_root.is_dir():
        raise DataPackValidationPreconditionError(f"artifact root not found: {artifact_root}")

    validation_expectations = expectations or canonical_v1_validation_expectations()
    paths = resolve_data_pack_paths(artifact_root)
    started = time.monotonic()
    timestamp = datetime.now(UTC).isoformat()

    pack_reader = reader or FilesystemDataPackReader(artifact_root)
    manifest: DataPackManifest | None = None
    shard_index: ShardIndex | None = None
    build_state = None
    phase_results: list[PhaseValidationResult] = []

    try:
        manifest = pack_reader.read_manifest()
    except (VpiDataPackIntegrityError, ValueError) as exc:
        phase_results.append(
            _phase_result(
                DataPackValidationPhase.MANIFEST_IDENTITY,
                (
                    ValidationCheck(
                        name="manifest_readable",
                        status=ValidationStatus.FAIL,
                        detail=str(exc),
                    ),
                ),
                failure_category=DataPackValidationFailureCategory.MANIFEST_FAIL,
                continue_validation=False,
            )
        )
    else:
        try:
            shard_index = pack_reader.read_shard_index()
        except (VpiDataPackFormatError, VpiDataPackIntegrityError, ValueError) as exc:
            phase_results.append(
                _phase_result(
                    DataPackValidationPhase.SHARD_INDEX,
                    (
                        ValidationCheck(
                            name="shard_index_readable",
                            status=ValidationStatus.FAIL,
                            detail=str(exc),
                        ),
                    ),
                    failure_category=DataPackValidationFailureCategory.SHARD_INDEX_FAIL,
                    continue_validation=False,
                )
            )

    structure_checks = validate_artifact_structure(paths, shard_index)
    structure_passed = all(check.status is ValidationStatus.PASS for check in structure_checks)
    phase_results.insert(
        0,
        _phase_result(
            DataPackValidationPhase.ARTIFACT_STRUCTURE,
            structure_checks,
            failure_category=(
                None if structure_passed else DataPackValidationFailureCategory.STRUCTURE_FAIL
            ),
            continue_validation=structure_passed,
        ),
    )

    if manifest is not None:
        manifest_checks = validate_manifest_identity(manifest, validation_expectations)
        manifest_passed = all(check.status is ValidationStatus.PASS for check in manifest_checks)
        phase_results.append(
            _phase_result(
                DataPackValidationPhase.MANIFEST_IDENTITY,
                manifest_checks,
                failure_category=(
                    None if manifest_passed else DataPackValidationFailureCategory.MANIFEST_FAIL
                ),
                continue_validation=manifest_passed,
            )
        )

    if shard_index is not None:
        index_checks = validate_shard_index_contract(shard_index).checks
        index_checks = index_checks + validate_shard_index_extended(
            shard_index,
            validation_expectations,
        )
        index_passed = all(check.status is ValidationStatus.PASS for check in index_checks)
        phase_results.append(
            _phase_result(
                DataPackValidationPhase.SHARD_INDEX,
                index_checks,
                failure_category=(
                    None if index_passed else DataPackValidationFailureCategory.SHARD_INDEX_FAIL
                ),
                continue_validation=index_passed,
            )
        )

    can_stream_shards = (
        manifest is not None
        and shard_index is not None
        and all(result.continue_validation for result in phase_results[:3])
    )

    relational_checks: list[ValidationCheck] = []
    embedding_checks: list[ValidationCheck] = []
    cross_checks: list[ValidationCheck] = []
    shard_summaries: list[ShardValidationSummary] = []
    boundaries: list[ShardBoundary] = []
    observed_relational = 0
    observed_embedding = 0
    non_finite_total = 0
    zero_vector_total = 0
    semantic_mismatch_total = 0
    first_invalid_shard: int | None = None

    scratch_path = scratch_root or (
        Path(".tmp") / "session" / "vpi-data-pack-validation" / "scratch"
    )
    duplicate_writer = DuplicatePartitionWriter(scratch_path)
    duplicate_summary = finalize_duplicate_validation(duplicate_writer)

    if can_stream_shards and manifest is not None and shard_index is not None:
        total_shards = validation_expectations.shard_count
        for ordinal in range(1, total_shards + 1):
            relational_descriptor = shard_index.relational_shards[ordinal - 1]
            embedding_descriptor = shard_index.embedding_shards[ordinal - 1]
            relational_path = paths.root / relational_descriptor.relative_path
            embedding_path = paths.root / embedding_descriptor.relative_path
            _emit_progress(progress_callback, phase="RELATIONAL", current=ordinal, total=total_shards)
            try:
                relational_records = read_relational_parquet(relational_path)
            except VpiDataPackIntegrityError as exc:
                relational_records = ()
                relational_checks.append(
                    ValidationCheck(
                        name=f"relational_shard_read_{ordinal}",
                        status=ValidationStatus.FAIL,
                        detail=str(exc),
                    )
                )
            relational_shard_checks = validate_relational_shard_records(
                relational_records,
                ordinal=ordinal,
                expectations=validation_expectations,
            )
            relational_checks.extend(relational_shard_checks)
            _emit_progress(progress_callback, phase="EMBEDDING", current=ordinal, total=total_shards)
            try:
                embedding_records = read_embedding_parquet(
                    embedding_path,
                    expected_dimension=manifest.embedding_identity.dimension,
                )
            except VpiDataPackIntegrityError as exc:
                embedding_records = ()
                embedding_checks.append(
                    ValidationCheck(
                        name=f"embedding_shard_read_{ordinal}",
                        status=ValidationStatus.FAIL,
                        detail=str(exc),
                    )
                )
            embedding_shard_checks = validate_embedding_shard_records(
                embedding_records,
                ordinal=ordinal,
                expectations=validation_expectations,
            )
            embedding_checks.extend(embedding_shard_checks)
            _emit_progress(
                progress_callback,
                phase="CROSS_IDENTITY",
                current=ordinal,
                total=total_shards,
            )
            cross_shard_checks = validate_cross_shard_identity(
                relational_records,
                embedding_records,
                ordinal=ordinal,
            )
            cross_checks.extend(cross_shard_checks)
            metrics = collect_shard_metrics(
                relational_records,
                embedding_records,
                ordinal=ordinal,
                expectations=validation_expectations,
            )
            observed_relational += metrics.relational_record_count
            observed_embedding += metrics.embedding_record_count
            non_finite_total += metrics.non_finite_vector_count
            zero_vector_total += metrics.zero_vector_count
            semantic_mismatch_total += metrics.semantic_hash_mismatch_count
            boundaries.append(boundary_from_metrics(ordinal, metrics))
            shard_passed = all(
                check.status is ValidationStatus.PASS
                for check in (
                    *relational_shard_checks,
                    *embedding_shard_checks,
                    *cross_shard_checks,
                )
            )
            if not shard_passed and first_invalid_shard is None:
                first_invalid_shard = ordinal
            shard_summaries.append(
                ShardValidationSummary(
                    ordinal=ordinal,
                    relational_record_count=metrics.relational_record_count,
                    embedding_record_count=metrics.embedding_record_count,
                    global_row_index_start=metrics.global_row_index_start,
                    global_row_index_end_exclusive=metrics.global_row_index_end_exclusive,
                    passed=shard_passed,
                )
            )
            for category, _key, value in identity_lines_for_global_validation(
                relational_records,
                embedding_records,
            ):
                duplicate_writer.write_line(category, value)

        duplicate_summary = finalize_duplicate_validation(duplicate_writer)

    if relational_checks:
        relational_passed = all(check.status is ValidationStatus.PASS for check in relational_checks)
        phase_results.append(
            _phase_result(
                DataPackValidationPhase.RELATIONAL_SHARD,
                tuple(relational_checks),
                failure_category=(
                    None if relational_passed else DataPackValidationFailureCategory.RELATIONAL_FAIL
                ),
                continue_validation=relational_passed,
            )
        )
    if embedding_checks:
        embedding_passed = all(check.status is ValidationStatus.PASS for check in embedding_checks)
        phase_results.append(
            _phase_result(
                DataPackValidationPhase.EMBEDDING_SHARD,
                tuple(embedding_checks),
                failure_category=(
                    None if embedding_passed else DataPackValidationFailureCategory.EMBEDDING_FAIL
                ),
                continue_validation=embedding_passed,
            )
        )
    if cross_checks:
        cross_passed = all(check.status is ValidationStatus.PASS for check in cross_checks)
        phase_results.append(
            _phase_result(
                DataPackValidationPhase.CROSS_ARTIFACT_IDENTITY,
                tuple(cross_checks),
                failure_category=(
                    None
                    if cross_passed
                    else DataPackValidationFailureCategory.CROSS_IDENTITY_FAIL
                ),
                continue_validation=cross_passed,
            )
        )

    coverage_summary, coverage_checks = validate_global_coverage(
        tuple(boundaries),
        expectations=validation_expectations,
        observed_relational_count=observed_relational,
        observed_embedding_count=observed_embedding,
    )
    coverage_passed = all(check.status is ValidationStatus.PASS for check in coverage_checks)
    phase_results.append(
        _phase_result(
            DataPackValidationPhase.GLOBAL_COVERAGE,
            coverage_checks,
            failure_category=(
                None if coverage_passed else DataPackValidationFailureCategory.COVERAGE_FAIL
            ),
            continue_validation=coverage_passed,
        )
    )

    duplicate_checks = (
        ValidationCheck(
            name="duplicate_global_row_index",
            status=ValidationStatus.PASS
            if duplicate_summary.duplicate_global_row_index_count == 0
            else ValidationStatus.FAIL,
            detail=str(duplicate_summary.duplicate_global_row_index_count),
        ),
        ValidationCheck(
            name="duplicate_source_ref",
            status=ValidationStatus.PASS
            if duplicate_summary.duplicate_source_ref_count == 0
            else ValidationStatus.FAIL,
            detail=str(duplicate_summary.duplicate_source_ref_count),
        ),
        ValidationCheck(
            name="duplicate_logical_point_id",
            status=ValidationStatus.PASS
            if duplicate_summary.duplicate_logical_point_id_count == 0
            else ValidationStatus.FAIL,
            detail=str(duplicate_summary.duplicate_logical_point_id_count),
        ),
    )
    duplicate_passed = duplicate_summary.passed
    phase_results.append(
        _phase_result(
            DataPackValidationPhase.GLOBAL_DUPLICATES,
            duplicate_checks,
            failure_category=(
                None if duplicate_passed else DataPackValidationFailureCategory.DUPLICATE_FAIL
            ),
            continue_validation=duplicate_passed,
        )
    )

    _emit_progress(progress_callback, phase="CHECKSUMS", current=1, total=1)
    checksum_checks = validate_checksums_extended(paths)
    checksum_passed = all(check.status is ValidationStatus.PASS for check in checksum_checks)
    phase_results.append(
        _phase_result(
            DataPackValidationPhase.CHECKSUMS,
            checksum_checks,
            failure_category=(
                None if checksum_passed else DataPackValidationFailureCategory.INTEGRITY_FAIL
            ),
            continue_validation=checksum_passed,
        )
    )

    try:
        build_state = read_build_state_file(paths.build_state_file)
    except VpiDataPackBuildStateError as exc:
        build_state_checks = (
            ValidationCheck(
                name="build_state_readable",
                status=ValidationStatus.FAIL,
                detail=str(exc),
            ),
        )
        build_state_passed = False
    else:
        if manifest is not None and shard_index is not None:
            build_state_checks = validate_build_state_consistency(
                build_state,
                paths=paths,
                manifest=manifest,
                shard_index=shard_index,
                expectations=validation_expectations,
            )
            build_state_passed = all(
                check.status is ValidationStatus.PASS for check in build_state_checks
            )
        else:
            build_state_checks = (
                ValidationCheck(
                    name="build_state_prerequisite",
                    status=ValidationStatus.FAIL,
                    detail="manifest and shard index required",
                ),
            )
            build_state_passed = False

    phase_results.append(
        _phase_result(
            DataPackValidationPhase.BUILD_STATE,
            build_state_checks,
            failure_category=(
                None if build_state_passed else DataPackValidationFailureCategory.BUILD_STATE_FAIL
            ),
            continue_validation=build_state_passed,
        )
    )

    mandatory_passed = all(_phase_passed(result) for result in phase_results[:-1])
    if manifest is not None and build_state is not None:
        finalization_checks = validate_finalization_semantics(
            paths=paths,
            manifest=manifest,
            build_state=build_state,
            expectations=validation_expectations,
            prior_phases_passed=mandatory_passed,
        )
    else:
        finalization_checks = (
            ValidationCheck(
                name="finalization_prerequisite",
                status=ValidationStatus.FAIL,
                detail="manifest and build state required",
            ),
        )
    finalization_passed = all(check.status is ValidationStatus.PASS for check in finalization_checks)
    phase_results.append(
        _phase_result(
            DataPackValidationPhase.FINALIZATION,
            finalization_checks,
            failure_category=(
                None if finalization_passed else DataPackValidationFailureCategory.FINALIZATION_FAIL
            ),
            continue_validation=finalization_passed,
        )
    )

    if not keep_scratch:
        cleanup_scratch(scratch_path)

    elapsed = time.monotonic() - started
    logger.info(
        "validation complete elapsed_seconds=%.2f records=%s verdict_pending",
        elapsed,
        observed_relational,
    )

    ready_shard_count = 0
    if build_state is not None:
        ready_shard_count = build_state.completed_shards

    finalized_valid = finalization_passed and any(
        check.name == "finalization_artifact_valid" and check.status is ValidationStatus.PASS
        for check in finalization_checks
    )

    summary = DataPackValidationSummary(
        artifact_root=str(artifact_root),
        validation_timestamp_utc=timestamp,
        expected_record_count=validation_expectations.record_count,
        observed_relational_count=observed_relational,
        observed_embedding_count=observed_embedding,
        relational_shard_count=(
            len(shard_index.relational_shards) if shard_index is not None else 0
        ),
        embedding_shard_count=len(shard_index.embedding_shards) if shard_index is not None else 0,
        ready_shard_count=ready_shard_count,
        first_invalid_shard_ordinal=first_invalid_shard,
        embedding_dimension=validation_expectations.embedding_dimension,
        non_finite_vector_count=non_finite_total,
        zero_vector_count=zero_vector_total,
        semantic_hash_mismatch_count=semantic_mismatch_total,
        finalized_artifact_valid=finalized_valid,
    )

    verdict = (
        DataPackValidationVerdict.PASS
        if all(_phase_passed(result) for result in phase_results)
        else DataPackValidationVerdict.FAIL
    )

    return FullDataPackValidationReport(
        verdict=verdict,
        summary=summary,
        phase_results=tuple(phase_results),
        coverage_summary=coverage_summary,
        duplicate_summary=duplicate_summary,
        shard_summaries=tuple(shard_summaries),
    )
