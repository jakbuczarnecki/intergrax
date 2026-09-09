"""Global coverage, duplicate detection, and checksum validation."""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.checksums import (
    sha256_file,
    verify_sha256sums,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.build_state import (
    DataPackBuildState,
    DataPackShardStatus,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.errors import (
    VpiDataPackIntegrityError,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.manifest import (
    DataPackManifest,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.paths import (
    DataPackPaths,
    shard_file_name,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.shard_index import (
    ShardIndex,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.validation.contracts import (
    DuplicateValidationSummary,
    GlobalCoverageSummary,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.validation.plan import (
    DataPackValidationExpectations,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.validation.shard_validation import (
    ShardValidationMetrics,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.results import (
    ValidationCheck,
    ValidationStatus,
)


@dataclass(frozen=True, slots=True)
class ShardBoundary:
    ordinal: int
    start_row_index: int
    end_row_index_exclusive: int


def _check(name: str, passed: bool, detail: str) -> ValidationCheck:
    return ValidationCheck(
        name=name,
        status=ValidationStatus.PASS if passed else ValidationStatus.FAIL,
        detail=detail,
    )


def validate_artifact_structure(
    paths: DataPackPaths,
    shard_index: ShardIndex | None,
) -> tuple[ValidationCheck, ...]:
    required_dirs = (
        paths.manifest_dir,
        paths.relational_dir,
        paths.embeddings_dir,
        paths.indexes_dir,
        paths.checksums_dir,
        paths.state_dir,
        paths.evidence_dir,
    )
    required_files = (
        paths.manifest_file,
        paths.shards_index_file,
        paths.checksums_file,
        paths.build_state_file,
    )
    checks: list[ValidationCheck] = []
    for directory in required_dirs:
        checks.append(
            _check(
                f"structure_dir_{directory.name}",
                directory.is_dir(),
                str(directory),
            )
        )
    for file_path in required_files:
        checks.append(
            _check(
                f"structure_file_{file_path.name}",
                file_path.is_file(),
                str(file_path),
            )
        )
    if shard_index is None:
        return tuple(checks)

    seen_relational: set[int] = set()
    seen_embedding: set[int] = set()
    for descriptor in shard_index.relational_shards:
        duplicate = descriptor.ordinal in seen_relational
        seen_relational.add(descriptor.ordinal)
        expected_name = shard_file_name(descriptor.ordinal)
        filename_ok = Path(descriptor.relative_path).name == expected_name
        file_exists = (paths.root / descriptor.relative_path).is_file()
        checks.extend(
            (
                _check(
                    f"structure_relational_duplicate_{descriptor.ordinal}",
                    not duplicate,
                    "duplicate relational ordinal",
                ),
                _check(
                    f"structure_relational_filename_{descriptor.ordinal}",
                    filename_ok,
                    f"expected={expected_name}",
                ),
                _check(
                    f"structure_relational_exists_{descriptor.ordinal}",
                    file_exists,
                    descriptor.relative_path,
                ),
            )
        )
    for descriptor in shard_index.embedding_shards:
        duplicate = descriptor.ordinal in seen_embedding
        seen_embedding.add(descriptor.ordinal)
        expected_name = shard_file_name(descriptor.ordinal)
        filename_ok = Path(descriptor.relative_path).name == expected_name
        file_exists = (paths.root / descriptor.relative_path).is_file()
        checks.extend(
            (
                _check(
                    f"structure_embedding_duplicate_{descriptor.ordinal}",
                    not duplicate,
                    "duplicate embedding ordinal",
                ),
                _check(
                    f"structure_embedding_filename_{descriptor.ordinal}",
                    filename_ok,
                    f"expected={expected_name}",
                ),
                _check(
                    f"structure_embedding_exists_{descriptor.ordinal}",
                    file_exists,
                    descriptor.relative_path,
                ),
            )
        )
    return tuple(checks)


def validate_manifest_identity(
    manifest: DataPackManifest,
    expectations: DataPackValidationExpectations,
) -> tuple[ValidationCheck, ...]:
    embedding = manifest.embedding_identity
    return (
        _check(
            "manifest_record_count",
            manifest.record_count == expectations.record_count,
            f"expected={expectations.record_count} actual={manifest.record_count}",
        ),
        _check(
            "manifest_data_pack_version",
            manifest.data_pack_version == expectations.data_pack_version,
            manifest.data_pack_version,
        ),
        _check(
            "manifest_embedding_provider",
            embedding.provider == expectations.embedding_provider,
            embedding.provider,
        ),
        _check(
            "manifest_embedding_model",
            embedding.model == expectations.embedding_model,
            embedding.model,
        ),
        _check(
            "manifest_embedding_revision",
            embedding.model_revision == expectations.embedding_model_revision,
            str(embedding.model_revision),
        ),
        _check(
            "manifest_embedding_dimension",
            embedding.dimension == expectations.embedding_dimension,
            str(embedding.dimension),
        ),
        _check(
            "manifest_input_policy_version",
            embedding.input_policy_version == expectations.input_policy_version,
            embedding.input_policy_version,
        ),
        _check(
            "manifest_source_dataset_name",
            manifest.source_dataset.dataset_name == expectations.source_dataset_name,
            manifest.source_dataset.dataset_name,
        ),
        _check(
            "manifest_source_dataset_sha256",
            manifest.source_dataset.dataset_sha256 == expectations.source_dataset_sha256,
            manifest.source_dataset.dataset_sha256,
        ),
        _check(
            "manifest_shard_count",
            manifest.shard_count == expectations.shard_count,
            f"expected={expectations.shard_count} actual={manifest.shard_count}",
        ),
        _check(
            "manifest_shard_size_implied",
            manifest.record_count == expectations.record_count,
            f"record_count={manifest.record_count}",
        ),
    )


def validate_shard_index_extended(
    shard_index: ShardIndex,
    expectations: DataPackValidationExpectations,
) -> tuple[ValidationCheck, ...]:
    checks: list[ValidationCheck] = [
        _check(
            "shard_index_relational_count",
            len(shard_index.relational_shards) == expectations.shard_count,
            f"expected={expectations.shard_count}",
        ),
        _check(
            "shard_index_embedding_count",
            len(shard_index.embedding_shards) == expectations.shard_count,
            f"expected={expectations.shard_count}",
        ),
    ]
    for ordinal in range(1, expectations.shard_count + 1):
        expected_records = expectations.expected_shard_record_count(ordinal)
        if ordinal > len(shard_index.relational_shards):
            checks.append(
                _check(
                    f"shard_index_relational_missing_{ordinal}",
                    False,
                    "missing relational descriptor",
                )
            )
            continue
        if ordinal > len(shard_index.embedding_shards):
            checks.append(
                _check(
                    f"shard_index_embedding_missing_{ordinal}",
                    False,
                    "missing embedding descriptor",
                )
            )
            continue
        relational = shard_index.relational_shards[ordinal - 1]
        embedding = shard_index.embedding_shards[ordinal - 1]
        checks.extend(
            (
                _check(
                    f"shard_index_relational_ordinal_{ordinal}",
                    relational.ordinal == ordinal,
                    f"actual={relational.ordinal}",
                ),
                _check(
                    f"shard_index_embedding_ordinal_{ordinal}",
                    embedding.ordinal == ordinal,
                    f"actual={embedding.ordinal}",
                ),
                _check(
                    f"shard_index_relational_record_count_{ordinal}",
                    relational.record_count == expected_records,
                    f"expected={expected_records} actual={relational.record_count}",
                ),
                _check(
                    f"shard_index_embedding_record_count_{ordinal}",
                    embedding.record_count == expected_records,
                    f"expected={expected_records} actual={embedding.record_count}",
                ),
                _check(
                    f"shard_index_relational_schema_{ordinal}",
                    relational.schema_version == expectations.relational_schema_version,
                    relational.schema_version,
                ),
                _check(
                    f"shard_index_embedding_schema_{ordinal}",
                    embedding.schema_version == expectations.embedding_schema_version,
                    embedding.schema_version,
                ),
            )
        )
    return tuple(checks)


def validate_global_coverage(
    boundaries: tuple[ShardBoundary, ...],
    *,
    expectations: DataPackValidationExpectations,
    observed_relational_count: int,
    observed_embedding_count: int,
) -> tuple[GlobalCoverageSummary, tuple[ValidationCheck, ...]]:
    checks: list[ValidationCheck] = []
    first_gap: int | None = None
    first_overlap: int | None = None
    final_end: int | None = None
    if boundaries:
        previous_end = boundaries[0].start_row_index
        if previous_end != 0:
            first_gap = boundaries[0].ordinal
        for boundary in boundaries:
            if boundary.start_row_index != previous_end:
                if boundary.start_row_index > previous_end and first_gap is None:
                    first_gap = boundary.ordinal
                if boundary.start_row_index < previous_end and first_overlap is None:
                    first_overlap = boundary.ordinal
            previous_end = boundary.end_row_index_exclusive
            final_end = boundary.end_row_index_exclusive
            expected_start, expected_end = expectations.expected_global_row_range(boundary.ordinal)
            checks.extend(
                (
                    _check(
                        f"coverage_shard_{boundary.ordinal}_start",
                        boundary.start_row_index == expected_start,
                        f"expected={expected_start} actual={boundary.start_row_index}",
                    ),
                    _check(
                        f"coverage_shard_{boundary.ordinal}_end",
                        boundary.end_row_index_exclusive == expected_end,
                        f"expected={expected_end} actual={boundary.end_row_index_exclusive}",
                    ),
                )
            )
        if final_end != expectations.record_count:
            checks.append(
                _check(
                    "coverage_final_end",
                    False,
                    f"expected={expectations.record_count} actual={final_end}",
                )
            )
        else:
            checks.append(
                _check(
                    "coverage_final_end",
                    True,
                    f"final_end={final_end}",
                )
            )
    checks.extend(
        (
            _check(
                "coverage_relational_total",
                observed_relational_count == expectations.record_count,
                f"expected={expectations.record_count} actual={observed_relational_count}",
            ),
            _check(
                "coverage_embedding_total",
                observed_embedding_count == expectations.record_count,
                f"expected={expectations.record_count} actual={observed_embedding_count}",
            ),
        )
    )
    passed = all(check.status is ValidationStatus.PASS for check in checks)
    summary = GlobalCoverageSummary(
        expected_record_count=expectations.record_count,
        observed_relational_count=observed_relational_count,
        observed_embedding_count=observed_embedding_count,
        shard_count=expectations.shard_count,
        first_gap_ordinal=first_gap,
        first_overlap_ordinal=first_overlap,
        final_end_index=final_end,
        passed=passed and first_gap is None and first_overlap is None,
    )
    return summary, tuple(checks)


class DuplicatePartitionWriter:
    def __init__(self, scratch_root: Path, *, partition_count: int = 256) -> None:
        self._scratch_root = scratch_root
        self._partition_count = partition_count
        self._scratch_root.mkdir(parents=True, exist_ok=True)

    def write_line(self, category: str, value: str) -> None:
        category_dir = self._scratch_root / category
        category_dir.mkdir(parents=True, exist_ok=True)
        partition = self._partition_index(value)
        target = category_dir / f"part-{partition:04d}.txt"
        with target.open("a", encoding="utf-8") as handle:
            handle.write(value)
            handle.write("\n")

    def count_duplicates(self, category: str) -> int:
        category_dir = self._scratch_root / category
        if not category_dir.is_dir():
            return 0
        duplicate_count = 0
        for partition_file in sorted(category_dir.glob("part-*.txt")):
            lines = partition_file.read_text(encoding="utf-8").splitlines()
            lines.sort()
            for index in range(1, len(lines)):
                if lines[index] == lines[index - 1]:
                    duplicate_count += 1
        return duplicate_count

    def _partition_index(self, value: str) -> int:
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
        return int(digest[:8], 16) % self._partition_count


def finalize_duplicate_validation(writer: DuplicatePartitionWriter) -> DuplicateValidationSummary:
    global_dupes = writer.count_duplicates("global_row_index")
    source_ref_dupes = writer.count_duplicates("source_ref")
    point_id_dupes = writer.count_duplicates("logical_point_id")
    return DuplicateValidationSummary(
        duplicate_global_row_index_count=global_dupes,
        duplicate_source_ref_count=source_ref_dupes,
        duplicate_logical_point_id_count=point_id_dupes,
        passed=global_dupes == 0 and source_ref_dupes == 0 and point_id_dupes == 0,
    )


def validate_checksums_extended(paths: DataPackPaths) -> tuple[ValidationCheck, ...]:
    checksums_path = paths.checksums_file
    if not checksums_path.is_file():
        return (_check("checksum_file_present", False, str(checksums_path)),)
    checks: list[ValidationCheck] = []
    seen_paths: set[str] = set()
    duplicate_paths = False
    invalid_lines: list[str] = []
    mismatches: list[str] = []
    missing_targets: list[str] = []
    for line in checksums_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            invalid_lines.append(line)
            continue
        expected_checksum, relative_name = parts[0], parts[1].strip()
        if relative_name in seen_paths:
            duplicate_paths = True
        seen_paths.add(relative_name)
        if len(expected_checksum) != 64:
            invalid_lines.append(line)
            continue
        target = paths.root / relative_name
        if not target.is_file():
            missing_targets.append(relative_name)
            continue
        try:
            actual = sha256_file(target)
        except OSError:
            missing_targets.append(relative_name)
            continue
        if actual != expected_checksum:
            mismatches.append(relative_name)
    checks.extend(
        (
            _check("checksum_format_valid", not invalid_lines, f"invalid={len(invalid_lines)}"),
            _check("checksum_no_duplicate_paths", not duplicate_paths, "duplicate checksum paths"),
            _check("checksum_no_missing_targets", not missing_targets, f"missing={len(missing_targets)}"),
            _check("checksum_hash_matches", not mismatches, f"mismatches={len(mismatches)}"),
        )
    )
    if not missing_targets and not mismatches and not invalid_lines and not duplicate_paths:
        try:
            verify_sha256sums(checksums_path, paths.root)
            checks.append(_check("checksum_verify_sha256sums", True, "verified"))
        except VpiDataPackIntegrityError as exc:
            checks.append(_check("checksum_verify_sha256sums", False, str(exc)))
    else:
        checks.append(_check("checksum_verify_sha256sums", False, "extended checksum checks failed"))
    return tuple(checks)


def validate_build_state_consistency(
    build_state: DataPackBuildState,
    *,
    paths: DataPackPaths,
    manifest: DataPackManifest,
    shard_index: ShardIndex,
    expectations: DataPackValidationExpectations,
) -> tuple[ValidationCheck, ...]:
    checks: list[ValidationCheck] = [
        _check(
            "build_state_expected_record_count",
            build_state.expected_record_count == expectations.record_count,
            f"expected={expectations.record_count} actual={build_state.expected_record_count}",
        ),
        _check(
            "build_state_shard_size",
            build_state.shard_size == expectations.shard_size,
            f"expected={expectations.shard_size} actual={build_state.shard_size}",
        ),
        _check(
            "build_state_completed_shards",
            build_state.completed_shards == expectations.shard_count,
            f"expected={expectations.shard_count} actual={build_state.completed_shards}",
        ),
        _check(
            "build_state_content_identity",
            build_state.content_identity == manifest.content_identity,
            build_state.content_identity,
        ),
    ]
    relational_by_ordinal = {
        descriptor.ordinal: descriptor for descriptor in shard_index.relational_shards
    }
    embedding_by_ordinal = {
        descriptor.ordinal: descriptor for descriptor in shard_index.embedding_shards
    }
    for shard in build_state.shards:
        ordinal = shard.ordinal
        relational_descriptor = relational_by_ordinal.get(ordinal)
        embedding_descriptor = embedding_by_ordinal.get(ordinal)
        ready = shard.status is DataPackShardStatus.READY
        checks.append(
            _check(
                f"build_state_shard_{ordinal}_ready",
                ready,
                shard.status.value,
            )
        )
        if not ready:
            continue
        expected_records = expectations.expected_shard_record_count(ordinal)
        checks.extend(
            (
                _check(
                    f"build_state_shard_{ordinal}_records_processed",
                    shard.records_processed == expected_records,
                    f"expected={expected_records} actual={shard.records_processed}",
                ),
                _check(
                    f"build_state_shard_{ordinal}_embedding_count",
                    shard.embedding_count == expected_records,
                    f"expected={expected_records} actual={shard.embedding_count}",
                ),
            )
        )
        if relational_descriptor is not None and shard.relational_relative_path is not None:
            checks.append(
                _check(
                    f"build_state_shard_{ordinal}_relational_path",
                    shard.relational_relative_path == relational_descriptor.relative_path,
                    shard.relational_relative_path,
                )
            )
            relational_path = paths.root / shard.relational_relative_path
            if relational_path.is_file() and shard.relational_sha256 is not None:
                checks.append(
                    _check(
                        f"build_state_shard_{ordinal}_relational_sha256",
                        shard.relational_sha256 == sha256_file(relational_path),
                        shard.relational_relative_path,
                    )
                )
        if embedding_descriptor is not None and shard.embedding_relative_path is not None:
            checks.append(
                _check(
                    f"build_state_shard_{ordinal}_embedding_path",
                    shard.embedding_relative_path == embedding_descriptor.relative_path,
                    shard.embedding_relative_path,
                )
            )
            embedding_path = paths.root / shard.embedding_relative_path
            if embedding_path.is_file() and shard.embedding_sha256 is not None:
                checks.append(
                    _check(
                        f"build_state_shard_{ordinal}_embedding_sha256",
                        shard.embedding_sha256 == sha256_file(embedding_path),
                        shard.embedding_relative_path,
                    )
                )
    return tuple(checks)


def validate_finalization_semantics(
    *,
    paths: DataPackPaths,
    manifest: DataPackManifest,
    build_state: DataPackBuildState,
    expectations: DataPackValidationExpectations,
    prior_phases_passed: bool,
) -> tuple[ValidationCheck, ...]:
    proof_report_ok = (
        paths.proof_report_file.is_file()
        if expectations.require_proof_report
        else True
    )
    ready_shards = all(
        shard.status is DataPackShardStatus.READY for shard in build_state.shards
    )
    checks = (
        _check("finalization_manifest_exists", paths.manifest_file.is_file(), "manifest"),
        _check("finalization_shard_index_exists", paths.shards_index_file.is_file(), "shards.json"),
        _check("finalization_checksums_exist", paths.checksums_file.is_file(), "SHA256SUMS"),
        _check(
            "finalization_proof_report_exists",
            proof_report_ok,
            str(paths.proof_report_file),
        ),
        _check("finalization_all_shards_ready", ready_shards, "all shards READY"),
        _check(
            "finalization_record_count",
            manifest.record_count == expectations.record_count,
            str(manifest.record_count),
        ),
        _check(
            "finalization_manifest_status_ready",
            manifest.status.value == "READY",
            manifest.status.value,
        ),
        _check(
            "finalization_prior_phases_passed",
            prior_phases_passed,
            "all mandatory phases passed",
        ),
        _check(
            "finalization_artifact_valid",
            prior_phases_passed
            and ready_shards
            and manifest.record_count == expectations.record_count
            and proof_report_ok,
            "FINALIZED_ARTIFACT_VALID",
        ),
    )
    return checks


def boundary_from_metrics(ordinal: int, metrics: ShardValidationMetrics) -> ShardBoundary:
    return ShardBoundary(
        ordinal=ordinal,
        start_row_index=metrics.global_row_index_start,
        end_row_index_exclusive=metrics.global_row_index_end_exclusive,
    )


def cleanup_scratch(scratch_root: Path) -> None:
    if scratch_root.is_dir():
        shutil.rmtree(scratch_root)
