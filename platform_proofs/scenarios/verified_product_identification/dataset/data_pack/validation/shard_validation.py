"""Per-shard relational, embedding, and cross-artifact validation."""

from __future__ import annotations

import math
from dataclasses import dataclass

from platform_proofs.scenarios.verified_product_identification.application.domain.wdc_source_offer import (
    parse_wdc_source_offer_json,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.embedding import (
    EmbeddingDataPackRecord,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.identity import (
    semantic_text_hash,
    source_ref_key,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.relational import (
    RelationalDataPackRecord,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.validation.plan import (
    DataPackValidationExpectations,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.results import (
    ValidationCheck,
    ValidationStatus,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.manifest.deterministic_ids import (
    search_representation_point_id,
)


@dataclass(frozen=True, slots=True)
class ShardValidationMetrics:
    relational_record_count: int
    embedding_record_count: int
    global_row_index_start: int
    global_row_index_end_exclusive: int
    non_finite_vector_count: int
    zero_vector_count: int
    semantic_hash_mismatch_count: int


def _check(name: str, passed: bool, detail: str) -> ValidationCheck:
    return ValidationCheck(
        name=name,
        status=ValidationStatus.PASS if passed else ValidationStatus.FAIL,
        detail=detail,
    )


def _source_ref_key_text(source_ref_key_value: tuple[str, str, str | None]) -> str:
    catalog_id, offer_id, source_revision = source_ref_key_value
    revision = source_revision if source_revision is not None else ""
    return f"{catalog_id}|{offer_id}|{revision}"


def validate_relational_shard_records(
    records: tuple[RelationalDataPackRecord, ...],
    *,
    ordinal: int,
    expectations: DataPackValidationExpectations,
) -> tuple[ValidationCheck, ...]:
    expected_count = expectations.expected_shard_record_count(ordinal)
    expected_start, expected_end = expectations.expected_global_row_range(ordinal)
    checks: list[ValidationCheck] = [
        _check(
            f"relational_shard_{ordinal}_record_count",
            len(records) == expected_count,
            f"expected={expected_count} actual={len(records)}",
        )
    ]
    if not records:
        return tuple(checks)

    row_indices = [record.global_row_index for record in records]
    checks.append(
        _check(
            f"relational_shard_{ordinal}_global_row_index_unique",
            len(set(row_indices)) == len(row_indices),
            "duplicate global_row_index in shard",
        )
    )
    min_index = min(row_indices)
    max_index = max(row_indices)
    checks.append(
        _check(
            f"relational_shard_{ordinal}_global_row_index_range",
            min_index >= expected_start and max_index < expected_end,
            f"expected=[{expected_start},{expected_end}) actual=[{min_index},{max_index}]",
        )
    )
    ascending = all(
        records[index].global_row_index < records[index + 1].global_row_index
        for index in range(len(records) - 1)
    )
    checks.append(
        _check(
            f"relational_shard_{ordinal}_global_row_index_order",
            ascending,
            "global_row_index must be strictly ascending",
        )
    )

    refs: list[tuple[str, str, str | None]] = []
    for record in records:
        refs.append(source_ref_key(record.source_ref))
        record_json_ok = bool(record.record_json.strip())
        checks.append(
            _check(
                f"relational_shard_{ordinal}_record_json_{record.global_row_index}",
                record_json_ok,
                "record_json must be non-empty",
            )
        )
        if record_json_ok:
            try:
                parse_wdc_source_offer_json(record.record_json)
                parse_ok = True
            except ValueError:
                parse_ok = False
            checks.append(
                _check(
                    f"relational_shard_{ordinal}_record_json_parse_{record.global_row_index}",
                    parse_ok,
                    "record_json must parse as WDC source offer",
                )
            )
        derivation_ok = bool(record.derivation_version.strip())
        semantic_text_ok = bool(record.semantic_text.strip())
        expected_hash = semantic_text_hash(record.semantic_text)
        hash_ok = record.semantic_text_hash == expected_hash
        checks.extend(
            (
                _check(
                    f"relational_shard_{ordinal}_derivation_{record.global_row_index}",
                    derivation_ok,
                    "derivation_version required",
                ),
                _check(
                    f"relational_shard_{ordinal}_semantic_text_{record.global_row_index}",
                    semantic_text_ok,
                    "semantic_text required",
                ),
                _check(
                    f"relational_shard_{ordinal}_semantic_hash_{record.global_row_index}",
                    hash_ok,
                    "semantic_text_hash mismatch",
                ),
            )
        )
    duplicate_refs = len(set(refs)) != len(refs)
    checks.append(
        _check(
            f"relational_shard_{ordinal}_source_ref_unique",
            not duplicate_refs,
            "duplicate source_ref in shard",
        )
    )
    return tuple(checks)


def validate_embedding_shard_records(
    records: tuple[EmbeddingDataPackRecord, ...],
    *,
    ordinal: int,
    expectations: DataPackValidationExpectations,
) -> tuple[ValidationCheck, ...]:
    expected_count = expectations.expected_shard_record_count(ordinal)
    checks: list[ValidationCheck] = [
        _check(
            f"embedding_shard_{ordinal}_record_count",
            len(records) == expected_count,
            f"expected={expected_count} actual={len(records)}",
        )
    ]
    if not records:
        return tuple(checks)

    point_ids: list[str] = []
    refs: list[tuple[str, str, str | None]] = []
    for record in records:
        point_ids.append(record.logical_point_id)
        refs.append(source_ref_key(record.source_ref))
        provider_ok = record.embedding_provider == expectations.embedding_provider
        model_ok = record.embedding_model == expectations.embedding_model
        revision_ok = record.embedding_model_revision == expectations.embedding_model_revision
        dimension_ok = record.embedding_dimension == expectations.embedding_dimension
        vector_length_ok = len(record.dense_embedding) == expectations.embedding_dimension
        finite_ok = all(math.isfinite(value) for value in record.dense_embedding)
        norm = math.sqrt(sum(value * value for value in record.dense_embedding))
        norm_ok = norm > 0.0
        checks.extend(
            (
                _check(
                    f"embedding_shard_{ordinal}_logical_point_id_{record.logical_point_id}",
                    bool(record.logical_point_id.strip()),
                    "logical_point_id required",
                ),
                _check(
                    f"embedding_shard_{ordinal}_provider_{record.logical_point_id}",
                    provider_ok,
                    f"expected={expectations.embedding_provider}",
                ),
                _check(
                    f"embedding_shard_{ordinal}_model_{record.logical_point_id}",
                    model_ok,
                    f"expected={expectations.embedding_model}",
                ),
                _check(
                    f"embedding_shard_{ordinal}_revision_{record.logical_point_id}",
                    revision_ok,
                    f"expected={expectations.embedding_model_revision}",
                ),
                _check(
                    f"embedding_shard_{ordinal}_dimension_{record.logical_point_id}",
                    dimension_ok and vector_length_ok,
                    f"expected={expectations.embedding_dimension}",
                ),
                _check(
                    f"embedding_shard_{ordinal}_finite_{record.logical_point_id}",
                    finite_ok,
                    "vector must contain only finite values",
                ),
                _check(
                    f"embedding_shard_{ordinal}_norm_{record.logical_point_id}",
                    norm_ok,
                    "vector norm must be > 0",
                ),
            )
        )
    checks.extend(
        (
            _check(
                f"embedding_shard_{ordinal}_logical_point_id_unique",
                len(set(point_ids)) == len(point_ids),
                "duplicate logical_point_id in shard",
            ),
            _check(
                f"embedding_shard_{ordinal}_source_ref_unique",
                len(set(refs)) == len(refs),
                "duplicate source_ref in shard",
            ),
        )
    )
    return tuple(checks)


def validate_cross_shard_identity(
    relational_records: tuple[RelationalDataPackRecord, ...],
    embedding_records: tuple[EmbeddingDataPackRecord, ...],
    *,
    ordinal: int,
) -> tuple[ValidationCheck, ...]:
    relational_refs = {source_ref_key(record.source_ref) for record in relational_records}
    embedding_refs = {source_ref_key(record.source_ref) for record in embedding_records}
    checks: list[ValidationCheck] = [
        _check(
            f"cross_shard_{ordinal}_source_ref_equality",
            relational_refs == embedding_refs,
            f"relational={len(relational_refs)} embedding={len(embedding_refs)}",
        )
    ]
    embedding_by_ref = {source_ref_key(record.source_ref): record for record in embedding_records}
    mismatches = 0
    for relational_record in relational_records:
        ref = source_ref_key(relational_record.source_ref)
        embedding_record = embedding_by_ref.get(ref)
        if embedding_record is None:
            mismatches += 1
            continue
        derivation_ok = relational_record.derivation_version == embedding_record.derivation_version
        expected_hash = semantic_text_hash(relational_record.semantic_text)
        hash_ok = (
            relational_record.semantic_text_hash == expected_hash
            and embedding_record.semantic_text_hash == expected_hash
        )
        expected_point_id = search_representation_point_id(
            catalog_id=relational_record.source_ref.catalog_id,
            offer_id=relational_record.source_ref.offer_id.value,
            derivation_version=relational_record.derivation_version,
        )
        point_id_ok = embedding_record.logical_point_id == expected_point_id
        if not derivation_ok or not hash_ok or not point_id_ok:
            mismatches += 1
        checks.extend(
            (
                _check(
                    f"cross_shard_{ordinal}_derivation_{ref[1]}",
                    derivation_ok,
                    "derivation_version mismatch",
                ),
                _check(
                    f"cross_shard_{ordinal}_semantic_hash_{ref[1]}",
                    hash_ok,
                    "semantic_text_hash mismatch",
                ),
                _check(
                    f"cross_shard_{ordinal}_logical_point_id_{ref[1]}",
                    point_id_ok,
                    f"expected={expected_point_id}",
                ),
            )
        )
    checks.append(
        _check(
            f"cross_shard_{ordinal}_orphan_count",
            mismatches == 0,
            f"mismatches={mismatches}",
        )
    )
    return tuple(checks)


def collect_shard_metrics(
    relational_records: tuple[RelationalDataPackRecord, ...],
    embedding_records: tuple[EmbeddingDataPackRecord, ...],
    *,
    ordinal: int,
    expectations: DataPackValidationExpectations,
) -> ShardValidationMetrics:
    expected_start, expected_end = expectations.expected_global_row_range(ordinal)
    non_finite = 0
    zero_vectors = 0
    semantic_mismatches = 0
    embedding_by_ref = {source_ref_key(record.source_ref): record for record in embedding_records}
    for relational_record in relational_records:
        ref = source_ref_key(relational_record.source_ref)
        embedding_record = embedding_by_ref.get(ref)
        expected_hash = semantic_text_hash(relational_record.semantic_text)
        if (
            relational_record.semantic_text_hash != expected_hash
            or embedding_record is None
            or embedding_record.semantic_text_hash != expected_hash
        ):
            semantic_mismatches += 1
    for record in embedding_records:
        if not all(math.isfinite(value) for value in record.dense_embedding):
            non_finite += 1
        norm = math.sqrt(sum(value * value for value in record.dense_embedding))
        if norm == 0.0:
            zero_vectors += 1
    if relational_records:
        start = min(record.global_row_index for record in relational_records)
        end_exclusive = max(record.global_row_index for record in relational_records) + 1
    else:
        start = expected_start
        end_exclusive = expected_start
    return ShardValidationMetrics(
        relational_record_count=len(relational_records),
        embedding_record_count=len(embedding_records),
        global_row_index_start=start,
        global_row_index_end_exclusive=end_exclusive,
        non_finite_vector_count=non_finite,
        zero_vector_count=zero_vectors,
        semantic_hash_mismatch_count=semantic_mismatches,
    )


def identity_lines_for_global_validation(
    relational_records: tuple[RelationalDataPackRecord, ...],
    embedding_records: tuple[EmbeddingDataPackRecord, ...],
) -> tuple[tuple[str, str, str], ...]:
    lines: list[tuple[str, str, str]] = []
    for record in relational_records:
        lines.append(
            (
                "global_row_index",
                str(record.global_row_index),
                str(record.global_row_index),
            )
        )
        lines.append(
            (
                "source_ref",
                _source_ref_key_text(source_ref_key(record.source_ref)),
                _source_ref_key_text(source_ref_key(record.source_ref)),
            )
        )
    for record in embedding_records:
        lines.append(
            (
                "logical_point_id",
                record.logical_point_id,
                record.logical_point_id,
            )
        )
    return tuple(lines)
