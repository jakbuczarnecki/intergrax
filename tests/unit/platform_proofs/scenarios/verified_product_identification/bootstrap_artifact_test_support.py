"""Shared fakes for storage bootstrap artifact-input unit tests."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from platform_proofs.scenarios.verified_product_identification.application.config.embedding_configuration import (
    EMBEDDING_CONFIGURATION_VERSION,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.search_representation import (
    SEARCH_REPRESENTATION_DERIVATION_VERSION,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.artifact.record import (
    EmbeddingArtifactRecord,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.manifest.compatibility import (
    EmbeddingArtifactCompatibilityIdentity,
    assert_manifest_compatible,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.manifest.model import (
    EMBEDDING_ARTIFACT_SCHEMA_VERSION,
    EmbeddingArtifactManifest,
    EmbeddingArtifactShardDescriptor,
    EmbeddingArtifactState,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.results import (
    ValidationCheck,
    ValidationReport,
    ValidationStatus,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.manifest.deterministic_ids import (
    search_representation_point_id,
)


def artifact_record_for_wdc_row(
    row_index: int,
    *,
    catalog_id: str = "wdc-v2-selected",
    source_revision: str | None = None,
    dimension: int = 8,
    offer_id: str | None = None,
    derivation_version: str = SEARCH_REPRESENTATION_DERIVATION_VERSION,
) -> EmbeddingArtifactRecord:
    resolved_offer_id = offer_id if offer_id is not None else str(1000 + row_index)
    vector = tuple(0.1 * (index + 1) for index in range(dimension))
    return EmbeddingArtifactRecord(
        global_row_index=row_index,
        logical_point_id=search_representation_point_id(
            catalog_id=catalog_id,
            offer_id=resolved_offer_id,
            derivation_version=derivation_version,
        ),
        catalog_id=catalog_id,
        offer_id=resolved_offer_id,
        source_revision=source_revision,
        derivation_version=derivation_version,
        semantic_text=f"semantic {row_index}",
        lexical_text=f"lexical {row_index}",
        embedding_provider="hf",
        embedding_model="fake-model",
        embedding_dimension=dimension,
        dense_embedding=vector,
    )


def build_ready_manifest(
    records: tuple[EmbeddingArtifactRecord, ...],
    *,
    dataset_checksum: str = "testchecksum",
    dataset_record_count: int,
    shard_size: int | None = None,
    **overrides: object,
) -> EmbeddingArtifactManifest:
    effective_shard_size = shard_size or len(records)
    committed_shards: list[EmbeddingArtifactShardDescriptor] = []
    for shard_ordinal, start in enumerate(
        range(0, len(records), effective_shard_size)
    ):
        shard_records = records[start : start + effective_shard_size]
        committed_shards.append(
            EmbeddingArtifactShardDescriptor(
                shard_ordinal=shard_ordinal,
                file_name=f"shard_{shard_ordinal:05d}.parquet",
                first_global_row_index=shard_records[0].global_row_index,
                last_global_row_index=shard_records[-1].global_row_index,
                record_count=len(shard_records),
                sha256_checksum=f"checksum-{shard_ordinal}",
            )
        )
    base = EmbeddingArtifactManifest(
        state=EmbeddingArtifactState.READY,
        artifact_schema_version=EMBEDDING_ARTIFACT_SCHEMA_VERSION,
        dataset_path="/data/selected_offers.parquet",
        dataset_checksum=dataset_checksum,
        dataset_record_count=dataset_record_count,
        search_representation_derivation_version=SEARCH_REPRESENTATION_DERIVATION_VERSION,
        embedding_configuration_version=EMBEDDING_CONFIGURATION_VERSION,
        embedding_provider="hf",
        embedding_model="fake-model",
        embedding_dimension=8,
        catalog_id="wdc-v2-selected",
        source_revision=None,
        checkpoint_shard_ordinal=committed_shards[-1].shard_ordinal if committed_shards else None,
        checkpoint_rows_materialized=len(records),
        target_max_records=None,
        total_artifact_record_count=len(records),
        shard_count=len(committed_shards),
        committed_shards=tuple(committed_shards),
    )
    if not overrides:
        return base
    from dataclasses import fields

    values = {item.name: getattr(base, item.name) for item in fields(base)}
    values.update(overrides)
    return EmbeddingArtifactManifest(**values)


@dataclass
class FakeArtifactReader:
    manifest: EmbeddingArtifactManifest
    shard_records: dict[int, tuple[EmbeddingArtifactRecord, ...]] = field(
        default_factory=dict
    )
    identity_should_pass: bool = True

    def read_manifest(self) -> EmbeddingArtifactManifest:
        return self.manifest

    def iterate_shard_records(
        self,
        descriptor: EmbeddingArtifactShardDescriptor,
    ) -> Iterator[EmbeddingArtifactRecord]:
        return iter(self.shard_records[descriptor.shard_ordinal])

    def validate_identity(
        self,
        expected: EmbeddingArtifactCompatibilityIdentity,
    ) -> ValidationReport:
        if not self.identity_should_pass:
            return ValidationReport.from_checks(
                (
                    ValidationCheck(
                        "artifact_identity",
                        ValidationStatus.FAIL,
                        "injected identity failure",
                    ),
                )
            )
        try:
            assert_manifest_compatible(existing=self.manifest, expected=expected)
            return ValidationReport.from_checks(
                (
                    ValidationCheck(
                        "artifact_identity",
                        ValidationStatus.PASS,
                        "ok",
                    ),
                )
            )
        except Exception as exc:
            return ValidationReport.from_checks(
                (
                    ValidationCheck(
                        "artifact_identity",
                        ValidationStatus.FAIL,
                        str(exc),
                    ),
                )
            )

    def close(self) -> None:
        return None


def reader_from_records(
    records: tuple[EmbeddingArtifactRecord, ...],
    *,
    dataset_record_count: int,
    shard_size: int | None = None,
    **manifest_overrides: object,
) -> FakeArtifactReader:
    manifest = build_ready_manifest(
        records,
        dataset_record_count=dataset_record_count,
        shard_size=shard_size,
        **manifest_overrides,
    )
    effective_shard_size = shard_size or len(records)
    shard_records: dict[int, tuple[EmbeddingArtifactRecord, ...]] = {}
    for shard_ordinal, start in enumerate(range(0, len(records), effective_shard_size)):
        shard_records[shard_ordinal] = records[start : start + effective_shard_size]
    return FakeArtifactReader(manifest=manifest, shard_records=shard_records)
