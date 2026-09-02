"""Provider-neutral search-index bootstrap adapter — platform contracts only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from intergrax.integrations.contracts.vector_index_administration import (
    DenseVectorChannelSpec,
    SparseLexicalChannelSpec,
    VectorIndexAdministration,
    VectorIndexCompatibilityError,
    VectorIndexDescription,
    VectorIndexIdentity,
    VectorIndexSpec,
    VectorSearchCapability,
)
from intergrax.integrations.contracts.vector_store import VectorStore, VectorStoreRecord, VectorStoreScope
from intergrax.knowledge.contracts.document import KnowledgeDocument
from intergrax.rag.vectorstore.config.vector_config import Metric

from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.errors import (
    VpiBootstrapCompatibilityError,
    VpiBootstrapProviderError,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.ports import (
    SearchIndexIngestBatch,
    SearchIndexIngestBatchResult,
    SearchIndexIngestRecord,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.results import (
    ValidationCheck,
    ValidationReport,
    ValidationStatus,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.manifest.deterministic_ids import (
    source_ref_payload,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.manifest.model import (
    VpiBootstrapManifest,
)


def _translate_index_compatibility_error(exc: VectorIndexCompatibilityError) -> VpiBootstrapCompatibilityError:
    return VpiBootstrapCompatibilityError(str(exc))


def _build_index_spec(
    identity: VectorIndexIdentity,
    *,
    dimension: int,
    sparse_required: bool,
    dense_channel_name: str,
    sparse_channel_name: str,
    metric: Metric = "cosine",
) -> VectorIndexSpec:
    required: set[VectorSearchCapability] = {VectorSearchCapability.DENSE}
    sparse_spec: SparseLexicalChannelSpec | None = None
    if sparse_required:
        required.add(VectorSearchCapability.SPARSE_LEXICAL)
        sparse_spec = SparseLexicalChannelSpec(channel_name=sparse_channel_name)
    return VectorIndexSpec(
        identity=identity,
        dense=DenseVectorChannelSpec(
            channel_name=dense_channel_name,
            dimension=dimension,
            metric=metric,
        ),
        required_capabilities=frozenset(required),
        sparse_lexical=sparse_spec,
    )


def _to_vector_store_record(
    record: SearchIndexIngestRecord,
    *,
    tenant_id: str,
) -> VectorStoreRecord:
    metadata = {
        "channel_lexical": record.lexical_text,
        "channel_vector_source": "semantic",
        "search_representation_derivation_version": record.derivation_version,
        "dataset_checksum": record.dataset_checksum,
        "embedding_provider": record.embedding_provider,
        "embedding_model": record.embedding_model,
        "embedding_dimension": record.embedding_dimension,
        **source_ref_payload(record.source_ref),
    }
    document = KnowledgeDocument.model_validate(
        {
            "schema_version": 1,
            "identity": {
                "document_id": record.logical_point_id,
                "root_document_id": record.logical_point_id,
            },
            "scope": {"tenant_id": tenant_id},
            "content": record.lexical_text,
            "metadata": metadata,
            "provenance": {
                "source_kind": "vpi_bootstrap",
                "source_id": record.source_ref.offer_id.value,
                "provider_id": record.embedding_provider,
            },
        }
    )
    embedding = np.asarray(record.dense_embedding, dtype=np.float32)
    return VectorStoreRecord(
        document=document,
        embedding=embedding,
        vector_id=record.logical_point_id,
    )


@dataclass(slots=True)
class PlatformSearchIndexBootstrapAdapter:
    """Reference ``SearchIndexBootstrapPort`` over platform vector index contracts."""

    _index_admin: VectorIndexAdministration
    _vector_store: VectorStore
    _index_identity: VectorIndexIdentity
    _dense_channel_name: str
    _sparse_channel_name: str
    _sparse_required: bool
    _dimension: int | None = None

    def probe_readiness(self) -> ValidationReport:
        health = self._index_admin.probe()
        if not health.healthy:
            raise VpiBootstrapProviderError(
                f"Search index readiness probe failed: {health.detail}"
            )
        return ValidationReport.from_checks(
            (
                ValidationCheck(
                    name="search_index_reachable",
                    status=ValidationStatus.PASS,
                    detail=health.detail or "probe succeeded",
                ),
            )
        )

    def prepare(self, manifest: VpiBootstrapManifest) -> ValidationReport:
        self._dimension = manifest.embedding_dimension
        spec = _build_index_spec(
            self._index_identity,
            dimension=manifest.embedding_dimension,
            sparse_required=self._sparse_required,
            dense_channel_name=self._dense_channel_name,
            sparse_channel_name=self._sparse_channel_name,
        )
        try:
            result = self._index_admin.prepare_index(spec)
        except VectorIndexCompatibilityError as exc:
            raise _translate_index_compatibility_error(exc) from exc
        except Exception as exc:
            raise VpiBootstrapProviderError("search index prepare failed") from exc

        if result.outcome.value == "created":
            check_name = "search_index_created"
            detail = f"logical_name={self._index_identity.logical_name}"
        else:
            check_name = "search_index_compatible"
            detail = (
                f"logical_name={self._index_identity.logical_name} "
                f"dimension={result.description.dense_dimension} "
                f"sparse={self._sparse_required}"
            )
        return ValidationReport.from_checks(
            (
                ValidationCheck(
                    name=check_name,
                    status=ValidationStatus.PASS,
                    detail=detail,
                ),
            )
        )

    def ingest_batch(self, batch: SearchIndexIngestBatch) -> SearchIndexIngestBatchResult:
        if self._dimension is None:
            raise VpiBootstrapProviderError("search index prepare must run before ingest")

        scope = VectorStoreScope(tenant_id=self._index_identity.tenant_id)
        records = tuple(
            _to_vector_store_record(record, tenant_id=self._index_identity.tenant_id)
            for record in batch.records
        )
        try:
            self._vector_store.add_records(records, scope=scope)
        except Exception as exc:
            raise VpiBootstrapProviderError(
                f"Search index ingest failed for batch {batch.batch_ordinal}"
            ) from exc
        return SearchIndexIngestBatchResult(point_count=self.count_points())

    def validate(self, manifest: VpiBootstrapManifest) -> ValidationReport:
        from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.manifest.run_target import (
            effective_run_target_rows,
        )

        description = self._index_admin.describe_index(self._index_identity)
        expected = effective_run_target_rows(manifest)
        return ValidationReport.from_checks(
            (
                *self._persisted_index_compatibility_checks(manifest, description),
                ValidationCheck(
                    name="search_index_point_count",
                    status=ValidationStatus.PASS
                    if description.point_count >= expected
                    else ValidationStatus.FAIL,
                    detail=f"point_count={description.point_count} expected>={expected}",
                ),
            )
        )

    def count_points(self) -> int:
        description = self._index_admin.describe_index(self._index_identity)
        return description.point_count

    def close(self) -> None:
        # VectorStore contract has no close(); platform opener owns data-plane lifecycle.
        self._index_admin.close()

    def _persisted_index_compatibility_checks(
        self,
        manifest: VpiBootstrapManifest,
        description: VectorIndexDescription,
    ) -> tuple[ValidationCheck, ...]:
        if not description.exists:
            checks: list[ValidationCheck] = [
                ValidationCheck(
                    name="search_index_exists",
                    status=ValidationStatus.FAIL,
                    detail=f"logical_name={self._index_identity.logical_name} not found",
                ),
                ValidationCheck(
                    name="search_index_dense_dimension",
                    status=ValidationStatus.FAIL,
                    detail="persisted_dimension=none",
                ),
            ]
            if self._sparse_required:
                checks.append(
                    ValidationCheck(
                        name="search_index_sparse_lexical",
                        status=ValidationStatus.FAIL,
                        detail="index missing",
                    )
                )
            return tuple(checks)

        checks: list[ValidationCheck] = []
        if description.dense_dimension is None:
            checks.append(
                ValidationCheck(
                    name="search_index_dense_channel",
                    status=ValidationStatus.FAIL,
                    detail=(
                        f"logical_name={self._index_identity.logical_name} has no dense vector config"
                    ),
                )
            )
            checks.append(
                ValidationCheck(
                    name="search_index_dense_dimension",
                    status=ValidationStatus.FAIL,
                    detail="persisted_dimension=none",
                )
            )
        else:
            checks.append(
                ValidationCheck(
                    name="search_index_dense_dimension",
                    status=(
                        ValidationStatus.PASS
                        if description.dense_dimension == manifest.embedding_dimension
                        else ValidationStatus.FAIL
                    ),
                    detail=(
                        f"persisted_dimension={description.dense_dimension} "
                        f"expected={manifest.embedding_dimension}"
                    ),
                )
            )

        if self._sparse_required:
            has_sparse = VectorSearchCapability.SPARSE_LEXICAL in description.present_capabilities
            checks.append(
                ValidationCheck(
                    name="search_index_sparse_lexical",
                    status=ValidationStatus.PASS if has_sparse else ValidationStatus.FAIL,
                    detail=(
                        f"sparse_channel={'present' if has_sparse else 'missing'} "
                        f"required={self._sparse_channel_name}"
                    ),
                )
            )
        return tuple(checks)
