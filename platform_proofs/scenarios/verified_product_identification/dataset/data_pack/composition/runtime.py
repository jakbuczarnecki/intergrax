"""Composition root for proof-50 storage and retrieval."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.integrations.contracts.vector_index_administration import VectorIndexIdentity
from intergrax.integrations.providers.vector_store.qdrant.config import QdrantIntegrationConfig
from intergrax.integrations.providers.vector_store.qdrant.opens import (
    open_qdrant_vector_index_administration,
    open_qdrant_vector_store,
)

from platform_proofs.scenarios.verified_product_identification.application.config.embedding_configuration import (
    EMBEDDING_CONFIGURATION_VERSION,
    load_vpi_embedding_configuration,
)
from platform_proofs.scenarios.verified_product_identification.application.config.embedding_execution_configuration import (
    load_vpi_embedding_provider_execution_configuration,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.search_representation import (
    SEARCH_REPRESENTATION_DERIVATION_VERSION,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.paths import (
    PROOF_50_POSTGRESQL_SCHEMA,
    PROOF_50_QDRANT_COLLECTION,
)
from platform_proofs.scenarios.verified_product_identification.integrations.catalog_store.postgresql.adapter import (
    PostgreSQLCatalogBootstrapAdapter,
)
from platform_proofs.scenarios.verified_product_identification.integrations.catalog_store.postgresql.catalog_search_adapter import (
    PostgreSQLExactIdentifierLookupAdapter,
    PostgreSQLLexicalSearchAdapter,
    PostgreSQLSourceRecordFetchAdapter,
    PostgreSQLStructuredSearchAdapter,
)
from platform_proofs.scenarios.verified_product_identification.integrations.search_store.platform_bootstrap_adapter import (
    PlatformSearchIndexBootstrapAdapter,
)
from platform_proofs.scenarios.verified_product_identification.integrations.search_store.platform_vector_search_adapter import (
    PlatformVectorSearchAdapter,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.manifest.model import (
    BOOTSTRAP_IMPLEMENTATION_VERSION,
    CATALOG_SCHEMA_VERSION,
    BootstrapState,
    SEARCH_INDEX_SCHEMA_VERSION,
    VpiBootstrapManifest,
)

_DENSE_CHANNEL_NAME = "dense"
_SPARSE_CHANNEL_NAME = "sparse"
_PROOF_CATALOG_ID = "wdc-v2-selected"


@dataclass(slots=True)
class ProofStorageRuntime:
    catalog: PostgreSQLCatalogBootstrapAdapter
    search: PlatformSearchIndexBootstrapAdapter
    bootstrap_manifest: VpiBootstrapManifest

    def close(self) -> None:
        self.catalog.close()
        self.search.close()


@dataclass(slots=True)
class ProofSearchRuntime:
    exact_lookup: PostgreSQLExactIdentifierLookupAdapter
    lexical_search: PostgreSQLLexicalSearchAdapter
    structured_search: PostgreSQLStructuredSearchAdapter
    vector_search: PlatformVectorSearchAdapter
    source_fetch: PostgreSQLSourceRecordFetchAdapter

    def close(self) -> None:
        self.exact_lookup.close()
        self.lexical_search.close()
        self.structured_search.close()
        self.vector_search.close()
        self.source_fetch.close()


def _build_bootstrap_manifest() -> VpiBootstrapManifest:
    embedding = load_vpi_embedding_configuration()
    model = embedding.model
    if model is None:
        msg = "embedding model is required"
        raise ValueError(msg)
    return VpiBootstrapManifest(
        state=BootstrapState.INGESTING,
        dataset_path="data_pack/proof-50",
        dataset_checksum="proof-50-data-pack",
        dataset_record_count=50,
        search_representation_derivation_version=SEARCH_REPRESENTATION_DERIVATION_VERSION,
        embedding_configuration_version=EMBEDDING_CONFIGURATION_VERSION,
        embedding_provider=embedding.provider,
        embedding_model=model,
        embedding_dimension=embedding.expected_dimension,
        catalog_schema_version=CATALOG_SCHEMA_VERSION,
        search_index_schema_version=SEARCH_INDEX_SCHEMA_VERSION,
        bootstrap_implementation_version=BOOTSTRAP_IMPLEMENTATION_VERSION,
        catalog_id=_PROOF_CATALOG_ID,
        source_revision=None,
        checkpoint_batch_ordinal=0,
        checkpoint_rows_processed=0,
        target_max_records=50,
        catalog_source_offer_count=0,
        catalog_identifier_count=0,
        catalog_structured_attribute_count=0,
        search_point_count=0,
        failure_stage=None,
        failure_detail=None,
    )


def build_proof_50_storage_runtime() -> ProofStorageRuntime:
    catalog = PostgreSQLCatalogBootstrapAdapter.from_env(
        schema_name=PROOF_50_POSTGRESQL_SCHEMA,
        ingestion_batch_label="vpi-proof-50-5c4d1",
    )
    qdrant_config = QdrantIntegrationConfig.from_env(
        collection_name=PROOF_50_QDRANT_COLLECTION,
        enable_sparse_vectors=True,
    )
    index_identity = VectorIndexIdentity(
        logical_name=PROOF_50_QDRANT_COLLECTION,
        tenant_id=qdrant_config.tenant_id,
    )
    search = PlatformSearchIndexBootstrapAdapter(
        _index_admin=open_qdrant_vector_index_administration(qdrant_config),
        _vector_store=open_qdrant_vector_store(qdrant_config),
        _index_identity=index_identity,
        _dense_channel_name=_DENSE_CHANNEL_NAME,
        _sparse_channel_name=_SPARSE_CHANNEL_NAME,
        _sparse_required=qdrant_config.enable_sparse_vectors,
    )
    return ProofStorageRuntime(
        catalog=catalog,
        search=search,
        bootstrap_manifest=_build_bootstrap_manifest(),
    )


def build_proof_50_search_runtime() -> ProofSearchRuntime:
    embedding_configuration = load_vpi_embedding_configuration()
    execution_configuration = load_vpi_embedding_provider_execution_configuration()
    return ProofSearchRuntime(
        exact_lookup=PostgreSQLExactIdentifierLookupAdapter.from_env(
            schema_name=PROOF_50_POSTGRESQL_SCHEMA,
            catalog_id=_PROOF_CATALOG_ID,
        ),
        lexical_search=PostgreSQLLexicalSearchAdapter.from_env(
            schema_name=PROOF_50_POSTGRESQL_SCHEMA,
            catalog_id=_PROOF_CATALOG_ID,
        ),
        structured_search=PostgreSQLStructuredSearchAdapter.from_env(
            schema_name=PROOF_50_POSTGRESQL_SCHEMA,
            catalog_id=_PROOF_CATALOG_ID,
        ),
        vector_search=PlatformVectorSearchAdapter.from_env(
            collection_name=PROOF_50_QDRANT_COLLECTION,
            catalog_id=_PROOF_CATALOG_ID,
            embedding_configuration=embedding_configuration,
            execution_configuration=execution_configuration,
        ),
        source_fetch=PostgreSQLSourceRecordFetchAdapter.from_env(
            schema_name=PROOF_50_POSTGRESQL_SCHEMA,
            catalog_id=_PROOF_CATALOG_ID,
        ),
    )
