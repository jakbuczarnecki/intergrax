"""Unit tests for legacy PlatformVectorSearchAdapter wrapper."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from intergrax.knowledge.contracts.document import KnowledgeDocument
from intergrax.rag.embedding.contracts.embedding_provider import EmbeddingProvider
from intergrax.rag.embedding.registry.execution_config import EmbeddingProviderExecutionConfig
from intergrax.rag.embedding.registry.profile import EmbeddingProfile
from intergrax.rag.vectorstore.contracts.native_vectorstore import VectorStoreHit, VectorStoreScope

from platform_proofs.scenarios.verified_product_identification.application.config.embedding_configuration import (
    VpiEmbeddingConfiguration,
)
from platform_proofs.scenarios.verified_product_identification.application.config.embedding_execution_configuration import (
    VpiEmbeddingProviderExecutionConfiguration,
)
from platform_proofs.scenarios.verified_product_identification.application.contracts.queries import (
    VectorSearchQuery,
)
from platform_proofs.scenarios.verified_product_identification.integrations.embedding.bootstrap import (
    ensure_embedding_provider_integrations_registered,
)
from platform_proofs.scenarios.verified_product_identification.integrations.embedding.intergrax_adapter import (
    IntergraxEmbeddingBootstrapAdapter,
)
from platform_proofs.scenarios.verified_product_identification.integrations.search_store.platform_vector_search_adapter import (
    PlatformVectorSearchAdapter,
)
from platform_proofs.scenarios.verified_product_identification.integrations.search_store.qdrant_vector_candidate_search_adapter import (
    QdrantVectorCandidateSearchAdapter,
)
from tests.unit.platform_proofs.scenarios.verified_product_identification.test_qdrant_vector_candidate_search import (
    _compatibility_gate,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[5]
_LEGACY_ADAPTER_PATH = (
    _REPO_ROOT
    / "platform_proofs/scenarios/verified_product_identification/integrations/search_store/platform_vector_search_adapter.py"
)


class _FakeEmbeddingProvider(EmbeddingProvider):
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def provider_name(self) -> str:
        return "hf"

    def dimension(self) -> int:
        return 4

    def embed(self, texts: list[str]) -> np.ndarray:
        self.calls.append(list(texts))
        return np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)


class _FakeVectorStore:
    def __init__(self) -> None:
        self.last_query_vector: np.ndarray | None = None

    def add_records(self, records, *, scope: VectorStoreScope):
        return None

    def query(
        self,
        query_embedding,
        *,
        scope: VectorStoreScope,
        top_k: int,
        metadata_filter=None,
        include_embeddings: bool = False,
    ):
        self.last_query_vector = np.asarray(query_embedding, dtype=np.float32)
        document = KnowledgeDocument.model_validate(
            {
                "schema_version": 1,
                "identity": {
                    "document_id": "point-1",
                    "root_document_id": "point-1",
                },
                "scope": {"tenant_id": scope.tenant_id},
                "content": "relay module",
                "metadata": {
                    "offer_id": "offer-1",
                    "catalog_id": "wdc-v2-selected",
                },
                "provenance": {
                    "source_kind": "vpi_bootstrap",
                    "source_id": "offer-1",
                    "provider_id": "hf",
                },
            }
        )
        return (
            VectorStoreHit(
                vector_id="point-1",
                document=document,
                similarity_score=0.99,
                rank=0,
            ),
        )

    def delete(self, ids, *, scope: VectorStoreScope) -> None:
        return None

    def count(self, *, scope: VectorStoreScope) -> int:
        return 1


def _configuration() -> VpiEmbeddingConfiguration:
    return VpiEmbeddingConfiguration(
        profile=EmbeddingProfile(provider="hf", model="BAAI/bge-m3"),
        expected_dimension=4,
    )


def _execution_configuration() -> VpiEmbeddingProviderExecutionConfiguration:
    return VpiEmbeddingProviderExecutionConfiguration(
        execution=EmbeddingProviderExecutionConfig(device="cpu", batch_size=1),
    )


def test_legacy_platform_vector_search_adapter_delegates_to_canonical_adapter() -> None:
    ensure_embedding_provider_integrations_registered()
    provider = _FakeEmbeddingProvider()
    vector_store = _FakeVectorStore()
    embedding = IntergraxEmbeddingBootstrapAdapter(
        _configuration(),
        provider=provider,
        execution_configuration=_execution_configuration(),
    )
    delegate = QdrantVectorCandidateSearchAdapter.from_dependencies(
        vector_store=vector_store,
        scope=VectorStoreScope(tenant_id="default"),
        embedding=embedding,
        embedding_configuration=_configuration(),
        catalog_scope_id="wdc-v2-selected",
        compatibility_gate=_compatibility_gate(),
    )
    adapter = PlatformVectorSearchAdapter(_delegate=delegate)
    result = adapter.search(VectorSearchQuery(query_text="relay 24V", limit=3))
    assert provider.calls == [["relay 24V"]]
    assert vector_store.last_query_vector is not None
    assert result.candidates[0].offer_id.value == "offer-1"
    assert result.candidates[0].channel_score.cosine_similarity == 0.99


def test_legacy_platform_vector_search_adapter_is_marked_reference_only() -> None:
    source = _LEGACY_ADAPTER_PATH.read_text(encoding="utf-8")
    assert "LEGACY / REFERENCE ONLY" in source
