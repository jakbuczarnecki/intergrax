"""Unit tests for canonical Qdrant vector candidate search adapter."""

from __future__ import annotations

import ast
from collections.abc import Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np
import pytest

from intergrax.integrations.contracts.base import IntegrationDependencyError
from intergrax.integrations.contracts.vector_index_administration import VectorIndexIdentity
from intergrax.integrations.contracts.vector_store import MetadataFilter, VectorStoreScope
from intergrax.knowledge.contracts.document import KnowledgeDocument
from intergrax.rag.embedding.contracts.embedding_provider import EmbeddingProvider
from intergrax.rag.embedding.registry.execution_config import EmbeddingProviderExecutionConfig
from intergrax.rag.embedding.registry.profile import EmbeddingProfile
from intergrax.rag.vectorstore.contracts.native_vectorstore import (
    VectorStoreContractError,
    VectorStoreHit,
)

from platform_proofs.scenarios.verified_product_identification.application.config.embedding_configuration import (
    VpiEmbeddingConfiguration,
)
from platform_proofs.scenarios.verified_product_identification.application.config.embedding_execution_configuration import (
    VpiEmbeddingProviderExecutionConfiguration,
)
from platform_proofs.scenarios.verified_product_identification.application.contracts.failures import (
    CatalogSearchFailureKind,
)
from platform_proofs.scenarios.verified_product_identification.application.contracts.queries import (
    VectorSearchQuery,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.candidates import (
    RetrievalChannel,
    VectorChannelScore,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.identifiers import (
    ProductOfferId,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.source import (
    SourceRecordRef,
)
from platform_proofs.scenarios.verified_product_identification.application.ports.catalog_search import (
    VectorCandidateSearchPort,
)
from platform_proofs.scenarios.verified_product_identification.application.retrieval import (
    MultiChannelRetrievalRequest,
)
from platform_proofs.scenarios.verified_product_identification.application.retrieval.service import (
    MultiChannelRetrievalService,
)
from platform_proofs.scenarios.verified_product_identification.integrations.embedding.bootstrap import (
    ensure_embedding_provider_integrations_registered,
)
from platform_proofs.scenarios.verified_product_identification.integrations.embedding.intergrax_adapter import (
    IntergraxEmbeddingBootstrapAdapter,
)
from platform_proofs.scenarios.verified_product_identification.integrations.search_store.qdrant_vector_candidate_search_adapter import (
    QdrantVectorCandidateSearchAdapter,
)
from platform_proofs.scenarios.verified_product_identification.integrations.search_store.vector_index_compatibility import (
    VectorIndexCompatibilityGate,
)
from platform_proofs.scenarios.verified_product_identification.integrations.search_store.vector_index_expectations import (
    expected_vector_index_identity_from_bootstrap_vector_identity,
)
from platform_proofs.scenarios.verified_product_identification.integrations.search_store.vector_index_runtime_identity import (
    ExpectedVectorIndexRuntimeIdentity,
    ResolvedVectorIndexRuntimeIdentity,
    VectorIndexIdentityResolution,
    VectorIndexIdentityResolutionStatus,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.qdrant.configuration import (
    ExpectedVectorIdentity,
)
from platform_proofs.scenarios.verified_product_identification.integrations.search_store.vector_hit_identity import (
    decode_vector_hit_identity_from_storage_payload,
)
from platform_proofs.scenarios.verified_product_identification.retrieval.composition import (
    build_vector_candidate_search,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.qdrant.payload import (
    payload_from_record,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.errors import (
    VpiBootstrapProviderError,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.contracts import (
    VectorLoadRecord,
)
from tests.unit.platform_proofs.scenarios.verified_product_identification.test_catalog_contracts import (
    FakeExactLookupA,
)
from tests.unit.platform_proofs.scenarios.verified_product_identification.test_multi_channel_retrieval import (
    RecordingStructuredSearch,
    RecordingVectorSearch,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[5]
_VPI_ROOT = _REPO_ROOT / "platform_proofs/scenarios/verified_product_identification"
_ADAPTER_ROOT = _VPI_ROOT / "integrations/search_store"
_APPLICATION_ROOT = _VPI_ROOT / "application"
_LEGACY_ADAPTER_PATH = _ADAPTER_ROOT / "platform_vector_search_adapter.py"
_CATALOG_ID = "wdc-v2-selected"


class _FakeEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        *,
        dimension: int = 4,
        vectors: Sequence[Sequence[float]] | None = None,
        fail: BaseException | None = None,
    ) -> None:
        self._dimension = dimension
        self._vectors = vectors
        self._fail = fail
        self.calls: list[list[str]] = []

    def provider_name(self) -> str:
        return "hf"

    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> np.ndarray:
        self.calls.append(list(texts))
        if self._fail is not None:
            raise self._fail
        if self._vectors is not None:
            return np.asarray(self._vectors, dtype=np.float32)
        return np.asarray([[1.0, 0.0, 0.0, 0.0] for _ in texts], dtype=np.float32)


@dataclass
class _FakeHitSpec:
    offer_id: str
    catalog_id: str = _CATALOG_ID
    source_revision: str | None = None
    similarity_score: float = 0.93
    vector_id: str = "point-1"


@dataclass
class _FakeVectorStore:
    hits: tuple[_FakeHitSpec, ...] = ()
    fail: BaseException | None = None
    last_top_k: int | None = None
    last_metadata_filter: MetadataFilter | None = None
    last_query_vector: np.ndarray | None = None

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
        self.last_top_k = top_k
        self.last_metadata_filter = metadata_filter
        self.last_query_vector = np.asarray(query_embedding, dtype=np.float32)
        if self.fail is not None:
            raise self.fail
        return tuple(self._build_hit(index, spec, scope) for index, spec in enumerate(self.hits))

    def delete(self, ids, *, scope: VectorStoreScope) -> None:
        return None

    def count(self, *, scope: VectorStoreScope) -> int:
        return len(self.hits)

    def _build_hit(self, rank: int, spec: _FakeHitSpec, scope: VectorStoreScope) -> VectorStoreHit:
        metadata = {
            "offer_id": spec.offer_id,
            "catalog_id": spec.catalog_id,
        }
        if spec.source_revision is not None:
            metadata["source_revision"] = spec.source_revision
        document = KnowledgeDocument.model_validate(
            {
                "schema_version": 1,
                "identity": {
                    "document_id": spec.vector_id,
                    "root_document_id": spec.vector_id,
                },
                "scope": {"tenant_id": scope.tenant_id},
                "content": "semantic offer text",
                "metadata": metadata,
                "provenance": {
                    "source_kind": "vpi_bootstrap",
                    "source_id": spec.offer_id,
                    "provider_id": "hf",
                },
            }
        )
        return VectorStoreHit(
            vector_id=spec.vector_id,
            document=document,
            similarity_score=spec.similarity_score,
            rank=rank,
        )


def _configuration(*, dimension: int = 4) -> VpiEmbeddingConfiguration:
    return VpiEmbeddingConfiguration(
        profile=EmbeddingProfile(provider="hf", model="BAAI/bge-m3"),
        expected_dimension=dimension,
    )


def _execution_configuration() -> VpiEmbeddingProviderExecutionConfiguration:
    return VpiEmbeddingProviderExecutionConfiguration(
        execution=EmbeddingProviderExecutionConfig(device="cpu", batch_size=1),
    )


_CANONICAL_EMBEDDING = ExpectedVectorIdentity(
    provider="hf",
    model="BAAI/bge-m3",
    revision="5617a9f61b028005a4858fdac845db406aefb181",
    dimension=4,
)


@dataclass
class _FakeVectorIndexIdentityResolver:
    resolution: VectorIndexIdentityResolution

    def resolve(self, expected: ExpectedVectorIndexRuntimeIdentity) -> VectorIndexIdentityResolution:
        return self.resolution


def _compatibility_gate(
    *,
    dimension: int = 4,
    tenant_id: str = "default",
    resolution: VectorIndexIdentityResolution | None = None,
) -> VectorIndexCompatibilityGate:
    embedding = ExpectedVectorIdentity(
        provider=_CANONICAL_EMBEDDING.provider,
        model=_CANONICAL_EMBEDDING.model,
        revision=_CANONICAL_EMBEDDING.revision,
        dimension=dimension,
    )
    expected = expected_vector_index_identity_from_bootstrap_vector_identity(
        target=VectorIndexIdentity(
            logical_name="vpi-product-embeddings",
            tenant_id=tenant_id,
        ),
        embedding_identity=embedding,
    )
    if resolution is None:
        resolution = VectorIndexIdentityResolution(
            status=VectorIndexIdentityResolutionStatus.RESOLVED,
            identity=ResolvedVectorIndexRuntimeIdentity(
                target=expected.target,
                exists=True,
                reachable=True,
                provider=expected.provider,
                model=expected.model,
                revision=expected.revision,
                dimension=expected.dimension,
                metric=expected.metric,
                content_identity=expected.content_identity,
            ),
        )
    return VectorIndexCompatibilityGate(
        expected=expected,
        resolver=_FakeVectorIndexIdentityResolver(resolution=resolution),
    )


def _adapter(
    *,
    vector_store: _FakeVectorStore,
    provider: _FakeEmbeddingProvider | None = None,
    catalog_scope_id: str | None = _CATALOG_ID,
    dimension: int = 4,
    compatibility_gate: VectorIndexCompatibilityGate | None = None,
) -> QdrantVectorCandidateSearchAdapter:
    ensure_embedding_provider_integrations_registered()
    embedding = IntergraxEmbeddingBootstrapAdapter(
        _configuration(dimension=dimension),
        provider=provider or _FakeEmbeddingProvider(dimension=dimension),
        execution_configuration=_execution_configuration(),
    )
    return QdrantVectorCandidateSearchAdapter.from_dependencies(
        vector_store=vector_store,
        scope=VectorStoreScope(tenant_id="default"),
        embedding=embedding,
        embedding_configuration=_configuration(dimension=dimension),
        catalog_scope_id=catalog_scope_id,
        compatibility_gate=compatibility_gate or _compatibility_gate(dimension=dimension),
    )


def _vector_load_record(*, offer_id: str, source_revision: str | None = None) -> VectorLoadRecord:
    return VectorLoadRecord(
        logical_point_id=f"point-{offer_id}",
        source_ref=SourceRecordRef(
            offer_id=ProductOfferId(offer_id),
            catalog_id=_CATALOG_ID,
            source_revision=source_revision,
        ),
        semantic_text_hash="hash-1",
        dense_embedding=(0.1, 0.2, 0.3, 0.4),
        embedding_provider="hf",
        embedding_model="BAAI/bge-m3",
        embedding_revision="5617a9f61b028005a4858fdac845db406aefb181",
        embedding_dimension=4,
        derivation_version="v1",
    )


def test_storage_write_payload_decodes_to_same_identity() -> None:
    record = _vector_load_record(offer_id="990-pro-2tb", source_revision="rev-a")
    payload = payload_from_record(record).to_provider_payload()
    identity = decode_vector_hit_identity_from_storage_payload(payload)
    assert identity.catalog_id == _CATALOG_ID
    assert identity.offer_id.value == "990-pro-2tb"
    assert identity.source_revision == "rev-a"
    assert identity.to_source_record_ref() == record.source_ref


def test_similarity_score_parity_exposes_cosine_similarity() -> None:
    adapter = _adapter(
        vector_store=_FakeVectorStore(
            hits=(_FakeHitSpec(offer_id="near-offer", similarity_score=0.93),),
        )
    )
    result = adapter.search(VectorSearchQuery(query_text="2TB high performance NVMe SSD", limit=3))
    assert result.failure is None
    assert result.candidates[0].channel_score == VectorChannelScore(cosine_similarity=0.93)


@pytest.mark.parametrize(
    ("score",),
    [
        (float("nan"),),
        (float("inf"),),
        (1.5,),
        (-1.5,),
    ],
)
def test_invalid_provider_score_fails_closed(score: float) -> None:
    adapter = _adapter(
        vector_store=_FakeVectorStore(
            hits=(_FakeHitSpec(offer_id="bad-score", similarity_score=score),),
        )
    )
    result = adapter.search(VectorSearchQuery(query_text="probe", limit=3))
    assert result.candidates == ()
    assert result.failure is not None
    assert result.failure.kind is CatalogSearchFailureKind.INVALID_QUERY


def test_missing_offer_id_fails_closed_without_catalog_fallback() -> None:
    document = KnowledgeDocument.model_validate(
        {
            "schema_version": 1,
            "identity": {"document_id": "p1", "root_document_id": "p1"},
            "scope": {"tenant_id": "default"},
            "content": "text",
            "metadata": {"catalog_id": _CATALOG_ID},
            "provenance": {
                "source_kind": "vpi_bootstrap",
                "source_id": "missing-offer",
                "provider_id": "hf",
            },
        }
    )

    class _BrokenIdentityStore(_FakeVectorStore):
        def query(self, query_embedding, *, scope, top_k, metadata_filter=None, include_embeddings=False):
            self.last_top_k = top_k
            self.last_metadata_filter = metadata_filter
            self.last_query_vector = np.asarray(query_embedding, dtype=np.float32)
            return (
                VectorStoreHit(
                    vector_id="p1",
                    document=document,
                    similarity_score=0.8,
                    rank=0,
                ),
            )

    adapter = _adapter(vector_store=_BrokenIdentityStore(hits=()))
    result = adapter.search(VectorSearchQuery(query_text="probe", limit=3))
    assert result.candidates == ()
    assert result.failure is not None
    assert result.failure.kind is CatalogSearchFailureKind.INVALID_QUERY


def test_query_limit_passes_exact_top_k_to_provider() -> None:
    vector_store = _FakeVectorStore(hits=())
    provider = _FakeEmbeddingProvider()
    adapter = _adapter(vector_store=vector_store, provider=provider)
    adapter.search(VectorSearchQuery(query_text="NVMe SSD", limit=7))
    assert vector_store.last_top_k == 7


def test_one_embedding_call_per_search() -> None:
    provider = _FakeEmbeddingProvider()
    adapter = _adapter(vector_store=_FakeVectorStore(hits=()), provider=provider)
    query_text = "2TB high performance NVMe SSD"
    adapter.search(VectorSearchQuery(query_text=query_text, limit=3))
    assert provider.calls == [[query_text]]


def test_compatibility_failure_prevents_embedding_call() -> None:
    provider = _FakeEmbeddingProvider()
    adapter = _adapter(
        vector_store=_FakeVectorStore(hits=(_FakeHitSpec(offer_id="blocked"),)),
        provider=provider,
        compatibility_gate=_compatibility_gate(
            resolution=VectorIndexIdentityResolution(
                status=VectorIndexIdentityResolutionStatus.INDEX_MISSING,
                identity=None,
            )
        ),
    )
    result = adapter.search(VectorSearchQuery(query_text="probe", limit=3))
    assert result.candidates == ()
    assert result.failure is not None
    assert provider.calls == []


def test_compatibility_failure_prevents_vector_store_query() -> None:
    vector_store = _FakeVectorStore(hits=(_FakeHitSpec(offer_id="blocked"),))
    adapter = _adapter(
        vector_store=vector_store,
        compatibility_gate=_compatibility_gate(
            resolution=VectorIndexIdentityResolution(
                status=VectorIndexIdentityResolutionStatus.INDEX_MISSING,
                identity=None,
            )
        ),
    )
    result = adapter.search(VectorSearchQuery(query_text="probe", limit=3))
    assert result.candidates == ()
    assert vector_store.last_top_k is None


def test_embedding_dimension_mismatch_fails_closed() -> None:
    provider = _FakeEmbeddingProvider(
        dimension=4,
        vectors=[[1.0, 0.0, 0.0]],
    )
    adapter = _adapter(vector_store=_FakeVectorStore(hits=()), provider=provider, dimension=4)
    result = adapter.search(VectorSearchQuery(query_text="probe", limit=3))
    assert result.candidates == ()
    assert result.failure is not None
    assert result.failure.kind is CatalogSearchFailureKind.INVALID_QUERY
    assert "dimension mismatch" in result.failure.message


def test_embedding_provider_unavailable_maps_to_failure() -> None:
    provider = _FakeEmbeddingProvider(fail=RuntimeError("provider down"))
    adapter = _adapter(vector_store=_FakeVectorStore(hits=()), provider=provider)
    result = adapter.search(VectorSearchQuery(query_text="probe", limit=3))
    assert result.candidates == ()
    assert result.failure is not None
    assert result.failure.kind is CatalogSearchFailureKind.UNAVAILABLE


def test_vector_store_unavailable_maps_to_failure() -> None:
    adapter = _adapter(
        vector_store=_FakeVectorStore(fail=IntegrationDependencyError("qdrant query failed")),
    )
    result = adapter.search(VectorSearchQuery(query_text="probe", limit=3))
    assert result.candidates == ()
    assert result.failure is not None
    assert result.failure.kind is CatalogSearchFailureKind.UNAVAILABLE


def test_vector_store_contract_error_maps_to_invalid_query() -> None:
    adapter = _adapter(
        vector_store=_FakeVectorStore(fail=VectorStoreContractError("bad query vector")),
    )
    result = adapter.search(VectorSearchQuery(query_text="probe", limit=3))
    assert result.candidates == ()
    assert result.failure is not None
    assert result.failure.kind is CatalogSearchFailureKind.INVALID_QUERY


def test_zero_hits_success_empty() -> None:
    adapter = _adapter(vector_store=_FakeVectorStore(hits=()))
    result = adapter.search(VectorSearchQuery(query_text="wireless mouse", limit=5))
    assert result.failure is None
    assert result.candidates == ()


def test_semantic_neighbor_fixture_ordering_with_fake_scores() -> None:
    adapter = _adapter(
        vector_store=_FakeVectorStore(
            hits=(
                _FakeHitSpec(
                    offer_id="990-pro-1tb",
                    similarity_score=0.81,
                    vector_id="near-1tb",
                ),
                _FakeHitSpec(
                    offer_id="990-pro-2tb",
                    similarity_score=0.95,
                    vector_id="near-2tb",
                ),
                _FakeHitSpec(
                    offer_id="wireless-mouse",
                    similarity_score=0.12,
                    vector_id="unrelated",
                ),
            ),
        )
    )
    result = adapter.search(
        VectorSearchQuery(query_text="2TB high performance NVMe SSD", limit=3)
    )
    assert result.failure is None
    assert [candidate.offer_id.value for candidate in result.candidates] == [
        "990-pro-2tb",
        "990-pro-1tb",
        "wireless-mouse",
    ]


def test_duplicate_identity_keeps_highest_similarity_within_top_k() -> None:
    adapter = _adapter(
        vector_store=_FakeVectorStore(
            hits=(
                _FakeHitSpec(offer_id="dup-offer", similarity_score=0.70, vector_id="dup-a"),
                _FakeHitSpec(offer_id="dup-offer", similarity_score=0.91, vector_id="dup-b"),
            ),
        )
    )
    result = adapter.search(VectorSearchQuery(query_text="probe", limit=2))
    assert result.failure is None
    assert len(result.candidates) == 1
    assert result.candidates[0].offer_id.value == "dup-offer"
    assert result.candidates[0].channel_score.cosine_similarity == 0.91


def test_catalog_scope_filter_is_provider_side() -> None:
    vector_store = _FakeVectorStore(hits=())
    adapter = _adapter(vector_store=vector_store, catalog_scope_id=_CATALOG_ID)
    adapter.search(VectorSearchQuery(query_text="probe", limit=3))
    assert vector_store.last_metadata_filter is not None
    assert vector_store.last_metadata_filter.conditions["catalog_id"] == _CATALOG_ID


def test_vector_score_is_not_verification_confidence() -> None:
    adapter = _adapter(
        vector_store=_FakeVectorStore(
            hits=(_FakeHitSpec(offer_id="verified-looking", similarity_score=1.0),),
        )
    )
    result = adapter.search(VectorSearchQuery(query_text="exact product", limit=1))
    candidate = result.candidates[0]
    assert candidate.channel is RetrievalChannel.VECTOR
    assert candidate.channel_score.cosine_similarity == 1.0
    assert type(candidate.channel_score) is VectorChannelScore


def test_build_vector_candidate_search_returns_port(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VPI_EMBEDDING_MODEL_REVISION", _CANONICAL_EMBEDDING.revision)
    port = build_vector_candidate_search(
        collection_name="vpi-product-embeddings",
        catalog_scope_id=_CATALOG_ID,
    )
    assert isinstance(port, QdrantVectorCandidateSearchAdapter)
    port.close()


def test_legacy_adapter_is_marked_reference_only() -> None:
    source = _LEGACY_ADAPTER_PATH.read_text(encoding="utf-8")
    assert "LEGACY / REFERENCE ONLY" in source


def test_composition_builds_vector_port_without_legacy_adapter() -> None:
    source = (_VPI_ROOT / "retrieval/composition.py").read_text(encoding="utf-8")
    assert "build_vector_candidate_search" in source
    assert "PlatformVectorSearchAdapter" not in source


def test_application_has_zero_qdrant_or_pgvector_imports() -> None:
    forbidden_roots = {"qdrant", "qdrant_client", "pgvector"}
    violations: list[str] = []
    for module_path in sorted(_APPLICATION_ROOT.rglob("*.py")):
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in forbidden_roots:
                        violations.append(str(module_path))
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.split(".")[0] in forbidden_roots:
                    violations.append(str(module_path))
    assert violations == []


def test_multi_channel_service_unchanged_signature() -> None:
    param_names = [service_field.name for service_field in fields(MultiChannelRetrievalService)]
    assert param_names == [
        "exact_lookup",
        "lexical_search",
        "structured_search",
        "vector_search",
    ]


@dataclass
class _RecordingVectorSearchPort:
    queries: list[VectorSearchQuery] = field(default_factory=list)

    def search(self, query: VectorSearchQuery):
        from platform_proofs.scenarios.verified_product_identification.application.contracts.results import (
            VectorSearchResult,
        )

        self.queries.append(query)
        return VectorSearchResult(candidates=())


def test_pluginability_with_fake_and_qdrant_vector_adapter() -> None:
    request = MultiChannelRetrievalRequest(
        vector_query=VectorSearchQuery(query_text="NVMe SSD", limit=2),
    )
    service_with_fake = MultiChannelRetrievalService(
        exact_lookup=FakeExactLookupA(),
        lexical_search=RecordingVectorSearch(),
        structured_search=RecordingStructuredSearch(),
        vector_search=_RecordingVectorSearchPort(),
    )
    service_with_vector = MultiChannelRetrievalService(
        exact_lookup=FakeExactLookupA(),
        lexical_search=RecordingVectorSearch(),
        structured_search=RecordingStructuredSearch(),
        vector_search=_adapter(vector_store=_FakeVectorStore(hits=())),
    )
    fake_result = service_with_fake.retrieve(request)
    vector_result = service_with_vector.retrieve(request)
    assert fake_result.vector.search_result is not None
    assert vector_result.vector.search_result is not None


def test_canonical_adapter_has_no_loose_metadata_fallback() -> None:
    source = (_ADAPTER_ROOT / "qdrant_vector_candidate_search_adapter.py").read_text(
        encoding="utf-8"
    )
    assert 'metadata.get("catalog_id"' not in source
    assert "configured_catalog_id" not in source.lower()


def test_platform_vector_search_adapter_delegates_to_canonical_implementation() -> None:
    source = _LEGACY_ADAPTER_PATH.read_text(encoding="utf-8")
    assert "QdrantVectorCandidateSearchAdapter" in source
    assert "PlatformVectorSearchAdapter" in source


def test_vector_channel_score_rejects_out_of_range_at_domain_boundary() -> None:
    with pytest.raises(ValueError, match="within \\[-1.0, 1.0\\]"):
        VectorChannelScore(cosine_similarity=1.5)
