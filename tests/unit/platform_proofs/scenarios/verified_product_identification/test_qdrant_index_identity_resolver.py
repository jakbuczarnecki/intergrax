"""Unit tests for Qdrant vector index identity resolver public seams."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from intergrax.integrations.contracts.vector_index_administration import (
    VectorIndexDescription,
    VectorIndexIdentity,
    VectorSearchCapability,
)
from intergrax.integrations.contracts.vector_index_metadata import VectorIndexPointPayload

from platform_proofs.scenarios.verified_product_identification.integrations.search_store.qdrant_index_identity_resolver import (
    QdrantVectorIndexIdentityResolver,
)
from platform_proofs.scenarios.verified_product_identification.integrations.search_store.vector_index_metadata import (
    INDEX_METADATA_LOGICAL_POINT_ID,
    INDEX_METADATA_MARKER_PAYLOAD_KEY,
    INDEX_METADATA_MARKER_VALUE,
    DATA_PACK_CONTENT_IDENTITY_PAYLOAD_KEY,
    VECTOR_TARGET_LOGICAL_NAME_PAYLOAD_KEY,
    VECTOR_TARGET_TENANT_ID_PAYLOAD_KEY,
)
from platform_proofs.scenarios.verified_product_identification.integrations.search_store.vector_index_runtime_identity import (
    ExpectedVectorIndexRuntimeIdentity,
    VectorIndexIdentityResolutionStatus,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.qdrant.configuration import (
    ExpectedVectorIdentity,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.qdrant.payload import (
    DERIVATION_VERSION_PAYLOAD_KEY,
    EMBEDDING_DIMENSION_PAYLOAD_KEY,
    EMBEDDING_MODEL_PAYLOAD_KEY,
    EMBEDDING_PROVIDER_PAYLOAD_KEY,
    EMBEDDING_REVISION_PAYLOAD_KEY,
    LOGICAL_POINT_ID_PAYLOAD_KEY,
    SEMANTIC_TEXT_HASH_PAYLOAD_KEY,
    SOURCE_CATALOG_PAYLOAD_KEY,
    SOURCE_OFFER_PAYLOAD_KEY,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[5]
_RESOLVER_PATH = (
    _REPO_ROOT
    / "platform_proofs/scenarios/verified_product_identification/integrations/search_store/qdrant_index_identity_resolver.py"
)
_TARGET = VectorIndexIdentity(logical_name="vpi-product-embeddings", tenant_id="default")
_EMBEDDING = ExpectedVectorIdentity(
    provider="hf",
    model="BAAI/bge-m3",
    revision="5617a9f61b028005a4858fdac845db406aefb181",
    dimension=4,
)


def _expected(*, content_identity: str | None = None) -> ExpectedVectorIndexRuntimeIdentity:
    return ExpectedVectorIndexRuntimeIdentity(
        target=_TARGET,
        provider=_EMBEDDING.provider,
        model=_EMBEDDING.model,
        revision=_EMBEDDING.revision,
        dimension=_EMBEDDING.dimension,
        metric="cosine",
        content_identity=content_identity,
    )


def _description(
    *,
    exists: bool = True,
    reachable: bool = True,
    dense_dimension: int | None = 4,
    dense_metric: str | None = "cosine",
) -> VectorIndexDescription:
    return VectorIndexDescription(
        identity=_TARGET,
        exists=exists,
        reachable=reachable,
        point_count=1 if exists else 0,
        dense_dimension=dense_dimension,
        dense_metric=dense_metric,
        present_capabilities=frozenset({VectorSearchCapability.DENSE}),
        dense_channel_name="dense" if dense_dimension is not None else None,
        sparse_lexical_channel_name=None,
    )


def _metadata_payload(*, content_identity: str = "pack-content-v1") -> dict[str, str | int]:
    return {
        INDEX_METADATA_MARKER_PAYLOAD_KEY: INDEX_METADATA_MARKER_VALUE,
        DATA_PACK_CONTENT_IDENTITY_PAYLOAD_KEY: content_identity,
        VECTOR_TARGET_LOGICAL_NAME_PAYLOAD_KEY: _TARGET.logical_name,
        VECTOR_TARGET_TENANT_ID_PAYLOAD_KEY: _TARGET.tenant_id,
        EMBEDDING_PROVIDER_PAYLOAD_KEY: _EMBEDDING.provider,
        EMBEDDING_MODEL_PAYLOAD_KEY: _EMBEDDING.model,
        EMBEDDING_REVISION_PAYLOAD_KEY: _EMBEDDING.revision,
        EMBEDDING_DIMENSION_PAYLOAD_KEY: _EMBEDDING.dimension,
    }


def _embedding_probe_payload() -> dict[str, str | int]:
    return {
        LOGICAL_POINT_ID_PAYLOAD_KEY: "probe-point-1",
        SOURCE_CATALOG_PAYLOAD_KEY: "wdc-v2-selected",
        SOURCE_OFFER_PAYLOAD_KEY: "offer-1",
        SEMANTIC_TEXT_HASH_PAYLOAD_KEY: "hash-v1",
        EMBEDDING_PROVIDER_PAYLOAD_KEY: _EMBEDDING.provider,
        EMBEDDING_MODEL_PAYLOAD_KEY: _EMBEDDING.model,
        EMBEDDING_REVISION_PAYLOAD_KEY: _EMBEDDING.revision,
        EMBEDDING_DIMENSION_PAYLOAD_KEY: _EMBEDDING.dimension,
        DERIVATION_VERSION_PAYLOAD_KEY: "v1",
    }


@dataclass(slots=True)
class _FakeIndexAdmin:
    description: VectorIndexDescription
    fail: BaseException | None = None
    closed: bool = False

    def probe(self):
        raise AssertionError("probe should not be called by resolver")

    def describe_index(self, identity: VectorIndexIdentity) -> VectorIndexDescription:
        if self.fail is not None:
            raise self.fail
        return self.description

    def prepare_index(self, spec):
        raise AssertionError("prepare_index should not be called by resolver")

    def close(self) -> None:
        self.closed = True


@dataclass(slots=True)
class _FakeMetadataReader:
    metadata_payload: dict[str, str | int] | None = None
    fallback_payload: dict[str, str | int] | None = None
    logical_point_ids: list[str] = field(default_factory=list)
    closed: bool = False

    def retrieve_point_by_logical_id(
        self,
        identity: VectorIndexIdentity,
        logical_point_id: str,
    ) -> VectorIndexPointPayload | None:
        self.logical_point_ids.append(logical_point_id)
        if self.metadata_payload is None:
            return None
        return VectorIndexPointPayload(payload=self.metadata_payload)

    def retrieve_first_point_payload(
        self,
        identity: VectorIndexIdentity,
        *,
        limit: int = 1,
    ) -> VectorIndexPointPayload | None:
        if self.fallback_payload is None:
            return None
        return VectorIndexPointPayload(payload=self.fallback_payload)

    def close(self) -> None:
        self.closed = True


def _resolver(
    *,
    description: VectorIndexDescription,
    metadata_payload: dict[str, str | int] | None = None,
    fallback_payload: dict[str, str | int] | None = None,
) -> QdrantVectorIndexIdentityResolver:
    return QdrantVectorIndexIdentityResolver(
        _index_admin=_FakeIndexAdmin(description=description),
        _metadata_reader=_FakeMetadataReader(
            metadata_payload=metadata_payload,
            fallback_payload=fallback_payload,
        ),
    )


def test_resolver_uses_public_provider_seams_only() -> None:
    resolver = _resolver(
        description=_description(),
        metadata_payload=_metadata_payload(),
    )
    resolution = resolver.resolve(_expected(content_identity="pack-content-v1"))
    assert resolution.status is VectorIndexIdentityResolutionStatus.RESOLVED
    assert resolution.identity is not None
    assert resolution.identity.provider == "hf"
    assert resolution.identity.metric == "cosine"
    assert resolution.identity.content_identity == "pack-content-v1"


def test_metadata_logical_id_resolves_without_provider_encoding_knowledge() -> None:
    reader = _FakeMetadataReader(metadata_payload=_metadata_payload())
    resolver = QdrantVectorIndexIdentityResolver(
        _index_admin=_FakeIndexAdmin(description=_description()),
        _metadata_reader=reader,
    )
    resolver.resolve(_expected(content_identity="pack-content-v1"))
    assert reader.logical_point_ids == [INDEX_METADATA_LOGICAL_POINT_ID]


def test_missing_metadata_point_uses_bounded_fallback_payload() -> None:
    resolver = _resolver(
        description=_description(),
        metadata_payload=None,
        fallback_payload=_embedding_probe_payload(),
    )
    resolution = resolver.resolve(_expected())
    assert resolution.status is VectorIndexIdentityResolutionStatus.RESOLVED
    assert resolution.identity is not None
    assert resolution.identity.content_identity is None


def test_provider_unavailable_when_administration_unreachable() -> None:
    resolver = QdrantVectorIndexIdentityResolver(
        _index_admin=_FakeIndexAdmin(
            description=_description(reachable=False),
        ),
        _metadata_reader=_FakeMetadataReader(),
    )
    resolution = resolver.resolve(_expected())
    assert resolution.status is VectorIndexIdentityResolutionStatus.PROVIDER_UNAVAILABLE


def test_index_missing_when_collection_absent() -> None:
    resolver = _resolver(
        description=_description(exists=False, dense_dimension=None, dense_metric=None),
    )
    resolution = resolver.resolve(_expected())
    assert resolution.status is VectorIndexIdentityResolutionStatus.INDEX_MISSING


def test_metadata_unavailable_when_dimension_missing() -> None:
    resolver = _resolver(description=_description(dense_dimension=None, dense_metric=None))
    resolution = resolver.resolve(_expected())
    assert resolution.status is VectorIndexIdentityResolutionStatus.METADATA_UNAVAILABLE


def test_close_closes_injected_public_seams() -> None:
    admin = _FakeIndexAdmin(description=_description())
    reader = _FakeMetadataReader(metadata_payload=_metadata_payload())
    resolver = QdrantVectorIndexIdentityResolver(
        _index_admin=admin,
        _metadata_reader=reader,
    )
    resolver.close()
    assert admin.closed is True
    assert reader.closed is True


def test_resolver_module_has_no_private_qdrant_provider_imports() -> None:
    tree = ast.parse(_RESOLVER_PATH.read_text(encoding="utf-8"))
    forbidden_symbols = {"_build_qdrant_client", "_normalize_point_id"}
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            if not node.module.startswith("intergrax.integrations.providers.vector_store.qdrant"):
                continue
            for alias in node.names:
                if alias.name in forbidden_symbols or alias.name.startswith("_"):
                    violations.append(f"{node.module}:{alias.name}")
    assert violations == []


def test_search_store_rejects_private_qdrant_provider_symbols() -> None:
    search_store_root = (
        _REPO_ROOT
        / "platform_proofs/scenarios/verified_product_identification/integrations/search_store"
    )
    forbidden_symbols = {"_build_qdrant_client", "_normalize_point_id"}
    violations: list[str] = []
    for module_path in sorted(search_store_root.rglob("*.py")):
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.module is None:
                continue
            if not node.module.startswith("intergrax.integrations.providers.vector_store.qdrant"):
                continue
            for alias in node.names:
                if alias.name in forbidden_symbols or alias.name.startswith("_"):
                    violations.append(
                        f"{module_path.relative_to(_REPO_ROOT)} -> {node.module}.{alias.name}"
                    )
    assert violations == []
