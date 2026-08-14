from __future__ import annotations

import inspect
from typing import Any

import pytest

from intergrax.integrations._shared.p7.factories import _TypesenseHttpVectorStore
from intergrax.integrations.providers.vector_store.lancedb.opens import _open_rag_store
from intergrax.integrations.providers.vector_store.vespa.adapter import _VespaVectorStore
from intergrax.integrations.providers.vector_store.vespa.config import VespaIntegrationConfig
from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.vectorstore.contracts.native_vectorstore import (
    MetadataFilter,
    VectorStoreContractError,
    VectorStoreRecord,
    VectorStoreScope,
)
from intergrax.utils import attribute_access

pytestmark = pytest.mark.unit


def _scope(
    tenant_id: str = "tenant-a",
    namespace: str | None = "namespace-a",
    workspace_id: str | None = "workspace-a",
) -> VectorStoreScope:
    return VectorStoreScope(
        tenant_id=tenant_id,
        namespace=namespace,
        workspace_id=workspace_id,
    )


def _record(vector_id: str, scope: VectorStoreScope) -> VectorStoreRecord:
    document = KnowledgeDocument.model_validate(
        {
            "schema_version": 1,
            "identity": {"document_id": vector_id, "root_document_id": vector_id},
            "scope": {
                "tenant_id": scope.tenant_id,
                "namespace": scope.namespace,
                "workspace_id": scope.workspace_id,
            },
            "content": f"content-{vector_id}",
            "metadata": {"owner": "test"},
            "provenance": {"source_kind": "test", "source_id": vector_id},
        }
    )
    return VectorStoreRecord(
        document=document,
        embedding=[1.0, 0.0],
        vector_id=vector_id,
    )


class _Response:
    status_code = 200

    def __init__(self, payload: object) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self._payload


class _VespaFakeClient:
    def __init__(self) -> None:
        self.fed: list[dict[str, object]] = []
        self.deleted: list[str] = []
        self.yql: list[str] = []

    def feed_document(self, *, doc_id: str, fields: dict[str, object]) -> str:
        self.fed.append({"id": doc_id, "fields": fields})
        return doc_id

    def query_yql(self, yql: str, *, hits: int, ranking: str | None = None) -> list[dict[str, object]]:
        del hits, ranking
        self.yql.append(yql)
        return [
            {
                "id": "doc-a",
                "fields": {
                    "content": "content-doc-a",
                    "metadata": {
                        "schema_version": 1,
                        "document_id": "doc-a",
                        "root_document_id": "doc-a",
                        "tenant_id": "tenant-a",
                        "namespace": "namespace-a",
                        "workspace_id": "workspace-a",
                        "source_kind": "test",
                        "source_id": "doc-a",
                    },
                },
            }
        ]

    def delete_document(self, doc_id: str) -> None:
        self.deleted.append(doc_id)

    def count_documents(self, *, yql: str) -> int:
        self.yql.append(yql)
        return 1


class _TypesenseFakeHttp:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, object] | None]] = []

    def post(self, path: str, json: dict[str, object]) -> _Response:
        self.calls.append(("POST", path, json))
        if path.endswith("/search"):
            return _Response(
                {
                    "hits": [
                        {
                            "document": {
                                "id": "doc-a",
                                "content": "content-doc-a",
                                "embedding": [1.0, 0.0],
                                "metadata": {
                                    "schema_version": 1,
                                    "document_id": "doc-a",
                                    "root_document_id": "doc-a",
                                    "tenant_id": "tenant-a",
                                    "namespace": "namespace-a",
                                    "workspace_id": "workspace-a",
                                    "source_kind": "test",
                                    "source_id": "doc-a",
                                },
                            },
                            "vector_distance": 0.0,
                        }
                    ]
                }
            )
        return _Response({"success": True})

    def get(self, path: str) -> _Response:
        self.calls.append(("GET", path, None))
        return _Response({})


def _provider_method_names(provider: object) -> dict[str, inspect.Signature]:
    return {
        name: inspect.signature(object.__getattribute__(type(provider), name))
        for name in ("add_records", "query", "delete", "count")
    }


@pytest.mark.parametrize(
    "provider",
    [
        _VespaVectorStore(VespaIntegrationConfig(tenant_id="tenant-a"), _VespaFakeClient()),
        _open_rag_store(),
        _TypesenseHttpVectorStore(_TypesenseFakeHttp(), collection="intergrax"),
    ],
    ids=["vespa", "lancedb", "typesense"],
)
def test_remaining_providers_expose_only_native_scoped_methods(provider: object) -> None:
    signatures = _provider_method_names(provider)
    assert "add_documents" not in type(provider).__dict__
    assert signatures["add_records"].parameters["scope"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signatures["query"].parameters["scope"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signatures["delete"].parameters["scope"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signatures["count"].parameters["scope"].kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize("provider_name", ["vespa", "lancedb", "typesense"])
def test_invalid_batch_is_rejected_before_provider_transport(provider_name: str) -> None:
    scope = _scope()
    foreign = _record("foreign", _scope(tenant_id="tenant-b"))

    if provider_name == "vespa":
        client = _VespaFakeClient()
        provider: Any = _VespaVectorStore(
            VespaIntegrationConfig(tenant_id="tenant-a"),
            client,
        )
    elif provider_name == "lancedb":
        provider = _open_rag_store()
        client = provider
    else:
        client = _TypesenseFakeHttp()
        provider = _TypesenseHttpVectorStore(client, collection="intergrax")

    with pytest.raises(VectorStoreContractError):
        provider.add_records([foreign], scope=scope)
    assert not (
        attribute_access.optional(client, "fed", None)
        or attribute_access.optional(client, "calls", [])
    )


@pytest.mark.parametrize("provider_name", ["vespa", "lancedb", "typesense"])
def test_scope_is_injected_and_native_hits_reconstruct_documents(provider_name: str) -> None:
    scope = _scope()
    record = _record("doc-a", scope)

    if provider_name == "vespa":
        client = _VespaFakeClient()
        provider: Any = _VespaVectorStore(
            VespaIntegrationConfig(tenant_id="tenant-a"),
            client,
        )
    elif provider_name == "lancedb":
        client = _open_rag_store()
        provider = client
    else:
        client = _TypesenseFakeHttp()
        provider = _TypesenseHttpVectorStore(client, collection="intergrax")

    assert provider.add_records([record], scope=scope) == ["doc-a"]
    hits = provider.query(
        [1.0, 0.0],
        scope=scope,
        top_k=1,
        metadata_filter=MetadataFilter(conditions={"owner": "test"}),
    )
    assert hits[0].document.scope.tenant_id == "tenant-a"
    assert hits[0].document.scope.namespace == "namespace-a"
    assert hits[0].document.scope.workspace_id == "workspace-a"

    if provider_name == "vespa":
        fields = client.fed[0]["fields"]
        assert fields["tenant_id"] == "tenant-a"
        assert fields["namespace"] == "namespace-a"
        assert "tenant_id contains" in client.yql[-1]
    elif provider_name == "typesense":
        query = next(body for method, path, body in client.calls if method == "POST" and path.endswith("/search"))
        assert query is not None
        assert "tenant_id:=`tenant-a`" in str(query["filter_by"])


def test_typesense_filter_spoof_and_unsupported_operations_fail_closed() -> None:
    client = _TypesenseFakeHttp()
    provider = _TypesenseHttpVectorStore(client, collection="intergrax")
    scope = _scope()

    with pytest.raises(VectorStoreContractError):
        provider.query([1.0, 0.0], scope=scope, top_k=1, metadata_filter=MetadataFilter(conditions={"bad field": "x"}))
    with pytest.raises(VectorStoreContractError, match="scoped delete"):
        provider.delete(["doc-a"], scope=scope)
    with pytest.raises(VectorStoreContractError, match="scoped count"):
        provider.count(scope=scope)
