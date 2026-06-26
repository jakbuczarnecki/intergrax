# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from intergrax.integrations.providers.vector_store.inmemory.rag_store import InMemoryVectorStore
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.rag.vectorstore.vectorstore_manager import VectorstoreManager
from intergrax.tools.providers.rag.ingest_contracts import RagIngestInput
from intergrax.tools.providers.rag.ingest_service import perform_rag_ingest
from intergrax.tools.providers.rag.scope import (
    TENANT_ID_METADATA_CONFLICT,
    authoritative_tenant_id,
    resolve_tenant_scoped_vectorstore,
    vectorstore_tenant_id,
)
from intergrax.tools.providers.rag.service import perform_rag_retrieve
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.tools.providers.rag.contracts import RagRetrieveInput
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_lkw_resolve_request_scope_uses_runtime_request_tenant_id() -> None:
    import sys
    from pathlib import Path

    agents_root = Path(__file__).resolve().parents[5] / "agents"
    if str(agents_root) not in sys.path:
        sys.path.insert(0, str(agents_root))
    from lkw_shared.runtime_helpers import resolve_request_scope

    exec_ctx = RuntimeExecutionContext(
        task_id="task-1",
        run_id="run-1",
        agent_id="local_indexer",
        request=RuntimeRequest(
            agent_id="local_indexer",
            tenant_id="lkw-smoke",
            user_id="local-user",
            session_id="s1",
            message="index",
            metadata={"tenant_id": "default", "collection_id": "ws-1"},
        ),
    )

    scope = resolve_request_scope(exec_ctx)

    assert scope["tenant_id"] == "lkw-smoke"
    assert scope["user_id"] == "local-user"


@dataclass(frozen=True)
class _Cfg:
    tenant_id: str


class _StoreStub:
    def __init__(self, tenant_id: str) -> None:
        self.cfg = _Cfg(tenant_id=tenant_id)


def test_authoritative_tenant_id_uses_request_when_metadata_absent() -> None:
    tenant, conflict = authoritative_tenant_id(request_tenant="lkw-smoke", metadata_tenant=None)
    assert tenant == "lkw-smoke"
    assert conflict is None


def test_authoritative_tenant_id_allows_matching_metadata() -> None:
    tenant, conflict = authoritative_tenant_id(request_tenant="lkw-smoke", metadata_tenant="lkw-smoke")
    assert tenant == "lkw-smoke"
    assert conflict is None


def test_authoritative_tenant_id_rejects_metadata_conflict() -> None:
    tenant, conflict = authoritative_tenant_id(request_tenant="lkw-smoke", metadata_tenant="other")
    assert tenant is None
    assert conflict == TENANT_ID_METADATA_CONFLICT


def test_resolve_tenant_scoped_vectorstore_rebinds_when_wired_tenant_differs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    default_manager = VectorstoreManager(_StoreStub("default"))
    rebound_manager = VectorstoreManager(_StoreStub("lkw-smoke"))
    created: list[str] = []

    def _fake_create(*, tenant_id: str | None = None, profile: Any = None, **_: Any):
        created.append(str(tenant_id))
        assert tenant_id == "lkw-smoke"
        return rebound_manager

    monkeypatch.setattr(
        "intergrax.tools.providers.rag.scope.create_vectorstore_manager",
        _fake_create,
    )

    ctx = ToolWiringContext(
        vectorstore_manager=default_manager,
        integration_profile=IntegrationProfile(),
    )
    scoped = resolve_tenant_scoped_vectorstore(ctx, "lkw-smoke")

    assert scoped is rebound_manager
    assert created == ["lkw-smoke"]
    assert resolve_tenant_scoped_vectorstore(ctx, "lkw-smoke") is rebound_manager
    assert created == ["lkw-smoke"]


def test_vectorstore_tenant_id_reads_provider_cfg() -> None:
    assert vectorstore_tenant_id(VectorstoreManager(_StoreStub("tenant-a"))) == "tenant-a"


def test_vectorstore_tenant_id_unwraps_integration_adapter() -> None:
    class _Bridge:
        def __init__(self) -> None:
            self._config = _Cfg(tenant_id="default")
            self._inner = _StoreStub("default")

    assert vectorstore_tenant_id(VectorstoreManager(_Bridge())) == "default"


class _FakeEmbeddingManager:
    class _Result:
        def __init__(self, docs, embeddings):
            self.documents = docs
            self.embeddings = embeddings

    def embed_documents(self, docs):
        return self._Result(docs, [[0.1, 0.2] for _ in docs])


class _RecordingVectorstore:
    def __init__(self, tenant_id: str) -> None:
        self.cfg = _Cfg(tenant_id=tenant_id)
        self.added_tenants: list[str | None] = []

    def add_documents(self, *, documents, embeddings, ids=None):
        for doc in documents:
            self.added_tenants.append((doc.metadata or {}).get("tenant_id"))


class _FakeLoader:
    def load_document(self, source: str, *, use_default_metadata=True, call_custom_metadata=None):
        meta = call_custom_metadata(Document(page_content="x", metadata={}), source) if call_custom_metadata else {}
        return [Document(page_content="hello", metadata=dict(meta))]


class _FakeSplitter:
    def split_documents(self, docs):
        return docs


def test_rag_ingest_uses_tenant_scoped_vectorstore(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "doc.txt"
    source.write_text("hello", encoding="utf-8")

    default_store = _RecordingVectorstore("default")
    scoped_store = _RecordingVectorstore("lkw-smoke")

    def _fake_create(*, tenant_id: str | None = None, profile: Any = None, **_: Any):
        assert tenant_id == "lkw-smoke"
        return VectorstoreManager(scoped_store)

    monkeypatch.setattr(
        "intergrax.tools.providers.rag.scope.create_vectorstore_manager",
        _fake_create,
    )

    ctx = ToolWiringContext(
        vectorstore_manager=VectorstoreManager(default_store),
        embedding_manager=_FakeEmbeddingManager(),
        integration_profile=IntegrationProfile(),
        extras={
            "documents_loader": _FakeLoader(),
            "documents_splitter": _FakeSplitter(),
        },
    )

    out = perform_rag_ingest(
        ctx,
        RagIngestInput(
            source_path=str(source),
            tenant_id="lkw-smoke",
            metadata={"tenant_id": "default"},
        ),
    )

    assert out.used is False
    assert out.reason == TENANT_ID_METADATA_CONFLICT


def test_rag_ingest_stamps_authoritative_tenant(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "doc.txt"
    source.write_text("hello", encoding="utf-8")
    scoped_store = _RecordingVectorstore("lkw-smoke")

    monkeypatch.setattr(
        "intergrax.tools.providers.rag.scope.create_vectorstore_manager",
        lambda *, tenant_id, profile, **_: VectorstoreManager(scoped_store),
    )

    ctx = ToolWiringContext(
        vectorstore_manager=VectorstoreManager(_RecordingVectorstore("default")),
        embedding_manager=_FakeEmbeddingManager(),
        integration_profile=IntegrationProfile(),
        extras={
            "documents_loader": _FakeLoader(),
            "documents_splitter": _FakeSplitter(),
        },
    )

    out = perform_rag_ingest(
        ctx,
        RagIngestInput(source_path=str(source), tenant_id="lkw-smoke", workspace_id="ws-1"),
    )

    assert out.used is True
    assert scoped_store.added_tenants == ["lkw-smoke"]


def test_rag_retrieve_resolves_tenant_scoped_vectorstore(monkeypatch: pytest.MonkeyPatch) -> None:
    default_store = InMemoryVectorStore(tenant_id="default")
    scoped_store = InMemoryVectorStore(tenant_id="lkw-smoke")
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        "intergrax.tools.providers.rag.scope.create_vectorstore_manager",
        lambda *, tenant_id, profile, **_: VectorstoreManager(scoped_store),
    )

    class _FakeRetrievalService:
        def retrieve(self, request):
            from intergrax.rag.retrieval.retrieval_result import RetrievalChunk, RetrievalResult, RetrievalTrace

            return RetrievalResult(
                used=True,
                reason="ok",
                chunks=[
                    RetrievalChunk(
                        id="chunk-1",
                        text="fixture sentence",
                        score=0.9,
                        metadata={"tenant_id": "lkw-smoke", "workspace_id": "ws-1"},
                    )
                ],
                citations=[],
                trace=RetrievalTrace(),
            )

    def _fake_resolve(*, vectorstore_manager, **_: Any):
        captured["manager"] = vectorstore_manager
        return _FakeRetrievalService()

    monkeypatch.setattr(
        "intergrax.tools.providers.rag.service.resolve_retrieval_service",
        _fake_resolve,
    )

    ctx = ToolWiringContext(
        vectorstore_manager=VectorstoreManager(default_store),
        embedding_manager=MagicMock(),
        integration_profile=IntegrationProfile(),
    )

    out = perform_rag_retrieve(
        ctx,
        RagRetrieveInput(query="fixture", tenant_id="lkw-smoke", workspace_id="ws-1"),
    )

    assert out.used is True
    assert vectorstore_tenant_id(captured["manager"]) == "lkw-smoke"
