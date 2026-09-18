# © Artur Czarnecki. All rights reserved.

"""PLUG-04 — RAG public EP typed load evidence."""

from __future__ import annotations

import importlib.metadata

import pytest

from intergrax.core.plugins.admission import PluginAdmissionReasonCode
from intergrax.core.plugins.discovery import (
    EP_RAG_CHUNKERS,
    EP_RAG_RETRIEVERS,
    EP_RAG_RERANKERS,
    reset_entry_point_spec_cache_for_tests,
)
from intergrax.rag.bootstrap.entry_point_load import (
    collect_rag_plugin_load_evidence,
    register_rag_chunker_entry_points,
    register_rag_retriever_entry_points,
    register_rag_reranker_entry_points,
)
from intergrax.rag.document_splitters.contracts.base_chunking_strategy import (
    BaseChunkingStrategy,
)
from intergrax.rag.document_splitters.registry.strategy_registry import (
    ChunkingStrategyRegistry,
)
from intergrax.rag.embedding.bootstrap.default_embedding_engine import (
    create_default_embedding_manager,
)
from intergrax.rag.retrievers.contracts.base_retriever import (
    BaseRetriever,
    BaseRetrieverPlugin,
    RetrievalHit,
    RetrieverQuery,
)
from intergrax.rag.retrievers.registry.retriever_registry import RetrieverRegistry
from intergrax.rag.rerankers.contracts.base_reranker import BaseReranker, BaseRerankerPlugin
from intergrax.rag.rerankers.registry.reranker_registry import RerankerRegistry
from intergrax.rag.vectorstore.bootstrap.vectorstore_bootstrap import (
    create_default_vectorstore_manager,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_ep_cache() -> None:
    reset_entry_point_spec_cache_for_tests()
    yield
    reset_entry_point_spec_cache_for_tests()


class _EntryPoint:
    def __init__(self, name: str, value: str, group: str) -> None:
        self.name = name
        self.value = value
        self.group = group


class _EntryPoints:
    def __init__(self, entries: list[_EntryPoint]) -> None:
        self._entries = entries

    def select(self, *, group: str) -> list[_EntryPoint]:
        return [entry for entry in self._entries if entry.group == group]


class _BrokenChunker:
    pass


def test_rag_chunker_report_rejects_invalid_sibling_without_crashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = _EntryPoints(
        [
            _EntryPoint(
                "broken",
                f"{__name__}:_BrokenChunker",
                EP_RAG_CHUNKERS,
            ),
        ]
    )
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: entries)

    report = register_rag_chunker_entry_points(
        ChunkingStrategyRegistry(),
        discover_entry_points=True,
    )

    assert report.group == EP_RAG_CHUNKERS
    assert report.registered_count == 0
    assert len(report.rejected) == 1
    assert report.rejected[0].reason_code == PluginAdmissionReasonCode.INVALID_TARGET_TYPE
    assert report.failed == ()


def test_rag_chunker_report_ordering_is_deterministic_across_input_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_a = _EntryPoints(
        [
            _EntryPoint("z_bad", f"{__name__}:_BrokenChunker", EP_RAG_CHUNKERS),
            _EntryPoint("a_bad", f"{__name__}:_BrokenChunker", EP_RAG_CHUNKERS),
        ]
    )
    run_b = _EntryPoints(
        [
            _EntryPoint("a_bad", f"{__name__}:_BrokenChunker", EP_RAG_CHUNKERS),
            _EntryPoint("z_bad", f"{__name__}:_BrokenChunker", EP_RAG_CHUNKERS),
        ]
    )

    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: run_a)
    first = register_rag_chunker_entry_points(
        ChunkingStrategyRegistry(),
        discover_entry_points=True,
    )
    reset_entry_point_spec_cache_for_tests()
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: run_b)
    second = register_rag_chunker_entry_points(
        ChunkingStrategyRegistry(),
        discover_entry_points=True,
    )

    assert first.rejected == second.rejected


def test_collect_rag_evidence_discovery_disabled_returns_empty_reports() -> None:
    evidence = collect_rag_plugin_load_evidence(discover_entry_points=False)

    assert evidence.chunker_report.accepted == ()
    assert evidence.retriever_report.accepted == ()
    assert evidence.reranker_report.accepted == ()
    assert evidence.chunker_report.group == EP_RAG_CHUNKERS
    assert evidence.retriever_report.group == EP_RAG_RETRIEVERS


def test_rag_load_failure_isolated_in_failed_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = _EntryPoints(
        [_EntryPoint("broken_load", "not-a-valid-target", EP_RAG_CHUNKERS)]
    )
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: entries)

    report = register_rag_chunker_entry_points(
        ChunkingStrategyRegistry(),
        discover_entry_points=True,
    )

    assert report.rejected == ()
    assert len(report.failed) == 1
    assert report.failed[0].error is not None


class _BrokenChunkerConstructor(BaseChunkingStrategy):
    def strategy_id(self) -> str:
        return "broken_ctor"

    def chunk(self, document):  # type: ignore[no-untyped-def]
        return []


def test_rag_chunker_constructor_failure_is_failed_not_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_on_init(self) -> None:
        raise RuntimeError("ctor boom")

    monkeypatch.setattr(_BrokenChunkerConstructor, "__init__", _raise_on_init)
    entries = _EntryPoints(
        [
            _EntryPoint(
                "broken_ctor",
                f"{__name__}:_BrokenChunkerConstructor",
                EP_RAG_CHUNKERS,
            ),
        ]
    )
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: entries)

    report = register_rag_chunker_entry_points(
        ChunkingStrategyRegistry(),
        discover_entry_points=True,
    )

    assert report.rejected == ()
    assert len(report.failed) == 1


class _ValidRetriever(BaseRetriever):
    @classmethod
    def name(cls) -> str:
        return "valid_retriever"

    def retrieve(self, query: RetrieverQuery) -> list[RetrievalHit]:
        return []


class _BoomRetrieverPlugin(BaseRetrieverPlugin):
    @classmethod
    def create(cls, **kwargs: object) -> BaseRetriever:
        raise RuntimeError("boom")

    @classmethod
    def name(cls) -> str:
        return "boom_retriever"


class _WrongReturnRetrieverPlugin(BaseRetrieverPlugin):
    @classmethod
    def create(cls, **kwargs: object) -> object:
        return object()

    @classmethod
    def name(cls) -> str:
        return "wrong_return"


class _ValidRetrieverPlugin(BaseRetrieverPlugin):
    factory_calls = 0

    @classmethod
    def create(cls, **kwargs: object) -> BaseRetriever:
        cls.factory_calls += 1
        return _ValidRetriever()

    @classmethod
    def name(cls) -> str:
        return "valid_plugin"


def _retriever_bootstrap_report(
    monkeypatch: pytest.MonkeyPatch,
    entries: list[_EntryPoint],
) -> object:
    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda: _EntryPoints(entries),
    )
    vector_store = create_default_vectorstore_manager(tenant_id="tenant-rag-ep-test")
    embedding_manager = create_default_embedding_manager()
    return register_rag_retriever_entry_points(
        RetrieverRegistry(),
        vector_store=vector_store,
        embedding_manager=embedding_manager,
        toc_vector_store=None,
        graph_store=None,
        profile=None,
        llm_for_query_expansion=None,
        discover_entry_points=True,
    )


def test_rag_retriever_invalid_target_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _retriever_bootstrap_report(
        monkeypatch,
        [_EntryPoint("broken", f"{__name__}:_BrokenChunker", EP_RAG_RETRIEVERS)],
    )

    assert report.failed == ()
    assert len(report.rejected) == 1
    assert report.rejected[0].reason_code == PluginAdmissionReasonCode.INVALID_TARGET_TYPE


def test_rag_retriever_wrong_factory_return_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _retriever_bootstrap_report(
        monkeypatch,
        [
            _EntryPoint(
                "wrong",
                f"{__name__}:_WrongReturnRetrieverPlugin",
                EP_RAG_RETRIEVERS,
            ),
        ],
    )

    assert report.failed == ()
    assert len(report.rejected) == 1
    assert report.rejected[0].reason_code == PluginAdmissionReasonCode.INVALID_TARGET_TYPE


def test_rag_retriever_factory_exception_is_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _retriever_bootstrap_report(
        monkeypatch,
        [
            _EntryPoint(
                "boom",
                f"{__name__}:_BoomRetrieverPlugin",
                EP_RAG_RETRIEVERS,
            ),
        ],
    )

    assert report.rejected == ()
    assert len(report.failed) == 1


def test_rag_retriever_valid_sibling_survives_factory_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ValidRetrieverPlugin.factory_calls = 0
    report = _retriever_bootstrap_report(
        monkeypatch,
        [
            _EntryPoint(
                "valid",
                f"{__name__}:_ValidRetrieverPlugin",
                EP_RAG_RETRIEVERS,
            ),
            _EntryPoint(
                "boom",
                f"{__name__}:_BoomRetrieverPlugin",
                EP_RAG_RETRIEVERS,
            ),
        ],
    )

    assert report.registered_count == 1
    assert len(report.failed) == 1
    assert report.rejected == ()
    assert _ValidRetrieverPlugin.factory_calls == 1


class _BoomRerankerPlugin(BaseRerankerPlugin):
    @classmethod
    def create(cls, *, embedding_manager: object) -> BaseReranker:
        raise RuntimeError("boom")

    @classmethod
    def name(cls) -> str:
        return "boom_reranker"


def test_rag_reranker_factory_exception_is_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = _EntryPoints(
        [
            _EntryPoint(
                "boom",
                f"{__name__}:_BoomRerankerPlugin",
                EP_RAG_RERANKERS,
            ),
        ]
    )
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: entries)
    report = register_rag_reranker_entry_points(
        RerankerRegistry(),
        embedding_manager=create_default_embedding_manager(),
        discover_entry_points=True,
    )

    assert report.rejected == ()
    assert len(report.failed) == 1
