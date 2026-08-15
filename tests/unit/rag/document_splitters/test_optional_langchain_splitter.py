# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import builtins
import importlib
import sys
import types

import pytest

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.document_splitters.registry.strategy_registry import ChunkingStrategyRegistry
from intergrax.rag.document_splitters.strategies.langchain_recursive_chunking_strategy import (
    LangChainRecursiveChunkingStrategy,
)
from intergrax.rag.profiles.rag_profile import RagProfile


pytestmark = pytest.mark.unit


def _source(document_id: str, content: str) -> KnowledgeDocument:
    return KnowledgeDocument.model_validate(
        {
            "schema_version": 1,
            "identity": {
                "document_id": document_id,
                "root_document_id": document_id,
            },
            "scope": {"tenant_id": "tenant.test"},
            "content": content,
            "metadata": {"source": document_id},
            "provenance": {
                "source_kind": "file",
                "source_id": document_id,
            },
        }
    )


def _block_langchain_text_splitters(
    real_import: object,
    *,
    error_name: str = "langchain_text_splitters",
):
    def _blocked_import(
        name: str,
        globals: object | None = None,
        locals: object | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "langchain_text_splitters":
            raise ModuleNotFoundError("blocked optional dependency", name=error_name)
        return real_import(name, globals, locals, fromlist, level)  # type: ignore[operator]

    return _blocked_import


def _install_fake_embedding_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = types.ModuleType(
        "intergrax.rag.embedding.bootstrap.default_embedding_engine"
    )

    def _create_default_embedding_manager() -> object:
        return object()

    fake_module.create_default_embedding_manager = _create_default_embedding_manager
    monkeypatch.setitem(
        sys.modules,
        "intergrax.rag.embedding.bootstrap.default_embedding_engine",
        fake_module,
    )


def test_default_bootstrap_is_core_safe_without_optional_splitter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bootstrap_module_name = (
        "intergrax.rag.document_splitters.bootstrap.default_chunking_engine"
    )
    strategy_module_name = (
        "intergrax.rag.document_splitters.strategies.langchain_recursive_chunking_strategy"
    )
    sys.modules.pop(bootstrap_module_name, None)
    sys.modules.pop(strategy_module_name, None)
    _install_fake_embedding_bootstrap(monkeypatch)

    real_import = builtins.__import__
    monkeypatch.setattr(
        builtins,
        "__import__",
        _block_langchain_text_splitters(real_import),
    )

    bootstrap = importlib.import_module(bootstrap_module_name)
    splitter = bootstrap.create_default_document_splitter()
    source = _source("doc-default", "abcdefghij" * 20)
    profile = RagProfile()

    chunks = splitter.split_documents([source], strategy_id=profile.chunking_strategy_id)

    assert chunks
    assert profile.chunking_strategy_id == "recursive"
    assert all(chunk.metadata["chunk_strategy"] == "recursive" for chunk in chunks)
    with pytest.raises(RuntimeError, match="Chunking strategy not registered"):
        splitter.split_documents([source], strategy_id="langchain_recursive")


def test_missing_optional_splitter_has_stable_configuration_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__
    monkeypatch.setattr(
        builtins,
        "__import__",
        _block_langchain_text_splitters(real_import),
    )

    with pytest.raises(RuntimeError) as exc_info:
        LangChainRecursiveChunkingStrategy()

    message = str(exc_info.value)
    assert "rag-langchain-splitters" in message
    assert "Intergrax-ai[rag-langchain-splitters]" in message


def test_internal_missing_module_error_is_not_masked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__
    monkeypatch.setattr(
        builtins,
        "__import__",
        _block_langchain_text_splitters(
            real_import,
            error_name="langchain_internal_dependency",
        ),
    )

    with pytest.raises(ModuleNotFoundError) as exc_info:
        LangChainRecursiveChunkingStrategy()

    assert exc_info.value.name == "langchain_internal_dependency"


def test_optional_splitter_can_be_explicitly_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeSplitter:
        def __init__(self, *, chunk_size: int, chunk_overlap: int) -> None:
            del chunk_overlap
            self._chunk_size = chunk_size

        def split_text(self, text: str) -> list[str]:
            return [
                text[index : index + self._chunk_size]
                for index in range(0, len(text), self._chunk_size)
            ]

    fake_module = types.ModuleType("langchain_text_splitters")
    fake_module.RecursiveCharacterTextSplitter = _FakeSplitter
    monkeypatch.setitem(sys.modules, "langchain_text_splitters", fake_module)

    strategy = LangChainRecursiveChunkingStrategy(chunk_size=10, chunk_overlap=2)
    assert strategy.strategy_id() == "langchain_recursive"

    registry = ChunkingStrategyRegistry()
    registry.register(strategy)
    _install_fake_embedding_bootstrap(monkeypatch)
    from intergrax.rag.document_splitters.bootstrap.default_chunking_engine import (
        create_default_chunking_engine,
    )

    engine = create_default_chunking_engine(registry=registry)
    source = _source("doc-explicit", "abcdefghij" * 3)

    chunks = engine.chunk([source], strategy_id="langchain_recursive")
    repeated_chunks = engine.chunk([source], strategy_id="langchain_recursive")

    assert chunks
    assert all(isinstance(chunk, KnowledgeDocument) for chunk in chunks)
    assert all(
        chunk.identity.parent_document_id == source.identity.document_id
        and chunk.identity.root_document_id == source.identity.root_document_id
        and chunk.scope == source.scope
        for chunk in chunks
    )
    assert [chunk.identity.document_id for chunk in chunks] == [
        chunk.identity.document_id for chunk in repeated_chunks
    ]
