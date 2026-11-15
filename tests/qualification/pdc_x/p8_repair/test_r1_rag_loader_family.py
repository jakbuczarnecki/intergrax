# © Artur Czarnecki. All rights reserved.

"""T-IGX-041-M / R1 (G-01): the RAG loader family builds, shares one runtime document class, lists collections."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from intergrax.knowledge.contracts import KnowledgeDocument

pytestmark = [pytest.mark.unit, pytest.mark.gate]

REPO_ROOT = Path(__file__).resolve().parents[4]
HANDLERS_DIR = REPO_ROOT / "intergrax" / "rag" / "document_loaders" / "handlers"
PIPELINE_MODULE = "intergrax.rag.document_loaders.pipeline.document_handler_loading"
RUNTIME_CLASS = "_KnowledgeDocumentWithParserRuntime"


def _wrong_pipeline_handler_imports(source: str) -> list[str]:
    """Imports of ``PipelineDocumentHandler`` that do not come from the pipeline module."""
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and any(alias.name == "PipelineDocumentHandler" for alias in node.names):
            if node.module != PIPELINE_MODULE:
                found.append(str(node.module))
    return found


def _sample_document() -> KnowledgeDocument:
    return KnowledgeDocument.model_validate(
        {
            "schema_version": 1,
            "identity": {"document_id": "docid1234567890ab", "root_document_id": "docid1234567890ab"},
            "scope": {"tenant_id": "t-r1"},
            "content": "content",
            "metadata": {"source": "file.pdf", "parser": "tests.dummy", "position": 0},
            "provenance": {"source_kind": "file", "source_id": "file.pdf", "provider_id": "tests.dummy"},
        }
    )


def test_doc_smart_handler_imports_and_is_a_pipeline_handler() -> None:
    from intergrax.rag.document_loaders.handlers.doc_smart_document_handler import DocSmartDocumentHandler
    from intergrax.rag.document_loaders.pipeline.document_handler_loading import PipelineDocumentHandler

    assert issubclass(DocSmartDocumentHandler, PipelineDocumentHandler)


def test_default_documents_loader_builds_with_all_eight_handlers() -> None:
    from intergrax.rag.document_loaders.bootstrap.default_loader import create_default_documents_loader
    from intergrax.rag.document_loaders.documents_loader import DocumentsLoader

    loader = create_default_documents_loader()

    assert isinstance(loader, DocumentsLoader)
    handlers = loader._registry._handlers  # type: ignore[attr-defined]
    assert sorted(type(handler).__name__ for handler in handlers) == [
        "AudioSmartDocumentHandler",
        "DocSmartDocumentHandler",
        "ExcelSmartDocumentHandler",
        "HtmlSmartDocumentHandler",
        "ImageSmartDocumentHandler",
        "PdfSmartDocumentHandler",
        "TextSmartDocumentHandler",
        "VideoSmartDocumentHandler",
    ]


def test_every_handler_imports_pipeline_document_handler_from_the_pipeline_module() -> None:
    offenders = {
        path.name: _wrong_pipeline_handler_imports(path.read_text(encoding="utf-8"))
        for path in sorted(HANDLERS_DIR.glob("*.py"))
    }
    assert {name: wrong for name, wrong in offenders.items() if wrong} == {}


def test_the_import_gate_detects_the_old_path() -> None:
    wrong_module = "intergrax.rag.document_loaders.contracts.base_document_handler"
    old = f"from {wrong_module} import (\n    PipelineDocumentHandler,\n)\n"
    assert _wrong_pipeline_handler_imports(old) == [wrong_module]
    assert _wrong_pipeline_handler_imports(f"from {PIPELINE_MODULE} import PipelineDocumentHandler\n") == []


def test_exactly_one_parser_runtime_document_class_is_defined() -> None:
    definitions: list[str] = []
    for path in sorted((REPO_ROOT / "intergrax").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if RUNTIME_CLASS not in text:
            continue
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.ClassDef) and node.name == RUNTIME_CLASS:
                definitions.append(path.relative_to(REPO_ROOT).as_posix())
    assert len(definitions) == 1, definitions


def test_a_native_handle_attached_by_either_module_is_read_by_both() -> None:
    from intergrax.rag.document_loaders.compat import legacy_runtime_document as compat
    from intergrax.rag.document_loaders.contracts import parser_native_handle as contract

    handle = {"doc": 1}
    for attach in (contract.attach_parser_native_handle, compat.attach_parser_native_handle):
        attached = attach(_sample_document(), handle)
        assert compat.get_parser_native_handle(attached) is handle
        assert contract.get_parser_native_handle(attached) is handle
    assert compat.get_parser_native_handle(_sample_document()) is None


def test_inmemory_vector_store_integration_lists_the_inner_collections() -> None:
    from intergrax.integrations.contracts.base import IntegrationConfigurationError
    from intergrax.integrations.providers.vector_store.inmemory.integration import InmemoryVectorStoreIntegration
    from intergrax.integrations.providers.vector_store.inmemory.rag_store import InMemoryVectorStore

    inner = InMemoryVectorStore("t-r1")
    integration = InmemoryVectorStoreIntegration.from_store(object(), inner)

    assert integration.list_collections() == inner.list_collections() == ["inmemory:t-r1"]

    empty = InmemoryVectorStoreIntegration.from_client(object())  # type: ignore[arg-type]
    with pytest.raises(IntegrationConfigurationError):
        empty.list_collections()
