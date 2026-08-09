# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import builtins
import sys
import types

import numpy as np
import pytest

from intergrax.integrations.providers.document_parser.pymupdf.config import PymupdfIntegrationConfig
from intergrax.integrations.providers.document_parser.python_docx.config import PythonDocxIntegrationConfig
from intergrax.rag.embedding.providers.ollama_embedding_provider import OllamaEmbeddingProvider


def _block_import(monkeypatch: pytest.MonkeyPatch, root: str, *, error_name: str | None = None) -> None:
    real_import = builtins.__import__

    def blocked(
        name: str,
        globals: object | None = None,
        locals: object | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == root or name.startswith(f"{root}."):
            raise ModuleNotFoundError("blocked optional dependency", name=error_name or root)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked)
    for name in tuple(sys.modules):
        if name == root or name.startswith(f"{root}."):
            monkeypatch.delitem(sys.modules, name, raising=False)


@pytest.mark.parametrize(
    ("module_name", "call"),
    [
        (
            "intergrax.integrations.providers.document_parser.pymupdf.opens",
            lambda: __import__(
                "intergrax.integrations.providers.document_parser.pymupdf.opens",
                fromlist=["parse_pymupdf_file"],
            ).parse_pymupdf_file(PymupdfIntegrationConfig(), "sample.pdf"),
        ),
        (
            "intergrax.integrations.providers.document_parser.python_docx.opens",
            lambda: __import__(
                "intergrax.integrations.providers.document_parser.python_docx.opens",
                fromlist=["parse_python_docx_file"],
            ).parse_python_docx_file(
                PythonDocxIntegrationConfig(strategy="fulltext"),
                "sample.docx",
            ),
        ),
        (
            "intergrax.integrations.providers.document_parser.unstructured.opens",
            lambda: __import__(
                "intergrax.integrations.providers.document_parser.unstructured.opens",
                fromlist=["parse_unstructured_html"],
            ).parse_unstructured_html("sample.html"),
        ),
    ],
)
def test_missing_loader_extra_has_stable_error(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    call: object,
) -> None:
    _block_import(monkeypatch, "langchain_community")
    with pytest.raises(RuntimeError, match="rag-langchain-loaders"):
        call()


def test_openpyxl_missing_extra_has_stable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from intergrax.integrations.providers.document_parser.openpyxl import opens

    _block_import(monkeypatch, "langchain_core")
    loader = opens._ExcelLoader(
        "sample.csv",
        mode="rows",
        header=0,
        sheet=None,
        na_filter=True,
        max_rows_per_sheet=None,
        encoding=None,
        delimiter=None,
    )
    loader._read_csv_like = lambda: {}
    with pytest.raises(RuntimeError, match="rag-langchain-loaders"):
        loader.load()


def test_openpyxl_provider_imports_with_mocked_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    documents_module = types.ModuleType("langchain_core.documents")

    class FakeDocument:
        def __init__(self, *, page_content: str, metadata: dict[str, object]) -> None:
            self.page_content = page_content
            self.metadata = metadata

    documents_module.Document = FakeDocument
    core_package = types.ModuleType("langchain_core")
    core_package.__path__ = []
    monkeypatch.setitem(sys.modules, "langchain_core", core_package)
    monkeypatch.setitem(sys.modules, "langchain_core.documents", documents_module)

    from intergrax.integrations.providers.document_parser.openpyxl import opens

    loader = opens._ExcelLoader(
        "sample.csv",
        mode="rows",
        header=0,
        sheet=None,
        na_filter=True,
        max_rows_per_sheet=None,
        encoding=None,
        delimiter=None,
    )
    loader._read_csv_like = lambda: {}
    assert loader.load() == []


def test_loader_import_error_inside_installed_provider_is_not_masked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = __import__(
        "intergrax.integrations.providers.document_parser.unstructured.opens",
        fromlist=["parse_unstructured_html"],
    )
    _block_import(
        monkeypatch,
        "langchain_community",
        error_name="unstructured_runtime_dependency",
    )
    with pytest.raises(ModuleNotFoundError, match="blocked optional dependency"):
        module.parse_unstructured_html("sample.html")


def test_loader_provider_keeps_fragment_output_with_mocked_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader_module = types.ModuleType("langchain_community.document_loaders")

    class FakeDocument:
        page_content = "body"
        metadata = {"source": "sample.html"}

    class FakeLoader:
        def __init__(self, source: str) -> None:
            self.source = source

        def load(self) -> list[FakeDocument]:
            return [FakeDocument()]

    loader_module.UnstructuredHTMLLoader = FakeLoader
    package = types.ModuleType("langchain_community")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "langchain_community", package)
    monkeypatch.setitem(sys.modules, "langchain_community.document_loaders", loader_module)

    from intergrax.integrations.providers.document_parser.unstructured.opens import (
        parse_unstructured_html,
    )

    fragments = parse_unstructured_html("sample.html")
    assert fragments[0].text == "body"
    assert fragments[0].metadata == {"parser_backend": "unstructured", "source": "sample.html"}


def test_pymupdf_provider_keeps_fragment_output_with_mocked_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader_module = types.ModuleType("langchain_community.document_loaders")

    class FakeDocument:
        page_content = "page"
        metadata = {"page": 0}

    class FakeLoader:
        def __init__(self, source: str) -> None:
            self.source = source

        def load(self) -> list[FakeDocument]:
            return [FakeDocument()]

    loader_module.PyMuPDFLoader = FakeLoader
    package = types.ModuleType("langchain_community")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "langchain_community", package)
    monkeypatch.setitem(sys.modules, "langchain_community.document_loaders", loader_module)

    from intergrax.integrations.providers.document_parser.pymupdf.opens import (
        parse_pymupdf_file,
    )

    fragments = parse_pymupdf_file(PymupdfIntegrationConfig(), "sample.pdf")
    assert fragments[0].text == "page"
    assert fragments[0].metadata == {"parser_backend": "pymupdf", "page": 0}


def test_python_docx_provider_keeps_fragment_output_with_mocked_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader_module = types.ModuleType("langchain_community.document_loaders")

    class FakeDocument:
        page_content = "docx text"
        metadata = {"source": "sample.docx"}

    class FakeLoader:
        def __init__(self, source: str) -> None:
            self.source = source

        def load(self) -> list[FakeDocument]:
            return [FakeDocument()]

    loader_module.Docx2txtLoader = FakeLoader
    package = types.ModuleType("langchain_community")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "langchain_community", package)
    monkeypatch.setitem(sys.modules, "langchain_community.document_loaders", loader_module)

    from intergrax.integrations.providers.document_parser.python_docx.opens import (
        parse_python_docx_file,
    )

    fragments = parse_python_docx_file(
        PythonDocxIntegrationConfig(strategy="fulltext"),
        "sample.docx",
    )
    assert fragments[0].text == "docx text"
    assert fragments[0].metadata == {"parser_backend": "python_docx", "source": "sample.docx"}


def test_ollama_embedding_provider_missing_extra_has_stable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _block_import(monkeypatch, "langchain_ollama")
    provider = OllamaEmbeddingProvider(model_name="test-model")

    with pytest.raises(RuntimeError, match="rag-langchain-embeddings"):
        provider.dimension()


def test_ollama_embedding_provider_keeps_abi_with_mocked_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = types.ModuleType("langchain_ollama")

    class FakeEmbeddings:
        def __init__(self, *, model: str) -> None:
            self.model = model

        def embed_query(self, text: str) -> list[float]:
            assert text == "probe-dimension"
            return [1.0, 2.0]

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[float(index), 2.0] for index, _ in enumerate(texts)]

    module.OllamaEmbeddings = FakeEmbeddings
    monkeypatch.setitem(sys.modules, "langchain_ollama", module)

    provider = OllamaEmbeddingProvider(model_name="test-model")
    assert provider.provider_name() == "ollama"
    assert provider.dimension() == 2
    assert provider.embed(["one", "two"]).dtype == np.float32
    assert provider.embed(["one", "two"]).shape == (2, 2)
