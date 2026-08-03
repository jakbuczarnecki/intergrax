# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import pytest
from pathlib import Path
from typing import Sequence

from intergrax.knowledge.contracts import KnowledgeDocument, KnowledgeDocumentScope
from intergrax.rag.document_loaders.bootstrap.default_loader import create_default_normalizer_pipeline
from intergrax.rag.document_loaders.documents_loader import DocumentsLoader
from intergrax.rag.document_loaders.registry.document_handler_registry import (
    DocumentHandlerRegistry,
)


pytestmark = pytest.mark.unit

_TENANT = "tenant.test"


class _DummyMetadataPipeline:

    def enrich(self, docs, source):
        return docs


class _DummyHandler:

    def supports(self, source: str) -> bool:
        return True

    def confidence(self, source: str) -> float:
        return 1.0

    def load(self, source: str, *, scope: KnowledgeDocumentScope) -> Sequence[KnowledgeDocument]:
        return [
            KnowledgeDocument.model_validate(
                {
                    "schema_version": 1,
                    "identity": {
                        "document_id": "docid1234567890ab",
                        "root_document_id": "docid1234567890ab",
                    },
                    "scope": {
                        "tenant_id": scope.tenant_id,
                        "namespace": scope.namespace,
                    },
                    "content": source,
                    "metadata": {
                        "source": source,
                        "parser": "tests.dummy",
                        "position": 0,
                    },
                    "provenance": {
                        "source_kind": "file",
                        "source_id": source,
                        "provider_id": "tests.dummy",
                    },
                }
            )
        ]

    def build_parsers(self):
        return []


def _build_loader():
    registry = DocumentHandlerRegistry()
    registry.register(_DummyHandler())

    metadata_pipeline = _DummyMetadataPipeline()
    normalizer_pipeline = create_default_normalizer_pipeline()

    return DocumentsLoader(
        registry=registry,
        metadata_pipeline=metadata_pipeline,
        normalizer_pipeline=normalizer_pipeline
    )


def test_loader_returns_empty_when_directory_missing(tmp_path: Path):

    loader = _build_loader()

    missing = tmp_path / "missing"

    docs = loader.load_documents(str(missing), tenant_id=_TENANT)

    assert docs == []


def test_loader_respects_allowed_extensions(tmp_path: Path):

    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.pdf").write_text("b")

    registry = DocumentHandlerRegistry()
    metadata_pipeline = _DummyMetadataPipeline()
    normalizer_pipeline = create_default_normalizer_pipeline()


    loader = DocumentsLoader(
        registry=registry,
        metadata_pipeline=metadata_pipeline,
        normalizer_pipeline=normalizer_pipeline,
        allowed_exts=[".txt"],
    )

    handler = _DummyHandler()
    loader._registry.register(handler)

    docs = loader.load_documents(str(tmp_path), tenant_id=_TENANT)

    assert len(docs) == 1
    assert docs[0].content.endswith("a.txt")


def test_loader_respects_max_files(tmp_path: Path):

    normalizer_pipeline = create_default_normalizer_pipeline()
    metadata_pipeline = _DummyMetadataPipeline()
    registry = DocumentHandlerRegistry()

    for i in range(5):
        (tmp_path / f"f{i}.txt").write_text("x")

    loader = DocumentsLoader(
        registry=registry,
        metadata_pipeline=metadata_pipeline,
        normalizer_pipeline=normalizer_pipeline,
        max_files=2,
    )

    loader._registry.register(_DummyHandler())

    docs = loader.load_documents(str(tmp_path), tenant_id=_TENANT)

    assert len(docs) == 2


def test_loader_calls_load_document_for_each_file(tmp_path: Path):

    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.txt").write_text("b")

    loader = _build_loader()

    docs = loader.load_documents(str(tmp_path), tenant_id=_TENANT)

    assert len(docs) == 2
