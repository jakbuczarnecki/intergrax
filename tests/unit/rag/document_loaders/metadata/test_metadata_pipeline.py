# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import pytest
from pathlib import Path
from typing import Sequence

from pydantic import ValidationError

from intergrax.knowledge.contracts import KnowledgeDocument, KnowledgeDocumentScope
from intergrax.rag.document_loaders.compat.legacy_runtime_document import (
    attach_parser_native_handle,
)
from intergrax.rag.document_loaders.pipeline.metadata_pipeline import MetadataPipeline
from intergrax.rag.document_loaders.contracts.metadata_provider import BaseMetadataProvider


pytestmark = pytest.mark.unit

_TENANT = "tenant.test"


def _sample_doc(**metadata) -> KnowledgeDocument:
    return KnowledgeDocument.model_validate(
        {
            "schema_version": 1,
            "identity": {
                "document_id": "docid1234567890ab",
                "root_document_id": "docid1234567890ab",
            },
            "scope": {"tenant_id": _TENANT},
            "content": "content",
            "metadata": {
                "source": "file.pdf",
                "parser": "tests.dummy",
                "position": 0,
                **metadata,
            },
            "provenance": {
                "source_kind": "file",
                "source_id": "file.pdf",
                "provider_id": "tests.dummy",
            },
        }
    )


class _TagProvider(BaseMetadataProvider):

    def __init__(self, tag: str):
        self.tag = tag

    def enrich(
        self,
        documents: Sequence[KnowledgeDocument],
        source: Path | str,
    ) -> Sequence[KnowledgeDocument]:
        out = []
        for doc in documents:
            payload = doc.model_dump(mode="python")
            payload["metadata"] = {**dict(doc.metadata), self.tag: True}
            out.append(KnowledgeDocument.model_validate(payload))
        return out


class _ContentMutatingProvider(BaseMetadataProvider):

    def enrich(
        self,
        documents: Sequence[KnowledgeDocument],
        source: Path | str,
    ) -> Sequence[KnowledgeDocument]:
        out = []
        for doc in documents:
            payload = doc.model_dump(mode="python")
            payload["content"] = "changed"
            out.append(KnowledgeDocument.model_validate(payload))
        return out


class _IdentityMutatingProvider(BaseMetadataProvider):

    def enrich(
        self,
        documents: Sequence[KnowledgeDocument],
        source: Path | str,
    ) -> Sequence[KnowledgeDocument]:
        out = []
        for doc in documents:
            payload = doc.model_dump(mode="python")
            payload["identity"] = {
                **payload["identity"],
                "document_id": "docid99999999999999",
            }
            out.append(KnowledgeDocument.model_validate(payload))
        return out


class _ScopeMutatingProvider(BaseMetadataProvider):

    def enrich(
        self,
        documents: Sequence[KnowledgeDocument],
        source: Path | str,
    ) -> Sequence[KnowledgeDocument]:
        out = []
        for doc in documents:
            payload = doc.model_dump(mode="python")
            payload["scope"] = KnowledgeDocumentScope(
                tenant_id="other.tenant"
            ).model_dump()
            out.append(KnowledgeDocument.model_validate(payload))
        return out


class _ProvenanceMutatingProvider(BaseMetadataProvider):

    def enrich(
        self,
        documents: Sequence[KnowledgeDocument],
        source: Path | str,
    ) -> Sequence[KnowledgeDocument]:
        out = []
        for doc in documents:
            payload = doc.model_dump(mode="python")
            payload["provenance"] = {
                **payload["provenance"],
                "source_id": "changed.pdf",
            }
            out.append(KnowledgeDocument.model_validate(payload))
        return out


class _ReservedMetadataProvider(BaseMetadataProvider):

    def enrich(
        self,
        documents: Sequence[KnowledgeDocument],
        source: Path | str,
    ) -> Sequence[KnowledgeDocument]:
        out = []
        for doc in documents:
            payload = doc.model_dump(mode="python")
            payload["metadata"] = {**dict(doc.metadata), "tenant_id": "override"}
            out.append(KnowledgeDocument.model_validate(payload))
        return out


class _CountChangingProvider(BaseMetadataProvider):

    def enrich(
        self,
        documents: Sequence[KnowledgeDocument],
        source: Path | str,
    ) -> Sequence[KnowledgeDocument]:
        return list(documents) + list(documents)


class _FailingProvider(BaseMetadataProvider):

    def enrich(
        self,
        documents: Sequence[KnowledgeDocument],
        source: Path | str,
    ) -> Sequence[KnowledgeDocument]:
        raise RuntimeError("provider failure")


def test_pipeline_executes_providers_in_sequence():

    docs = [_sample_doc()]

    pipeline = MetadataPipeline([_TagProvider("p1"), _TagProvider("p2")])

    result = pipeline.enrich(docs, "file.txt")

    assert result[0].metadata["p1"] is True
    assert result[0].metadata["p2"] is True


def test_pipeline_output_of_one_is_input_of_next():

    docs = [_sample_doc()]

    pipeline = MetadataPipeline([_TagProvider("first"), _TagProvider("second")])

    result = pipeline.enrich(docs, "file.txt")

    assert result[0].metadata["first"] is True
    assert result[0].metadata["second"] is True


def test_pipeline_preserves_parser_runtime_handle():

    handle = object()
    docs = [attach_parser_native_handle(_sample_doc(), handle)]

    pipeline = MetadataPipeline([_TagProvider("tag")])

    result = pipeline.enrich(docs, "file.txt")

    from intergrax.rag.document_loaders.compat.legacy_runtime_document import (
        get_parser_native_handle,
    )

    assert get_parser_native_handle(result[0]) is handle


def test_pipeline_rejects_content_change():

    pipeline = MetadataPipeline([_ContentMutatingProvider()])

    with pytest.raises(ValueError, match="content"):
        pipeline.enrich([_sample_doc()], "file.txt")


def test_pipeline_rejects_identity_change():

    pipeline = MetadataPipeline([_IdentityMutatingProvider()])

    with pytest.raises(ValueError, match="identity"):
        pipeline.enrich([_sample_doc()], "file.txt")


def test_pipeline_rejects_scope_change():

    pipeline = MetadataPipeline([_ScopeMutatingProvider()])

    with pytest.raises(ValueError, match="scope"):
        pipeline.enrich([_sample_doc()], "file.txt")


def test_pipeline_rejects_provenance_change():

    pipeline = MetadataPipeline([_ProvenanceMutatingProvider()])

    with pytest.raises(ValueError, match="provenance"):
        pipeline.enrich([_sample_doc()], "file.txt")


def test_pipeline_rejects_document_count_change():

    pipeline = MetadataPipeline([_CountChangingProvider()])

    with pytest.raises(ValueError, match="document count"):
        pipeline.enrich([_sample_doc()], "file.txt")


def test_pipeline_rejects_reserved_metadata():

    pipeline = MetadataPipeline([_ReservedMetadataProvider()])

    with pytest.raises(ValidationError):
        pipeline.enrich([_sample_doc()], "file.txt")


def test_pipeline_propagates_provider_exception():

    pipeline = MetadataPipeline([_FailingProvider()])

    with pytest.raises(RuntimeError, match="provider failure"):
        pipeline.enrich([_sample_doc()], "file.txt")


def test_pipeline_returns_input_when_no_providers():

    docs = [_sample_doc()]

    pipeline = MetadataPipeline([])

    result = pipeline.enrich(docs, "file.txt")

    assert result == docs


def test_pipeline_passes_source_to_providers():

    captured = {}

    class _SourceCaptureProvider(BaseMetadataProvider):

        def enrich(self, documents, source):
            captured["source"] = source
            return documents

    pipeline = MetadataPipeline([_SourceCaptureProvider()])

    pipeline.enrich([_sample_doc()], "abc.txt")

    assert captured["source"] == "abc.txt"
