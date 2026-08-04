# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from pathlib import Path
from typing import Sequence

import pytest

from intergrax.knowledge.contracts import KnowledgeDocument, KnowledgeDocumentScope
from intergrax.rag.document_loaders.compat.legacy_runtime_document import (
    attach_parser_native_handle,
)
from intergrax.rag.document_loaders.pipeline.normalizer_pipeline import NormalizerPipeline
from intergrax.rag.document_loaders.contracts.base_document_normalizer import BaseDocumentNormalizer


pytestmark = pytest.mark.unit

_TENANT = "tenant.test"


def _sample_doc(content: str = "text", **metadata) -> KnowledgeDocument:
    return KnowledgeDocument.model_validate(
        {
            "schema_version": 1,
            "identity": {
                "document_id": "docid1234567890ab",
                "root_document_id": "docid1234567890ab",
            },
            "scope": {"tenant_id": _TENANT},
            "content": content,
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


class _TagNormalizer(BaseDocumentNormalizer):

    def __init__(self, tag: str):
        self.tag = tag

    def normalize(
        self,
        documents: Sequence[KnowledgeDocument],
        source: Path | str,
    ) -> Sequence[KnowledgeDocument]:
        result = []
        for doc in documents:
            payload = doc.model_dump(mode="python")
            payload["content"] = f"{doc.content}:{self.tag}"
            result.append(KnowledgeDocument.model_validate(payload))
        return result


class _MetadataMutatingNormalizer(BaseDocumentNormalizer):

    def normalize(
        self,
        documents: Sequence[KnowledgeDocument],
        source: Path | str,
    ) -> Sequence[KnowledgeDocument]:
        result = []
        for doc in documents:
            payload = doc.model_dump(mode="python")
            payload["metadata"] = {**dict(doc.metadata), "mutated": True}
            result.append(KnowledgeDocument.model_validate(payload))
        return result


class _IdentityMutatingNormalizer(BaseDocumentNormalizer):

    def normalize(
        self,
        documents: Sequence[KnowledgeDocument],
        source: Path | str,
    ) -> Sequence[KnowledgeDocument]:
        result = []
        for doc in documents:
            payload = doc.model_dump(mode="python")
            payload["identity"] = {
                **payload["identity"],
                "document_id": "docid99999999999999",
            }
            result.append(KnowledgeDocument.model_validate(payload))
        return result


class _ScopeMutatingNormalizer(BaseDocumentNormalizer):

    def normalize(
        self,
        documents: Sequence[KnowledgeDocument],
        source: Path | str,
    ) -> Sequence[KnowledgeDocument]:
        result = []
        for doc in documents:
            payload = doc.model_dump(mode="python")
            payload["scope"] = KnowledgeDocumentScope(
                tenant_id="other.tenant"
            ).model_dump()
            result.append(KnowledgeDocument.model_validate(payload))
        return result


class _ProvenanceMutatingNormalizer(BaseDocumentNormalizer):

    def normalize(
        self,
        documents: Sequence[KnowledgeDocument],
        source: Path | str,
    ) -> Sequence[KnowledgeDocument]:
        result = []
        for doc in documents:
            payload = doc.model_dump(mode="python")
            payload["provenance"] = {
                **payload["provenance"],
                "source_id": "changed.pdf",
            }
            result.append(KnowledgeDocument.model_validate(payload))
        return result


class _CountChangingNormalizer(BaseDocumentNormalizer):

    def normalize(
        self,
        documents: Sequence[KnowledgeDocument],
        source: Path | str,
    ) -> Sequence[KnowledgeDocument]:
        return list(documents) + list(documents)


class _FailingNormalizer(BaseDocumentNormalizer):

    def normalize(
        self,
        documents: Sequence[KnowledgeDocument],
        source: Path | str,
    ) -> Sequence[KnowledgeDocument]:
        raise RuntimeError("normalizer failure")


def test_pipeline_executes_normalizers_in_order():

    docs = [_sample_doc()]

    pipeline = NormalizerPipeline(
        normalizers=[
            _TagNormalizer("n1"),
            _TagNormalizer("n2"),
        ]
    )

    result = pipeline.normalize(docs, source="file.txt")

    assert len(result) == 1
    assert result[0].content == "text:n1:n2"


def test_pipeline_output_of_one_is_input_of_next():

    docs = [_sample_doc(content="start")]

    pipeline = NormalizerPipeline(
        normalizers=[
            _TagNormalizer("first"),
            _TagNormalizer("second"),
        ]
    )

    result = pipeline.normalize(docs, source="file.txt")

    assert result[0].content == "start:first:second"


def test_pipeline_preserves_parser_runtime_handle():

    handle = object()
    docs = [attach_parser_native_handle(_sample_doc(), handle)]

    pipeline = NormalizerPipeline(normalizers=[_TagNormalizer("tag")])

    result = pipeline.normalize(docs, source="file.txt")

    from intergrax.rag.document_loaders.compat.legacy_runtime_document import (
        to_legacy_rag_document,
    )
    from intergrax.rag.document_loaders.contracts.document_metadata_key import (
        DocumentMetadataKey,
    )

    legacy = to_legacy_rag_document(result[0])
    assert legacy.metadata[DocumentMetadataKey.DOCLING_DOCUMENT_META.value] is handle


def test_pipeline_rejects_identity_change():

    pipeline = NormalizerPipeline(normalizers=[_IdentityMutatingNormalizer()])

    with pytest.raises(ValueError, match="identity"):
        pipeline.normalize([_sample_doc()], source="file.txt")


def test_pipeline_rejects_scope_change():

    pipeline = NormalizerPipeline(normalizers=[_ScopeMutatingNormalizer()])

    with pytest.raises(ValueError, match="scope"):
        pipeline.normalize([_sample_doc()], source="file.txt")


def test_pipeline_rejects_metadata_change():

    pipeline = NormalizerPipeline(normalizers=[_MetadataMutatingNormalizer()])

    with pytest.raises(ValueError, match="metadata"):
        pipeline.normalize([_sample_doc()], source="file.txt")


def test_pipeline_rejects_provenance_change():

    pipeline = NormalizerPipeline(normalizers=[_ProvenanceMutatingNormalizer()])

    with pytest.raises(ValueError, match="provenance"):
        pipeline.normalize([_sample_doc()], source="file.txt")


def test_pipeline_rejects_document_count_change():

    pipeline = NormalizerPipeline(normalizers=[_CountChangingNormalizer()])

    with pytest.raises(ValueError, match="document count"):
        pipeline.normalize([_sample_doc()], source="file.txt")


def test_pipeline_propagates_normalizer_exception():

    pipeline = NormalizerPipeline(normalizers=[_FailingNormalizer()])

    with pytest.raises(RuntimeError, match="normalizer failure"):
        pipeline.normalize([_sample_doc()], source="file.txt")


def test_pipeline_allows_empty_normalizer_list():

    docs = [_sample_doc(content="text")]

    pipeline = NormalizerPipeline(normalizers=[])

    result = pipeline.normalize(docs, source="file.txt")

    assert result[0].content == "text"
    assert result[0].metadata["source"] == "file.pdf"
