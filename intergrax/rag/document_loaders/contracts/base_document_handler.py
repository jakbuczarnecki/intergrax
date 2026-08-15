# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence
from urllib.parse import urlparse

from intergrax.integrations.contracts.document_parser import ParsedDocumentFragment
from intergrax.knowledge.contracts import (
    KnowledgeDocument,
    KnowledgeDocumentScope,
)
from intergrax.knowledge.contracts.document import RESERVED_METADATA_KEYS, SCHEMA_VERSION
from intergrax.knowledge.contracts.validation import JsonValue
from intergrax.rag.document_loaders.contracts.base_document_parser import BaseDocumentParser
from intergrax.rag.document_loaders.compat.legacy_runtime_document import (
    attach_parser_native_handle,
)
from intergrax.rag.document_loaders.contracts.document_metadata_key import DocumentMetadataKey
from intergrax.rag.document_loaders.pipeline.parser_pipeline import ParserPipeline


def _resolve_source_kind(source: str) -> str:
    if len(source) >= 2 and source[1] == ":" and source[0].isalpha():
        return "file"
    parsed = urlparse(source)
    if parsed.scheme:
        return parsed.scheme.lower()
    return "file"


def _fragment_to_knowledge_document(
    fragment: ParsedDocumentFragment,
    *,
    source: str,
    scope: KnowledgeDocumentScope,
) -> KnowledgeDocument:
    metadata: dict[str, JsonValue] = dict(fragment.metadata or {})

    document_id = metadata.pop(DocumentMetadataKey.DOCUMENT_ID.value, None)
    if not isinstance(document_id, str) or not document_id:
        raise ValueError("fragment metadata must include a non-empty document_id")

    for key in metadata:
        if key in RESERVED_METADATA_KEYS:
            raise ValueError(
                f"parser metadata must not contain reserved KnowledgeDocument key: {key}"
            )

    parser_value = metadata.get(DocumentMetadataKey.PARSER.value)
    if not isinstance(parser_value, str) or not parser_value:
        raise ValueError("fragment metadata must include a non-empty parser")

    document = KnowledgeDocument.model_validate(
        {
            "schema_version": SCHEMA_VERSION,
            "identity": {
                "document_id": document_id,
                "root_document_id": document_id,
                "parent_document_id": None,
            },
            "scope": {
                "tenant_id": scope.tenant_id,
                "namespace": scope.namespace,
                "workspace_id": scope.workspace_id,
            },
            "content": fragment.text,
            "metadata": metadata,
            "provenance": {
                "source_kind": _resolve_source_kind(source),
                "source_id": source,
                "source_uri": source,
                "provider_id": parser_value,
            },
        }
    )
    if fragment.native_handle is not None:
        document = attach_parser_native_handle(document, fragment.native_handle)
    return document


class BaseDocumentHandler(ABC):
    """
    Contract for document format handlers used in the Intergrax RAG ingestion system.

    Handlers convert a source URI into a sequence of KnowledgeDocument objects.
    """

    @abstractmethod
    def supports(self, source: str) -> bool:
        """
        Determine whether this handler supports the given source.

        Parameters
        ----------
        source : str
            Source URI (file path, HTTP URL, S3 URI, etc.).

        Returns
        -------
        bool
        """
        raise NotImplementedError

    @abstractmethod
    def confidence(self, source: str) -> float:
        """
        Estimate how well this handler can process the source.

        Returns
        -------
        float
            Value in range [0.0, 1.0].
        """
        raise NotImplementedError

    @abstractmethod
    def build_parsers(self) -> List[BaseDocumentParser]:
        """
        Return ordered list of parsers.

        Order defines priority.
        """
        raise NotImplementedError

    def load(
        self,
        source: str,
        *,
        scope: KnowledgeDocumentScope,
    ) -> Sequence[KnowledgeDocument]:
        """
        Execute parser pipeline and map fragments to KnowledgeDocument instances.
        """
        parsers = self.build_parsers()

        pipeline = ParserPipeline(parsers)

        fragments = pipeline.parse(source)

        documents: list[KnowledgeDocument] = []
        for fragment in fragments:
            documents.append(
                _fragment_to_knowledge_document(
                    fragment,
                    source=source,
                    scope=scope,
                )
            )
        return documents
