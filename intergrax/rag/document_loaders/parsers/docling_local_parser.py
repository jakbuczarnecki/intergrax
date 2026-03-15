# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from docling.document_converter import DocumentConverter, FormatOption
from docling.datamodel.base_models import (
    BaseFormatOption,
    ConversionStatus,
    DoclingComponentType,
    DocumentStream,
    ErrorItem,
    InputFormat,
)
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.backend.docling_parse_backend import DoclingParseDocumentBackend

from langchain_core.documents import Document

from intergrax.rag.document_loaders.config.document_loader_config import GLOBAL_DOCUMENT_LOADER_CONFIG, DoclingMode
from intergrax.rag.document_loaders.contracts.base_document_parser import BaseDocumentParser
from intergrax.rag.document_loaders.contracts.document_metadata_key import DocumentMetadataKey
from intergrax.rag.document_loaders.contracts.metadata_contract import build_loader_metadata
from intergrax.rag.document_loaders.parsers.docling_converter_factory import create_docling_converter


class DoclingLocalParser(BaseDocumentParser):
    """
    Local Docling backend parser.

    This parser uses a locally installed Docling runtime
    to convert documents into LangChain Document objects.

    Actual Docling integration will be implemented in a later step.
    """

    @classmethod
    def parser_id(cls) -> str:
        return "docling.local"

    @classmethod
    def is_available(cls) -> bool:
        cfg = GLOBAL_DOCUMENT_LOADER_CONFIG
        return cfg.docling_mode == DoclingMode.LOCAL

    def load(self, source: str) -> Sequence[Document]:

        converter = create_docling_converter()

        result = converter.convert(Path(source))

        doc = result.document

        text = doc.export_to_markdown() or ""

        metadata = build_loader_metadata(
            source=source,
            parser=self.parser_id(),
            position=0,            
        )
        metadata[DocumentMetadataKey.DOCLING_DOCUMENT_META] = doc

        return [
            Document(
                page_content=text,
                metadata=metadata,
            )
        ]