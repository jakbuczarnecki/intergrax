# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from docling.document_converter import DocumentConverter, FormatOption
from docling.datamodel.base_models import InputFormat
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.backend.docling_parse_backend import DoclingParseDocumentBackend

from intergrax.rag.document_loaders.config.document_loader_config import GLOBAL_DOCUMENT_LOADER_CONFIG


_DOC_CONVERTER: DocumentConverter | None = None


def create_docling_converter() -> DocumentConverter:
    """
    Returns a singleton instance of DocumentConverter configured
    according to GLOBAL_DOCUMENT_LOADER_CONFIG.
    """

    global _DOC_CONVERTER

    if _DOC_CONVERTER is None:
        _DOC_CONVERTER = _build_docling_converter()

    return _DOC_CONVERTER


def _build_docling_converter() -> DocumentConverter:
    """
    Builds DocumentConverter according to loader configuration.
    """

    cfg = GLOBAL_DOCUMENT_LOADER_CONFIG

    if cfg.docling_simple_pdf_mode:

        return DocumentConverter(
            format_options={
                InputFormat.PDF: FormatOption(
                    pipeline_cls=SimplePipeline,
                    backend=DoclingParseDocumentBackend,
                )
            }
        )

    return DocumentConverter()