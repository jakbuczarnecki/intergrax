# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations


from intergrax.rag.document_loaders.documents_loader import DocumentsLoader
from intergrax.rag.document_loaders.handlers.audio_smart_document_handler import AudioSmartDocumentHandler
from intergrax.rag.document_loaders.handlers.doc_smart_document_handler import DocSmartDocumentHandler
from intergrax.rag.document_loaders.handlers.excel_smart_document_handler import ExcelSmartDocumentHandler
from intergrax.rag.document_loaders.handlers.image_smart_document_handler import ImageSmartDocumentHandler
from intergrax.rag.document_loaders.handlers.pdf_smart_document_handler import PdfSmartDocumentHandler
from intergrax.rag.document_loaders.handlers.text_smart_document_handler import TextSmartDocumentHandler
from intergrax.rag.document_loaders.handlers.html_document_handler import HtmlSmartDocumentHandler
from intergrax.rag.document_loaders.handlers.video_smart_document_handler import VideoSmartDocumentHandler
from intergrax.rag.document_loaders.metadata_pipeline import MetadataPipeline
from intergrax.rag.document_loaders.metadata.default_metadata_provider import DefaultMetadataProvider
from intergrax.rag.document_loaders.normalizer_pipeline import NormalizerPipeline
from intergrax.rag.document_loaders.normalizers.whitespace_normalizer import WhitespaceNormalizer
from intergrax.rag.document_loaders.registry.document_handler_registry import DocumentHandlerRegistry


def create_default_documents_loader(
    registry: DocumentHandlerRegistry | None = None,
    normalizer_pipeline: NormalizerPipeline | None = None,
    metadata_pipeline: MetadataPipeline | None = None,
) -> DocumentsLoader:

    if registry is None:
        registry = DocumentHandlerRegistry()

        registry.register(PdfSmartDocumentHandler())
        registry.register(DocSmartDocumentHandler())
        registry.register(ExcelSmartDocumentHandler())
        registry.register(HtmlSmartDocumentHandler())
        registry.register(TextSmartDocumentHandler())
        registry.register(VideoSmartDocumentHandler())
        registry.register(AudioSmartDocumentHandler())
        registry.register(ImageSmartDocumentHandler())

    if normalizer_pipeline is None:
        normalizer_pipeline = NormalizerPipeline(
            normalizers=[
                WhitespaceNormalizer(),
            ]
        )

    if metadata_pipeline is None:
        metadata_pipeline = MetadataPipeline(
            providers=[
                DefaultMetadataProvider(),
            ]
        )

    return DocumentsLoader(
        registry=registry,
        normalizer_pipeline=normalizer_pipeline,
        metadata_pipeline=metadata_pipeline,
    )