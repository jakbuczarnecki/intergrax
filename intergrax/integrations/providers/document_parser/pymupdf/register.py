# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.document_parser.pymupdf.bundle import create_pymupdf_document_parser
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug


def register_pymupdf_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.PYMUPDF.value,
            categories=(IntegrationCategory.DOCUMENT_PARSER,),
            factory=create_pymupdf_document_parser,
            status=IntegrationStatus.STABLE,
            env_prefix="INTERGRAX_PYMUPDF",
            description="PyMuPDF PDF parser with optional Tesseract OCR fallback",
        ),
        override=override,
    )
