# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.document_parser.docling.bundle import create_docling_document_parser
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug


def register_docling_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.DOCLING.value,
            categories=(IntegrationCategory.DOCUMENT_PARSER,),
            factory=create_docling_document_parser,
            status=IntegrationStatus.BETA,
            env_prefix="INTERGRAX_DOCLING",
            description="Docling document parser — local library or HTTP server",
        ),
        override=override,
    )
