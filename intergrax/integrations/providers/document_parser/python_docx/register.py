# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.document_parser.python_docx.bundle import create_python_docx_document_parser
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug


def register_python_docx_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.PYTHON_DOCX.value,
            categories=(IntegrationCategory.DOCUMENT_PARSER,),
            factory=create_python_docx_document_parser,
            status=IntegrationStatus.STABLE,
            env_prefix="INTERGRAX_PYTHON_DOCX",
            description="Microsoft Word (.docx) parser via python-docx",
        ),
        override=override,
    )
