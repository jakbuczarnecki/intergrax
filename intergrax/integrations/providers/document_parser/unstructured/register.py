# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.document_parser.unstructured.bundle import create_unstructured_document_parser
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug


def register_unstructured_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.UNSTRUCTURED.value,
            categories=(IntegrationCategory.DOCUMENT_PARSER,),
            factory=create_unstructured_document_parser,
            status=IntegrationStatus.BETA,
            env_prefix="INTERGRAX_UNSTRUCTURED",
            description="Unstructured HTML document parser",
        ),
        override=override,
    )
