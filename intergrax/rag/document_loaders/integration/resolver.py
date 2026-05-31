# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.integrations.contracts.document_parser import DocumentParser
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.factory import resolve
from intergrax.integrations.registry.slugs import IntegrationSlug, coerce_slug
from intergrax.integrations.contracts.base import IntegrationCategory


def resolve_document_parser(
    slug: str | IntegrationSlug,
    **options: object,
) -> DocumentParser:
    """Resolve a catalog document parser by slug."""
    register_default_integrations()
    slug_enum = coerce_slug(str(slug))
    return resolve(
        IntegrationCategory.DOCUMENT_PARSER,
        slug=slug_enum.value,
        config=dict(options),
    )
