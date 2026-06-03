# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.contracts.document_parser import DocumentParser
from intergrax.integrations.core.slug import SlugInput, coerce_slug
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.factory import resolve


def resolve_document_parser(
    slug: SlugInput,
    **options: object,
) -> DocumentParser:
    """Resolve a catalog document parser by slug."""
    register_default_integrations()
    resolved_slug = coerce_slug(slug)
    return resolve(
        IntegrationCategory.DOCUMENT_PARSER,
        slug=resolved_slug,
        config=dict(options),
    )
