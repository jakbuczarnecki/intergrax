# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register docling in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.document_parser.docling.bundle import create_docling_document_parser
from intergrax.integrations.providers.document_parser.docling.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_docling_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_docling_document_parser, override=override)
