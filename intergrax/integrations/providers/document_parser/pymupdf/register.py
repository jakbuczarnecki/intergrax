# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register pymupdf in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.document_parser.pymupdf.bundle import create_pymupdf_document_parser
from intergrax.integrations.providers.document_parser.pymupdf.manifest import MANIFEST
from intergrax.integrations.providers.document_parser.pymupdf.contract_spec import CONTRACT_SPECS
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_pymupdf_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_pymupdf_document_parser, override=override, contract_specs=CONTRACT_SPECS)
