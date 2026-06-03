# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register python_docx in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.document_parser.python_docx.bundle import create_python_docx_document_parser
from intergrax.integrations.providers.document_parser.python_docx.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_python_docx_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_python_docx_document_parser, override=override)
