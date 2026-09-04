# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register openpyxl in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.document_parser.openpyxl.bundle import create_openpyxl_document_parser
from intergrax.integrations.providers.document_parser.openpyxl.manifest import MANIFEST
from intergrax.integrations.providers.document_parser.openpyxl.contract_spec import CONTRACT_SPECS
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_openpyxl_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_openpyxl_document_parser, override=override, contract_specs=CONTRACT_SPECS)
