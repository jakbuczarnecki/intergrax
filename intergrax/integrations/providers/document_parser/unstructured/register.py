# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register unstructured in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.document_parser.unstructured.bundle import create_unstructured_document_parser
from intergrax.integrations.providers.document_parser.unstructured.manifest import MANIFEST
from intergrax.integrations.providers.document_parser.unstructured.contract_spec import CONTRACT_SPECS
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_unstructured_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_unstructured_document_parser, override=override, contract_specs=CONTRACT_SPECS)
