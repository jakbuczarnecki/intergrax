# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register llamaparse in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.document_parser.llamaparse.bundle import create_llamaparse_document_parser
from intergrax.integrations.providers.document_parser.llamaparse.manifest import MANIFEST
from intergrax.integrations.providers.document_parser.llamaparse.contract_spec import CONTRACT_SPECS
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_llamaparse_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_llamaparse_document_parser, override=override, contract_specs=CONTRACT_SPECS)
