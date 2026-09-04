# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register whisper in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.document_parser.whisper.bundle import create_whisper_document_parser
from intergrax.integrations.providers.document_parser.whisper.manifest import MANIFEST
from intergrax.integrations.providers.document_parser.whisper.contract_spec import CONTRACT_SPECS
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_whisper_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_whisper_document_parser, override=override, contract_specs=CONTRACT_SPECS)
