# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register dynamodb in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.document_store.dynamodb.bundle import create_dynamodb_document_store
from intergrax.integrations.providers.document_store.dynamodb.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest
from intergrax.integrations.providers.document_store.dynamodb.contract_spec import CONTRACT_SPECS


def register_dynamodb_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_dynamodb_document_store, override=override, contract_specs=CONTRACT_SPECS)
