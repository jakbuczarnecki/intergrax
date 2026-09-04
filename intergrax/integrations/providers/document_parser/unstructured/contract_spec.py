# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for Unstructured."""

from __future__ import annotations

from intergrax.integrations.providers.document_parser.unstructured.bundle import (
    create_unstructured_document_parser_integration,
)
from intergrax.integrations.providers.document_parser.unstructured.integration import (
    UNSTRUCTURED_DOCUMENT_PARSER_PROVIDER_ID,
    UnstructuredDocumentParserIntegration,
    UnstructuredDocumentParserIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.ai import DocumentParserIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="document_parser",
    provider_id=UNSTRUCTURED_DOCUMENT_PARSER_PROVIDER_ID,
    integration_class=UnstructuredDocumentParserIntegration,
    contract_class=DocumentParserIntegrationContract,
    contract_factory=create_unstructured_document_parser_integration,
    display_name="Unstructured",
    config_class=UnstructuredDocumentParserIntegrationConfig,
    capabilities=(
        PlatformIntegrationCapability.READ,
        PlatformIntegrationCapability.HEALTH_CHECK,
    ),
    security_posture=PlatformIntegrationSecurityPosture(),
    supports_runtime_binding=True,
    supports_health_check=True,
    metadata={"source": "explicit_provider_declaration"},
)

CONTRACT_SPECS = (CONTRACT_SPEC,)

__all__ = ["CONTRACT_SPEC", "CONTRACT_SPECS"]
