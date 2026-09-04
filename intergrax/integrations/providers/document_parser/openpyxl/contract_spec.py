# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for Openpyxl."""

from __future__ import annotations

from intergrax.integrations.providers.document_parser.openpyxl.bundle import (
    create_openpyxl_document_parser_integration,
)
from intergrax.integrations.providers.document_parser.openpyxl.integration import (
    OPENPYXL_DOCUMENT_PARSER_PROVIDER_ID,
    OpenpyxlDocumentParserIntegration,
    OpenpyxlDocumentParserIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.ai import DocumentParserIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="document_parser",
    provider_id=OPENPYXL_DOCUMENT_PARSER_PROVIDER_ID,
    integration_class=OpenpyxlDocumentParserIntegration,
    contract_class=DocumentParserIntegrationContract,
    contract_factory=create_openpyxl_document_parser_integration,
    display_name="Openpyxl",
    config_class=OpenpyxlDocumentParserIntegrationConfig,
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
