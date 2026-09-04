# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for Llamaparse."""

from __future__ import annotations

from intergrax.integrations.providers.document_parser.llamaparse.bundle import (
    create_llamaparse_document_parser_integration,
)
from intergrax.integrations.providers.document_parser.llamaparse.integration import (
    LLAMAPARSE_DOCUMENT_PARSER_PROVIDER_ID,
    LlamaparseDocumentParserIntegration,
    LlamaparseDocumentParserIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.ai import DocumentParserIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="document_parser",
    provider_id=LLAMAPARSE_DOCUMENT_PARSER_PROVIDER_ID,
    integration_class=LlamaparseDocumentParserIntegration,
    contract_class=DocumentParserIntegrationContract,
    contract_factory=create_llamaparse_document_parser_integration,
    display_name="Llamaparse",
    config_class=LlamaparseDocumentParserIntegrationConfig,
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
