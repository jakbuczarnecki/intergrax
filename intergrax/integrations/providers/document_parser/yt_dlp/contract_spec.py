# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for yt-dlp."""

from __future__ import annotations

from intergrax.integrations.providers.document_parser.yt_dlp.bundle import (
    create_yt_dlp_document_parser_integration,
)
from intergrax.integrations.providers.document_parser.yt_dlp.integration import (
    YT_DLP_DOCUMENT_PARSER_PROVIDER_ID,
    YtDlpDocumentParserIntegration,
    YtDlpDocumentParserIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.ai import DocumentParserIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="document_parser",
    provider_id=YT_DLP_DOCUMENT_PARSER_PROVIDER_ID,
    integration_class=YtDlpDocumentParserIntegration,
    contract_class=DocumentParserIntegrationContract,
    contract_factory=create_yt_dlp_document_parser_integration,
    display_name="yt-dlp",
    config_class=YtDlpDocumentParserIntegrationConfig,
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
