# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p8.factories import create_llamaparse_document_parser as _legacy_create_llamaparse_document_parser

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.document_parser.llamaparse.integration import (
    LLAMAPARSE_DOCUMENT_PARSER_PROVIDER_ID,
    LlamaparseDocumentParserIntegration,
    LlamaparseDocumentParserIntegrationConfig,
    LlamaparseDocumentParserClient,
)

__all__ = [
    "create_llamaparse_document_parser",
    "create_llamaparse_document_parser_integration",
]


def create_llamaparse_document_parser_integration(
    *,
    client: LlamaparseDocumentParserClient | None = None,
    enabled: bool = False,
) -> LlamaparseDocumentParserIntegration:
    """
    Build a contract-based Llamaparse document parser integration.

    The legacy facade (create_llamaparse_document_parser) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Llamaparse document parser integration requires an injected client when enabled=True",
        )
    if client is not None:
        return LlamaparseDocumentParserIntegration.from_client(client, enabled=enabled)
    return LlamaparseDocumentParserIntegration.for_provider(
        provider_id=LLAMAPARSE_DOCUMENT_PARSER_PROVIDER_ID,
        display_name="Llamaparse",
        config=LlamaparseDocumentParserIntegrationConfig(enabled=enabled),
    )


def create_llamaparse_document_parser(**kwargs: object) -> LlamaparseDocumentParserIntegration:
    """Compatibility shim — constructs LlamaparseDocumentParserIntegration from legacy runtime."""
    runtime = _legacy_create_llamaparse_document_parser(**kwargs)
    if isinstance(runtime, LlamaparseDocumentParserIntegration):
        return runtime
    return LlamaparseDocumentParserIntegration.from_runtime(runtime)
