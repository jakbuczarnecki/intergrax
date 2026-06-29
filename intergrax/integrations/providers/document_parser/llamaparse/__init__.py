# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "LLAMAPARSE_DOCUMENT_PARSER_PROVIDER_ID",
    "LlamaparseDocumentParserIntegration",
    "LlamaparseDocumentParserIntegrationConfig",
    "LlamaparseDocumentParserClient",
    "create_llamaparse_document_parser",
    "create_llamaparse_document_parser_integration",
    "register_llamaparse_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_llamaparse_document_parser",
        "create_llamaparse_document_parser_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "LLAMAPARSE_DOCUMENT_PARSER_PROVIDER_ID",
        "LlamaparseDocumentParserIntegration",
        "LlamaparseDocumentParserIntegrationConfig",
        "LlamaparseDocumentParserClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "LLAMAPARSE_DOCUMENT_PARSER_PROVIDER_ID",
        "LlamaparseDocumentParserIntegration",
        "LlamaparseDocumentParserIntegrationConfig",
        "LlamaparseDocumentParserClient",
    }
)

def __getattr__(name: str):
    if name == "register_llamaparse_integration":
        from intergrax.integrations.providers.document_parser.llamaparse.register import register_llamaparse_integration

        return register_llamaparse_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.document_parser.llamaparse import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.document_parser.llamaparse import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.document_parser.llamaparse import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
