# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "UNSTRUCTURED_DOCUMENT_PARSER_PROVIDER_ID",
    "UnstructuredDocumentParserIntegration",
    "UnstructuredDocumentParserIntegrationConfig",
    "UnstructuredDocumentParserClient",
    "create_unstructured_document_parser",
    "create_unstructured_document_parser_integration",
    "register_unstructured_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_unstructured_document_parser",
        "create_unstructured_document_parser_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "UNSTRUCTURED_DOCUMENT_PARSER_PROVIDER_ID",
        "UnstructuredDocumentParserIntegration",
        "UnstructuredDocumentParserIntegrationConfig",
        "UnstructuredDocumentParserClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "UNSTRUCTURED_DOCUMENT_PARSER_PROVIDER_ID",
        "UnstructuredDocumentParserIntegration",
        "UnstructuredDocumentParserIntegrationConfig",
        "UnstructuredDocumentParserClient",
    }
)

def __getattr__(name: str):
    if name == "register_unstructured_integration":
        from intergrax.integrations.providers.document_parser.unstructured.register import register_unstructured_integration

        return register_unstructured_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.document_parser.unstructured import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.document_parser.unstructured import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.document_parser.unstructured import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
