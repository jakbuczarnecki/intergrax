# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "DOCLING_DOCUMENT_PARSER_PROVIDER_ID",
    "DoclingDocumentParserIntegration",
    "DoclingDocumentParserIntegrationConfig",
    "DoclingDocumentParserClient",
    "create_docling_document_parser",
    "create_docling_document_parser_integration",
    "register_docling_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_docling_document_parser",
        "create_docling_document_parser_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "DOCLING_DOCUMENT_PARSER_PROVIDER_ID",
        "DoclingDocumentParserIntegration",
        "DoclingDocumentParserIntegrationConfig",
        "DoclingDocumentParserClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "DOCLING_DOCUMENT_PARSER_PROVIDER_ID",
        "DoclingDocumentParserIntegration",
        "DoclingDocumentParserIntegrationConfig",
        "DoclingDocumentParserClient",
    }
)

def __getattr__(name: str):
    if name == "register_docling_integration":
        from intergrax.integrations.providers.document_parser.docling.register import register_docling_integration

        return register_docling_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.document_parser.docling import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.document_parser.docling import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.document_parser.docling import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
