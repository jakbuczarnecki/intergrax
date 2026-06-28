# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "PYMUPDF_DOCUMENT_PARSER_PROVIDER_ID",
    "PymupdfDocumentParserIntegration",
    "PymupdfDocumentParserIntegrationConfig",
    "PymupdfDocumentParserClient",
    "create_pymupdf_document_parser",
    "create_pymupdf_document_parser_integration",
    "register_pymupdf_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_pymupdf_document_parser",
        "create_pymupdf_document_parser_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "PYMUPDF_DOCUMENT_PARSER_PROVIDER_ID",
        "PymupdfDocumentParserIntegration",
        "PymupdfDocumentParserIntegrationConfig",
        "PymupdfDocumentParserClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "PYMUPDF_DOCUMENT_PARSER_PROVIDER_ID",
        "PymupdfDocumentParserIntegration",
        "PymupdfDocumentParserIntegrationConfig",
        "PymupdfDocumentParserClient",
    }
)

def __getattr__(name: str):
    if name == "register_pymupdf_integration":
        from intergrax.integrations.providers.document_parser.pymupdf.register import register_pymupdf_integration

        return register_pymupdf_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.document_parser.pymupdf import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.document_parser.pymupdf import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.document_parser.pymupdf import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
