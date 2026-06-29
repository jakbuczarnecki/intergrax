# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "OPENPYXL_DOCUMENT_PARSER_PROVIDER_ID",
    "OpenpyxlDocumentParserIntegration",
    "OpenpyxlDocumentParserIntegrationConfig",
    "OpenpyxlDocumentParserClient",
    "create_openpyxl_document_parser",
    "create_openpyxl_document_parser_integration",
    "register_openpyxl_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_openpyxl_document_parser",
        "create_openpyxl_document_parser_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "OPENPYXL_DOCUMENT_PARSER_PROVIDER_ID",
        "OpenpyxlDocumentParserIntegration",
        "OpenpyxlDocumentParserIntegrationConfig",
        "OpenpyxlDocumentParserClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "OPENPYXL_DOCUMENT_PARSER_PROVIDER_ID",
        "OpenpyxlDocumentParserIntegration",
        "OpenpyxlDocumentParserIntegrationConfig",
        "OpenpyxlDocumentParserClient",
    }
)

def __getattr__(name: str):
    if name == "register_openpyxl_integration":
        from intergrax.integrations.providers.document_parser.openpyxl.register import register_openpyxl_integration

        return register_openpyxl_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.document_parser.openpyxl import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.document_parser.openpyxl import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.document_parser.openpyxl import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
