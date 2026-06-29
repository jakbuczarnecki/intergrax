# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "PYTHON_DOCX_DOCUMENT_PARSER_PROVIDER_ID",
    "PythonDocxDocumentParserIntegration",
    "PythonDocxDocumentParserIntegrationConfig",
    "PythonDocxDocumentParserClient",
    "create_python_docx_document_parser",
    "create_python_docx_document_parser_integration",
    "register_python_docx_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_python_docx_document_parser",
        "create_python_docx_document_parser_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "PYTHON_DOCX_DOCUMENT_PARSER_PROVIDER_ID",
        "PythonDocxDocumentParserIntegration",
        "PythonDocxDocumentParserIntegrationConfig",
        "PythonDocxDocumentParserClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "PYTHON_DOCX_DOCUMENT_PARSER_PROVIDER_ID",
        "PythonDocxDocumentParserIntegration",
        "PythonDocxDocumentParserIntegrationConfig",
        "PythonDocxDocumentParserClient",
    }
)

def __getattr__(name: str):
    if name == "register_python_docx_integration":
        from intergrax.integrations.providers.document_parser.python_docx.register import register_python_docx_integration

        return register_python_docx_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.document_parser.python_docx import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.document_parser.python_docx import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.document_parser.python_docx import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
