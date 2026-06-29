# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "WHISPER_DOCUMENT_PARSER_PROVIDER_ID",
    "WhisperDocumentParserIntegration",
    "WhisperDocumentParserIntegrationConfig",
    "WhisperDocumentParserClient",
    "create_whisper_document_parser",
    "create_whisper_document_parser_integration",
    "register_whisper_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_whisper_document_parser",
        "create_whisper_document_parser_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "WHISPER_DOCUMENT_PARSER_PROVIDER_ID",
        "WhisperDocumentParserIntegration",
        "WhisperDocumentParserIntegrationConfig",
        "WhisperDocumentParserClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "WHISPER_DOCUMENT_PARSER_PROVIDER_ID",
        "WhisperDocumentParserIntegration",
        "WhisperDocumentParserIntegrationConfig",
        "WhisperDocumentParserClient",
    }
)

def __getattr__(name: str):
    if name == "register_whisper_integration":
        from intergrax.integrations.providers.document_parser.whisper.register import register_whisper_integration

        return register_whisper_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.document_parser.whisper import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.document_parser.whisper import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.document_parser.whisper import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
