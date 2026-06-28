# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "YT_DLP_DOCUMENT_PARSER_PROVIDER_ID",
    "YtDlpDocumentParserIntegration",
    "YtDlpDocumentParserIntegrationConfig",
    "YtDlpDocumentParserClient",
    "create_yt_dlp_document_parser",
    "create_yt_dlp_document_parser_integration",
    "register_yt_dlp_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_yt_dlp_document_parser",
        "create_yt_dlp_document_parser_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "YT_DLP_DOCUMENT_PARSER_PROVIDER_ID",
        "YtDlpDocumentParserIntegration",
        "YtDlpDocumentParserIntegrationConfig",
        "YtDlpDocumentParserClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "YT_DLP_DOCUMENT_PARSER_PROVIDER_ID",
        "YtDlpDocumentParserIntegration",
        "YtDlpDocumentParserIntegrationConfig",
        "YtDlpDocumentParserClient",
    }
)

def __getattr__(name: str):
    if name == "register_yt_dlp_integration":
        from intergrax.integrations.providers.document_parser.yt_dlp.register import register_yt_dlp_integration

        return register_yt_dlp_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.document_parser.yt_dlp import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.document_parser.yt_dlp import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.document_parser.yt_dlp import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
