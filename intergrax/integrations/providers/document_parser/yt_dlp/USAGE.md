# Yt Dlp (yt_dlp)

Category: `document_parser`

## Single public entrypoint

- **`YtDlpDocumentParserIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `YtDlpDocumentParserIntegration`.
- Contract factory: `create_yt_dlp_document_parser_integration()`.
