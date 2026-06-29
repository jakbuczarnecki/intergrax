# Whisper (whisper)

Category: `document_parser`

## Single public entrypoint

- **`WhisperDocumentParserIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `WhisperDocumentParserIntegration`.
- Contract factory: `create_whisper_document_parser_integration()`.
