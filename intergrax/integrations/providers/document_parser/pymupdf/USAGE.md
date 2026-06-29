# Pymupdf (pymupdf)

Category: `document_parser`

## Single public entrypoint

- **`PymupdfDocumentParserIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `PymupdfDocumentParserIntegration`.
- Contract factory: `create_pymupdf_document_parser_integration()`.
