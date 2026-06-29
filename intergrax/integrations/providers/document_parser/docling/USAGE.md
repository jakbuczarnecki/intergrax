# Docling (docling)

Category: `document_parser`

## Single public entrypoint

- **`DoclingDocumentParserIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `DoclingDocumentParserIntegration`.
- Contract factory: `create_docling_document_parser_integration()`.
