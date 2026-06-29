# Openpyxl (openpyxl)

Category: `document_parser`

## Single public entrypoint

- **`OpenpyxlDocumentParserIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `OpenpyxlDocumentParserIntegration`.
- Contract factory: `create_openpyxl_document_parser_integration()`.
