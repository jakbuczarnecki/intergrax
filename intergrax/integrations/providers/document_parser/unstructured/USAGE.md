# Unstructured (unstructured)

Category: `document_parser`

## Single public entrypoint

- **`UnstructuredDocumentParserIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `UnstructuredDocumentParserIntegration`.
- Contract factory: `create_unstructured_document_parser_integration()`.
