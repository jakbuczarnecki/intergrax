# Python Docx (python_docx)

Category: `document_parser`

## Single public entrypoint

- **`PythonDocxDocumentParserIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `PythonDocxDocumentParserIntegration`.
- Contract factory: `create_python_docx_document_parser_integration()`.
