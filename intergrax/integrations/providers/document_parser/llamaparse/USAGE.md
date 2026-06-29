# Llamaparse (llamaparse)

Category: `document_parser`

## Single public entrypoint

- **`LlamaparseDocumentParserIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `LlamaparseDocumentParserIntegration`.
- Contract factory: `create_llamaparse_document_parser_integration()`.
