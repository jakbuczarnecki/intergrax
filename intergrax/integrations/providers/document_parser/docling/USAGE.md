# Docling document parser

**Category:** `document_parser`  
**Slug:** `docling`  
**Env prefix:** `INTERGRAX_DOCLING_*`

## Modes

| Mode | Env | Description |
|------|-----|-------------|
| `local` | `INTERGRAX_DOCLING_MODE=local` | In-process Docling library |
| `server` | `INTERGRAX_DOCLING_MODE=server` | HTTP service (`INTERGRAX_DOCLING_SERVER_URL` + `INTERGRAX_DOCLING_SERVER_PATH`) |
| `none` | `INTERGRAX_DOCLING_MODE=none` | Disabled |

## Resolve from catalog

```python
from intergrax.integrations.registry.factory import resolve
from intergrax.integrations.contracts.base import IntegrationCategory

parser = resolve(IntegrationCategory.DOCUMENT_PARSER, slug="docling", config={"mode": "local"})
fragments = parser.parse_file("/path/to/file.pdf")
```

## RAG wiring

`intergrax/rag/document_loaders/` uses `CatalogDocumentParser` + `resolve_document_parser("docling")` — no direct `docling` imports in RAG parsers.
