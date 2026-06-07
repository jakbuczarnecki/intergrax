# LocalIndexerAgent — architecture

**Capability:** `local.workspace.index`  
**Host:** [`applications/local_workspace_application/`](../../applications/local_workspace_application/)  
**Status:** Scaffold — domain steps pending Wave LKW.1

---

## Purpose

Ingest user-local documents into the Intergrax RAG vector index so downstream agents can semantically search file content. The indexer is **read-only** with respect to the user's filesystem — it reads source files for parsing and embedding but never mutates originals.

---

## Responsibilities

| In scope | Out of scope |
|----------|--------------|
| Parse files via `document.parse` / ingest pipeline | Writing to user folders |
| Chunk, embed, store via `rag.ingest_document` | Web search |
| Report ingest stats (chunks, parser_id, trace) | Business synthesis |
| Honor `collection_id` / tenant metadata filters | Filesystem directory walk (Wave 3 — Tier-0 tools) |

---

## Inputs

| Source | Field | Description |
|--------|-------|-------------|
| Task message | Free text | Optional ingest instructions |
| Task metadata | `source_paths` | List of absolute paths to ingest (Wave 1) |
| Task metadata | `collection_id` | Vector collection / partition name |
| Task metadata | `chunking_strategy_id` | Optional RAG chunking override |

---

## Outputs

| Field | Description |
|-------|-------------|
| `num_chunks` | Chunks written to vector store |
| `vector_ids` | IDs from ingest pipeline |
| `parser_id` / `parser_trace` | Parser observability |
| `reason` | Skip/failure explanation when `used=false` |

---

## UAEP pipeline

```text
steps/pipeline.py
  1. validate_source_paths
  2. ingest_documents      → rag.ingest_document per path
  3. summarize_index_job   → structured StepOutput
```

Implement domain logic only in `steps/` — no Tier-3 imports.

---

## Tools (via Tier-3 ToolProfile)

- `rag.ingest_document` (primary)
- `document.parse` (optional pre-flight)
- `memory.write` (job status cache)

---

## Prompts

System instructions: [`prompts/system.md`](prompts/system.md)

---

## Tests

```bash
uv run pytest agents/local_indexer/tests -q
```

---

## References

- LKW application architecture: [`applications/local_workspace_application/ARCHITECTURE.md`](../../applications/local_workspace_application/ARCHITECTURE.md)
- RAG ingest tool: [`intergrax/tools/providers/rag/USAGE.md`](../../intergrax/tools/providers/rag/USAGE.md)
- Agent creation: [`docs/AGENT_CREATION_GUIDE.md`](../../docs/AGENT_CREATION_GUIDE.md)
