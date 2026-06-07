# LocalSearchAgent — architecture

**Capability:** `local.workspace.search`  
**Host:** [`applications/local_workspace_application/`](../../applications/local_workspace_application/)  
**Default agent** on LKW host roster.  
**Status:** Scaffold — domain steps pending Wave LKW.1

---

## Purpose

Answer user questions by retrieving relevant document fragments from the local RAG index. Packages **evidence-first** responses: chunk text, source path, and confidence metadata for the synthesizer or direct user display.

---

## Responsibilities

| In scope | Out of scope |
|----------|--------------|
| Semantic retrieval via `rag.retrieve` | Ingesting new files (delegate to indexer) |
| Metadata filtering (tenant, collection, path prefix) | Generating long-form reports |
| Ranking and deduplication of chunks | Writing user filesystem |
| Citation packaging for graph handoff | External web search |

---

## Inputs

| Source | Field | Description |
|--------|-------|-------------|
| Task message | Natural language query | e.g. „dokumenty o projekcie X” |
| Task metadata | `collection_id` | Scope retrieval to user workspace |
| Task metadata | `top_k` | Optional retrieval depth |
| Task metadata | `path_prefix` | Filter chunks by source path (Wave 2+) |

---

## Outputs

| Field | Description |
|-------|-------------|
| `evidence` | List of `{text, source_path, chunk_id, score}` |
| `answer_summary` | Short LLM summary grounded in evidence |
| `gaps` | What could not be found in the index |

---

## UAEP pipeline

```text
steps/pipeline.py
  1. parse_query
  2. retrieve_context    → rag.retrieve
  3. rank_and_dedupe
  4. format_evidence     → StepOutput for synthesizer or user
```

---

## Tools

- `rag.retrieve` (primary)
- `rag.list_collections` (diagnostics)
- `memory.read` / `memory.write` (session context)
- `cache.get` / `cache.set` (query result cache)

---

## Prompts

[`prompts/system.md`](prompts/system.md)

---

## Tests

```bash
uv run pytest agents/local_search/tests -q
```

---

## References

- LKW architecture: [`applications/local_workspace_application/ARCHITECTURE.md`](../../applications/local_workspace_application/ARCHITECTURE.md)
- Retrieval control plane: [`docs/AGENT_CREATION_GUIDE.md` Appendix K](../../docs/AGENT_CREATION_GUIDE.md#appendix-k--integration--rag-control-plane)
