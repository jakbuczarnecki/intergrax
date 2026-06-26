# LocalSearchAgent — architecture

Implementation tracker: [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md)

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
`on_next_step` / cognitive pattern hooks
  1. parse_query
  2. retrieve_context    → rag.retrieve
  3. rank_and_dedupe
  4. format_evidence     → StepOutput for synthesizer or user
```

Implement domain logic in `steps/` — no Tier-3 imports.

---

## Pattern anchor (Cursor — read instead of runtime grep)

| Item | Location |
|------|----------|
| Canonical `invoke_tool` + allowlist pattern | [`agents/lkw_shared/PATTERN.md`](../../lkw_shared/PATTERN.md) |
| Shared helpers | [`agents/lkw_shared/runtime_helpers.py`](../../lkw_shared/runtime_helpers.py) |
| **Implementation point** | [`steps/search_job.py`](steps/search_job.py) — `run_search_job` |

Do **not** read `uaep.py` or `boundary_demo` to discover tool invocation for this agent.

---

## Integrations, tools, and skills

### Integrations (indirect)

| Slot | Default slug | Used by |
|------|--------------|---------|
| `vector_store` | `inmemory` / `chroma` | `rag.retrieve` hybrid search |
| `rerank_provider` | `cohere_rerank` | optional rerank in `RetrievalService` |
| `relational_store` | `sqlite` | task / session memory |

### Tools

| tool_id | Role |
|---------|------|
| `rag.retrieve` | Primary — semantic retrieval + citations |
| `rag.list_collections` | Diagnostics / collection scope |
| `memory.read` / `memory.write` | Session evidence cache |
| `cache.get` / `cache.set` | Query-result dedup |

### Skills (planned LKW.2)

| `skill_id` | `tool_ids` | Status |
|------------|------------|--------|
| `local.workspace.search` | `rag.retrieve`, `rag.list_collections`, `cache.get`, `cache.set` | Planned |

Host baseline: [`applications/local_workspace_application/host/tool_wiring.py`](../../applications/local_workspace_application/host/tool_wiring.py).

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

- LKW architecture: [`applications/local_workspace_application/docs/ARCHITECTURE.md`](../../applications/local_workspace_application/docs/ARCHITECTURE.md)
- Retrieval control plane: [`docs/guides/AGENT_CREATION_GUIDE.md` Appendix K](../../docs/guides/AGENT_CREATION_GUIDE.md#appendix-k--integration--rag-control-plane)
