# LocalIndexerAgent — architecture

Implementation tracker: [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md)

**Capability:** `local.workspace.index`  
**Host:** [`applications/local_workspace_application`](../../../applications/local_workspace_application/)
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
`on_next_step` / cognitive pattern hooks
  1. validate_source_paths
  2. ingest_documents      → rag.ingest_document per path
  3. summarize_index_job   → structured StepOutput
```

Implement domain logic only in `steps` — no Tier-3 imports.

---

## Pattern anchor (Cursor — read instead of runtime grep)

| Item | Location |
|------|----------|
| Generic `invoke_tool` helpers | [`intergrax/agents/authoring/runtime_tool_helpers.py`(../../../intergrax/agents/authoring/runtime_tool_helpers.py) |
| Filesystem allowlist | [`intergrax/tools/providers/filesystem/allowlist.py`(../../../intergrax/tools/providers/filesystem/allowlist.py) |
| RAG ingest tool id | [`intergrax/tools/providers/rag/ingest_service.py`(../../../intergrax/tools/providers/rag/ingest_service.py) |
| **Implementation point** | [`steps/index_job.py`](steps/index_job.py) — `run_index_job` |

Do **not** read `uaep.py` or `boundary_demo` to discover tool invocation for this agent.

---

## Integrations, tools, and skills

### Integrations (indirect — Tier-3 `IntegrationProfile`)

| Slot | Default slug | Used by |
|------|--------------|---------|
| `document_parser` | `docling` | `rag.ingest_document`, `document.parse` |
| `vector_store` | `inmemory` / `chroma` | ingest pipeline embed + index |
| `relational_store` | `sqlite` | task memory for job status |

Agents do **not** import `integrations/providers` — see [`docs/project/architecture/INTEGRATIONS.md`](../../../../docs/project/architecture/INTEGRATIONS.md).

### Tools (`ToolProfile` on host)

| tool_id | Role |
|---------|------|
| `rag.ingest_document` | Primary — parse, chunk, embed, index |
| `document.parse` | Pre-flight / single-file parse |
| `rag.list_collections` | Verify collection after ingest |
| `memory.write` | Persist ingest job status |

Invoke via `ctx.invoke_tool(ToolRequest(...))` in UAEP steps.

### Skills (planned LKW.2)

| `skill_id` | `tool_ids` | Status |
|------------|------------|--------|
| `local.workspace.index` | `rag.ingest_document`, `document.parse`, `rag.list_collections` | Planned — register on `AgentContract.skill_ids` |

Until LKW.2: host `ToolProfile` enables tools; `contract.py` has `skills=[]`.

See [`docs/project/architecture/SKILLS.md`](../../../../docs/project/architecture/SKILLS.md) · LKW stack: [`applications/local_workspace_application/docs/ARCHITECTURE.md` §5](../../../applications/local_workspace_application/docs/ARCHITECTURE.md#5-integrations-tools-and-skills).

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

- LKW application architecture: [`applications/local_workspace_application/docs/ARCHITECTURE.md`](../../../applications/local_workspace_application/docs/ARCHITECTURE.md)
- RAG ingest tool: [`intergrax/tools/providers/rag/USAGE.md`(../../../intergrax/tools/providers/rag/USAGE.md)
- Agent creation: [`docs/project/technical/guides/AGENT_CREATION_GUIDE.md`](../../../../docs/project/technical/guides/AGENT_CREATION_GUIDE.md)
