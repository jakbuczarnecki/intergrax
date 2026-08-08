# ADR-TOOL-004: Semantic tool catalog index and selection boundary (TOOL-ENG-13)

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-06-12 |
| **Deciders** | Harness platform |
| **Related** | [`architecture/TOOLS.md`](../../architecture/TOOLS.md) · [`plan/TOOLS.md`](../../plan/TOOLS.md) TOOL-ENG-13,25 · ADR-TOOL-003 |

## Context

Large catalogs (190+ tools) need L6 semantic narrowing before LLM schema export. `retrieval_top_k` uses keyword overlap only. Document RAG (`rag.retrieve`) indexes product corpora — not tool metadata.

## Decision

1. Add `ToolCatalogEmbedder` + in-memory `ToolCatalogIndex` using Tier-0 `BaseEmbeddingManager` (from `RuntimeConfig.embedding_manager`).
2. Add `ToolSelectionMode.SEMANTIC` and `SemanticToolIndexSelectionStrategy`.
3. Index text = `tool_id` + description + tags + category; collection name canon `__harness_tool_catalog__` (logical — in-memory index per registry fingerprint, no document RAG pollution).
4. Add `ParallelSemanticBatchPattern` — semantic top-k → auto parallel invoke (read-only) → `ToolInvocationAggregate` (TOOL-ENG-25).
5. Defer vectorstore-persisted catalog index and entry-point strategy registry to TOOL-ENG-26.

**Rejected:** Reusing `rag.retrieve` document index for tool metadata. Keyword mode as semantic alias.

## Consequences

### Positive

- True embedding-based L6 selection at scale.
- Composite fan-out pattern for gather-then-synthesize flows.
- Clear boundary vs RAG document retrieval.

### Negative

- Requires `embedding_manager` on host when `tool_selection_mode=semantic`.
- In-memory reindex on registry change (acceptable for 190-tool scale).

## Compliance

- Tier-0 embedding only — no Tier-2 imports.
- Tests: `test_tool_catalog_embedder.py`, `test_parallel_semantic_batch_pattern.py`.

## Implementation notes

- `intergrax/runtime/nexus/tools/tool_catalog_embedder.py`
- `intergrax/runtime/nexus/tools/tool_selection.py`
- `intergrax/runtime/nexus/tools/patterns/parallel_semantic_batch.py`
