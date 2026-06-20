# MEMORY — §12+ scenarios & control

**Parent hub:** [`MEMORY.md`](../MEMORY.md)

## 12. Persistence backend matrix

| Layer | In-memory | SQLite (lab) | MongoDB | Postgres | Redis |
|-------|-----------|--------------|---------|----------|-------|
| Task KV | tests | `INTERGRAX_TASK_MEMORY_DB` | — | — | — |
| Session | fallback | sqlite bundle | `DocumentStoreSessionStorage` (MEM-DEPTH-2.1) | spike P3 | **not memory layer** |
| User LTM | tests | sqlite bundle | `DocumentStoreUserProfileStore` | spike P3 | — |
| Org profile | tests | sqlite bundle | — | — | — |
| Trace / events | tests | yes | — | — | — |

**Lab default:** `create_sqlite_integration()` bundles session + user LTM + org + task_memory + trace — coherent dev recovery.

---

## 13. Observability

Memory and context operations emit through the Harness Observability Spine:

| Signal | Examples |
|--------|----------|
| Runtime events | `MEMORY_READ`, `MEMORY_WRITE`, `CONTEXT_ASSEMBLED`, `CONTEXT_TRIMMED` |
| Diagnostics | `HistorySummaryDiagV1`, `UserLongtermMemorySummaryDiagV1`, `SessionConsolidationDiagV1` |
| Metrics | LTM hit rate, retention violations, memory write volume (MEM-OBS.1) |
| Debug filter | `ops:memory` (DX-5.7) |

Operators reconstruct: what was retrieved, what was trimmed, which strategy fired, and why LTM was skipped.

See [`architecture/OBSERVABILITY.md`](architecture/OBSERVABILITY.md) §3.

---

## 14. Market parity reference

| Capability | LangGraph | Mem0 / Zep | Intergrax as-built | Backlog |
|------------|-----------|------------|-------------------|--------|
| Thread persistence | Checkpointer | Session | Session + checkpoint + Mongo document_store ✅ | — |
| Scoped KV | Store API | — | TaskMemory ✅ | — |
| Auto fact extraction | — | Core | Consolidation + `consolidation_mode=auto` ✅ | — |
| Entity graph memory | — | Zep ✅ | `EntityGraphMemoryStore` ✅ (≠ Graph RAG) | — |
| Vector semantic LTM | Optional | ✅ | Tier-3 wired via `memory_vector_wiring.py` ✅ | — |
| Vector semantic session recall | Optional | ✅ | Episodic index + CE provider ✅ | — |
| Subagent isolation | Subgraph | — | Delegation namespace ✅ | Explore pattern MEM-DEPTH-4.* |
| Unified context budget | Partial | — | `ContextCompiler` ✅; per-step caps remain | CE ranker tuning |
| Temporal fact validity | — | Zep ✅ | ❌ | MEM-DEPTH-5.2 |
| Plugin episodic index EP | — | — | Default adapter only | MEM-VEC-3.1 |
| Unified semantic search skill | — | ✅ | `ltm.search` tool only | MEM-VEC-3.2 |

---

## 15. Anti-patterns

| Anti-pattern | Why forbidden | Correct approach |
|--------------|---------------|------------------|
| Agent concatenates full chat for LLM | Unbounded overflow | Nexus `HistoryLayer` + budget |
| Agent writes SQLite / Redis directly | Tier violation | `memory.*` tools + `MemoryView` |
| Storing documents in user LTM | Wrong store semantics | RAG ingest (`knowledge` domain) |
| Indexing session turns only in LTM consolidation | Loses verbatim recall | `episodic` index on append (MEM-VEC-2) |
| Enabling LTM flags without wiring RAG to `UserProfileManager` | Silent no-op semantic search | MEM-VEC-1.1 harness contract |
| Treating Graph RAG nodes as user entities | Conflates knowledge vs memory | Separate stores §5.3 |
| Silent context drop | No audit | Degradation ladder + events |
| Global shared KV without namespace | Cross-tenant leak risk | `tenant_id` + `task_id` + policy |
| Using trace as mutable memory | Immutability violation | Task KV or LTM |

---

## 16. Module inventory (do not duplicate)

| Module | Tier | Role |
|--------|------|------|
| `intergrax/memory/` | 0 | ConversationalMemory, UserProfileManager, stores |
| `intergrax/runtime/task_memory/` | 1 | TaskMemory, MemoryView, delegation, retention |
| `intergrax/runtime/nexus/session/` | 1 | SessionManager, consolidation coordinator |
| `intergrax/runtime/nexus/context/` | 1 | ContextManager, HistoryLayer, context_budget |
| `intergrax/runtime/user_profile/` | 1 | Consolidation + instructions services |
| `intergrax/applications/_shared/memory_wiring.py` | 3 | Platform wiring |
| `intergrax/runtime/nexus/context/context_compiler.py` | 1 | Context Compiler + degradation ladder |
| `intergrax/runtime/nexus/session/document_store_session_storage.py` | 1 | Mongo session persistence |
| `intergrax/memory/entity_graph_memory.py` | 0 | User entity graph (≠ Graph RAG) |
| `intergrax/tools/providers/memory/` | 0 | `memory.read/write/list_keys/delete_key` |
| `intergrax/rag/` | 0 | Knowledge retrieval (not agent LTM) |

---

## 17. Maturity scorecard and gap register

| Area | Score (1–5) | Phase MEM | Phase MEM-DEPTH | Phase MEM-VEC |
|------|-------------|-----------|-----------------|---------------|
| Task KV | 4 | Done | 4 (maintain) | — |
| Context / LLM window | 4.5 | Partial | Done — Context Compiler | CE owns ranker |
| STM session | 4 | Partial | Done — Mongo + SQLite parity | — |
| User LTM | 4 | Partial | Done — store + vector wiring | Done — MEM-VEC-1 |
| LTM / session vector recall | 4 | N/A | N/A | Done — MEM-VEC-1/2 |
| Org memory | 3.5 | Partial | Done — org LTM entries + Mongo fallback store | — |
| Consolidation | 4 | Partial | Done — job + modes | — |
| Graph agent memory | 3.5 | RFC only | Done — `EntityGraphMemoryStore` + consolidation indexing | — |
| Context compiler (unified) | 4.5 | N/A | Done | CE canon |
| **Overall** | **~4.5** | Platform wiring Done | Closed | MEM-VEC **Done** |

**FAUDIT-32:** Memory Layer **L4** for vector recall and layer-completion hardening (2026-06-17).

### Audit register (open / partial — re-validate)

| ID | Gap | Severity | Phase | Status |
|----|-----|----------|-------|--------|
| MEM-AUDIT-1 | No versioned procedural memory store (Prompt Registry only) | P2 | — | **Open** (by design minimal) |
| MEM-AUDIT-2 | Org memory maturity vs user LTM | P2 | — | **Partial** — org `memory_entries` + manager search shipped |
| MEM-AUDIT-3 | Temporal fact validity on LTM entries | P2 | MEM-DEPTH-5.2 | **Done** |
| MEM-AUDIT-4 | `SessionTurnIndexStore` plugin EP not shipped | P2 | MEM-VEC-3.1 | **Done** |
| MEM-AUDIT-5 | `memory.semantic_search` skill runtime (unified LTM + episodic) | P2 | MEM-VEC-3.2 | **Done** |
| MEM-AUDIT-6 | Explore delegation pattern (Cursor-class) | P2 | MEM-DEPTH-4.* | **Done** — graph executor wiring |
| MEM-AUDIT-7 | Per-step budget caps before CE collect | P2 | CE + ADR-MEM-001 | **Partial** — global allocator Done |

**Closed baselines:** MEM (48/48), MEM-DEPTH, AUDIT-IDEAL-15.1–15.3, AUDIT-IDEAL-16.1–16.2 (CE owner).

Implementation tasks: [Phase MEM-VEC](../plan/MEMORY.md#phase-mem-vec--vector-memory-integration-band-2aw) · [Phase MEM-DEPTH](../plan/MEMORY.md) (closed).

---

## 18. Related documents

| Document | Relationship |
|----------|--------------|
| [intergrax_runtime_architecture.md §27–§28](architecture/MEMORY.md#27-memory-model) | Canon summary — links here for depth |
| [architecture/CONTEXT_ENGINEERING.md](CONTEXT_ENGINEERING.md) | Context engineering engine (Layer C); AUDIT-IDEAL-16.1–16.2 owner |
| [guides/AGENT_CREATION_GUIDE.md Appendix G](guides/AGENT_CREATION_GUIDE.md#appendix-g--memory--rag-naming-phase-q) | Author control plane |
| [guides/AGENT_CREATION_GUIDE.md Appendix L](guides/AGENT_CREATION_GUIDE.md#appendix-l--context-engineering-control-plane) | Author control plane (links CE canon) |
| [adr/entries/2026-06-08/ADR-MEM-001.md](../adr/entries/2026-06-08/ADR-MEM-001.md) | Context Compiler + degradation ladder |
| [adr/entries/2026-06-14/ADR-MEM-002.md](../adr/entries/2026-06-14/ADR-MEM-002.md) | Three-domain vector catalog |
| [architecture/TOOLS.md](architecture/TOOLS.md) | `memory.*` and `rag.retrieve` tools |
| [architecture/NEXUS_EXECUTION_FLOW.md](architecture/NEXUS_EXECUTION_FLOW.md) | Runtime turn narrative |
| [IDEAL_HARNESS_AI_ARCHITECTURE.md §16](../guides/IDEAL_HARNESS_AI_ARCHITECTURE.md#16-context-engineering-layer) | Target context compiler vision |

---

*End of Memory Architecture canon.*

---

## 19. Context quality (delegated)

Context quality controls (scoring, dedup, regression, lineage) are owned by **[`CONTEXT_ENGINEERING.md`](CONTEXT_ENGINEERING.md) §11**.

---

## 20. Knowledge Graph and Hybrid Retrieval

Graph-native knowledge evolves from optional enhancement to first-class capability:

- graph RAG support (`intergrax/rag/graph/`),
- entity–relation semantic modeling,
- hybrid retrieval: vector + keyword + graph traversal,
- graph-backed explainability in reasoning traces.

| Module | Role |
|--------|------|
| `runtime/architecture/graph_rag.py` | Graph RAG contracts |
| `hybrid_retrieval.py` | Hybrid strategy |
| `graph_provenance.py` | Lineage for graph edges |

**Distinction:** Graph RAG indexes **document knowledge** — not user episodic memory (§4). Integration backends (Neo4j, etc.) are catalog providers in [`INTEGRATIONS.md`](INTEGRATIONS.md).

---
