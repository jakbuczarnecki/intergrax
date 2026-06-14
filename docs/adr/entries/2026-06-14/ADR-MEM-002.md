# ADR-MEM-002: Three-domain vector memory catalog (knowledge, LTM, episodic)

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-06-14 |
| **Deciders** | Harness platform architecture |
| **Related** | [`architecture/MEMORY.md`](../../architecture/MEMORY.md) §5.3–§6.5 · Phase MEM-VEC |

## Context

Phase MEM and MEM-DEPTH closed memory stores, consolidation, and Context Compiler. **Vector semantic recall** remained open:

1. `UserProfileManager` supports LTM vector upsert/search but Tier-3 wiring did not inject the RAG stack — semantic LTM was a silent no-op on reference hosts.
2. Session turn semantic recall (`episodic` domain) was not indexed at write time; CE shipped `SessionSemanticRecallProvider` (CE-VEC-1) but no backend populated `session_vector_hits`.
3. Document RAG (`knowledge`), user LTM (`ltm`), and session turns (`episodic`) must not share one undifferentiated index without metadata isolation.

Alternatives considered:

1. **Single merged collection** — rejected; breaks tombstone semantics and cross-domain leakage risk.
2. **Separate vector DB per domain** — rejected for lab/default hosts; over-provisions infra.
3. **One integration vector store + metadata domains + Tier-3 harness wiring (chosen)** — aligns with existing `VectorstoreManager` and [`RAG.md`](../../architecture/RAG.md) integration stack.

## Decision

Adopt a **three-domain vector catalog** on the shared integration stack:

| Domain | Indexed payload | Source of truth | Minimum metadata |
|--------|-----------------|-----------------|------------------|
| `knowledge` | Document chunks | RAG ingest | tenant, collection, workspace |
| `ltm` | `UserProfileMemoryEntry.content` | `UserProfileStore` | `user_id`, `entry_id`, `kind`, `deleted`, `index_domain=ltm` |
| `episodic` | Session turn text | `SessionStorage` | `tenant_id`, `session_id`, `user_id`, `entry_id`, `role`, `deleted`, `index_domain=episodic` |

**Tier-3 contract (MEM-VEC-1):** when `MemoryProfile.enable_long_term_memory` is true and the host resolves vector + embedding managers, `build_session_manager_from_environment(rag_stack=...)` MUST construct a vector-enabled `UserProfileManager`. The **same** instance is exposed on `ToolWiringContext.user_profile_manager` for `ltm.search` / `ltm.write_fact`.

**Episodic contract (MEM-VEC-2):** when `MemoryProfile.enable_session_vector_index` is true, `SessionManager.append_message` upserts into the episodic index; recall runs before CE assembly and fills `session_vector_hits` for `SessionSemanticRecallProvider`.

**Fail-closed:** if vector-index memory flags are true but no vector backend is resolved, host wiring MUST raise `MemoryVectorBackendUnavailableError` — not silently disable semantic recall.

**Rejected:**

- Agents opening vector SDKs directly — Tier violation.
- Neo4j as default episodic backend — integration catalog only.

## Consequences

### Positive

- LTM semantic search and episodic recall become testable end-to-end on lab profiles.
- CE `SESSION_HISTORY_SEMANTIC` and `LONGTERM_MEMORY` fragments receive attributable backend hits.
- FAUDIT-32 Memory Layer can advance past L3 vector-recall gap.

### Negative

- Shared vector store requires disciplined metadata filters on every query.
- Short async indexing lag possible on episodic writes — CE must tolerate empty hits with `reason=index_pending` when configured.

## Compliance

- Tier boundaries preserved — memory indexes in Tier-0/Tier-1; agents use Nexus APIs and catalog tools only.
- Tombstones on logical delete propagate to vector rows (`deleted=1` or delete by id).
- Linked from [`architecture/MEMORY.md`](../../architecture/MEMORY.md) and [`plan/MEMORY.md`](../../plan/MEMORY.md) Phase MEM-VEC.

## Implementation notes

- `intergrax/applications/_shared/memory_wiring.py` — RAG stack injection
- `intergrax/memory/session_turn_index_service.py` — episodic adapter
- `intergrax/runtime/nexus/context/memory_context_invocation.py` — LTM + episodic recall population
- Verification: `pytest -m gate -q`; `tests/integration/applications/test_memory_vector_ltm_wiring.py`
