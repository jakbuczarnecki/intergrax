# Memory

**Status:** Canonical architecture (domain pair 1:1)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Plan (1:1):** [`plan/MEMORY.md`](../plan/MEMORY.md)  
**Target:** [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../guides/IDEAL_HARNESS_AI_ARCHITECTURE.md)  
**Audit layer:** 15 (Memory)  
**Audit instruction:** [`guides/audit/MEMORY.md`](../guides/audit/MEMORY.md)  
**Context assembly (Layer C):** [`architecture/CONTEXT_ENGINEERING.md`](CONTEXT_ENGINEERING.md) · [`plan/CONTEXT_ENGINEERING.md`](../plan/CONTEXT_ENGINEERING.md)  
**Related:** [`architecture/RAG.md`](RAG.md) — Tier-0 retrieval engine; this doc covers **memory stores, lifecycle**, and the **Knowledge vs LTM** boundary.  
**ADR:** [ADR-MEM-001](../adr/entries/2026-06-08/ADR-MEM-001.md) (Context Compiler) · [ADR-MEM-002](../adr/entries/2026-06-14/ADR-MEM-002.md) (vector catalog)

## 2. Design principles

| Principle | Meaning in Intergrax |
|-----------|---------------------|
| **Explicit stores** | Memory is never an implicit side effect of chat history alone. Every durable fact has a store, scope, and write path. |
| **Tier-1 ownership** | Nexus owns session history, consolidation triggers, and memory read APIs. Agents consume via `MemoryView` — no direct DB access. **Context assembly** is [`CONTEXT_ENGINEERING`](CONTEXT_ENGINEERING.md). |
| **Bounded by default** | FIFO session limits, token budgets, LTM top_k, retention_days, namespace isolation. Unbounded growth is a defect. |
| **Separation of concerns** | **Memory** = persisted state. **Context** = what the model sees this turn. **Knowledge** = document RAG (agent-mutable only via explicit tools). **Trace** = immutable audit. |
| **Retrieval-first for scale** | Large corpora (documents, codebase, long history) enter context via retrieval, summarization, or delegation — not full dumps. |
| **Policy-governed writes** | Sensitive LTM writes pass `BEFORE_MEMORY_WRITE` hooks and `MemoryWritePolicy`. |
| **Provenance on read** | Memory reads consumed by CE MUST be attributable — see [`CONTEXT_ENGINEERING.md`](CONTEXT_ENGINEERING.md) §12. |
| **Graph RAG ≠ agent memory** | Document knowledge graphs (`intergrax/rag/graph/`) are retrieval infrastructure — not user entity / episodic memory (Zep-style). |
| **Vector index ≠ primary store** | Relational / document stores remain the source of truth for session turns and LTM entries. Vector backends are **retrieval indexes** — optional, scoped, and tombstoned on delete. |
| **Harness-owned vector wiring** | Tier-3 hosts MUST wire the integration RAG stack into memory facades (`UserProfileManager`, session turn index) — agents never open vector DBs directly. |

---

## 3. Three-layer memory model

Production-grade Harness AI separates three cooperating layers:

```text
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER A — Memory Stores (persisted, scoped, governed)                   │
│  STM │ Task KV │ User LTM │ Org Profile │ Knowledge (RAG) │ Trace       │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │ write path
┌───────────────────────────────▼─────────────────────────────────────────┐
│  LAYER B — Memory Lifecycle (when and how to persist)                    │
│  extract → score → dedup → store → index → forget/TTL                   │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │ read API (MemoryView, session, LTM search)
                                ▼
              ┌─────────────────────────────────────────┐
              │  LAYER C — Context Engineering         │
              │  (separate domain — see CONTEXT_ENGINEERING.md) │
              └─────────────────────────────────────────┘
```

| Layer | Phase MEM (Done) | Phase MEM-DEPTH (Done) | Phase MEM-VEC (P0–P1 Done) |
|-------|------------------|------------------------|----------------------------|
| **A — Stores** | Four operational stores + trace + RAG wired | Mongo session parity + entity graph store | LTM + episodic vector indexes wired |
| **B — Lifecycle** | Manual/scheduled consolidation service | Auto job, dedup, episodic, structured summaries | Episodic index on `append_message` |
| **C — Context Engineering** | Documented under MEMORY (legacy) | **Split to [`CONTEXT_ENGINEERING.md`](CONTEXT_ENGINEERING.md)** — `ContextCompiler` delivery Done | CE `SESSION_HISTORY_SEMANTIC` + Nexus recall handles |

---

## 4. Cognitive taxonomy vs Intergrax runtime

Industry systems (LangMem, Mem0, Zep, Letta, CoALA) use a cognitive taxonomy. Intergrax maps it as follows:

| Cognitive type | Purpose | Intergrax store / mechanism | Maturity |
|----------------|---------|----------------------------|----------|
| **Working memory** | Active turn context | `state.base_history` + injected LTM/RAG/tool blocks + `AgentContextBundle` | Strong — `ContextCompiler` |
| **Episodic** | Specific past events / trajectories | `EPISODIC_EVENT` + `SESSION_SUMMARY`; **session turn vector index** (MEM-VEC-2) | Strong — episodic index + CE recall |
| **Semantic** | Stable facts and preferences | `USER_FACT`, `PREFERENCE`, `ORG_FACT` + **LTM vector index** | Strong — LTM wiring + semantic search |
| **Procedural** | How the system should behave | `system_instructions` (user/org profile); Prompt Registry | Minimal — no versioned procedural store |
| **Knowledge** | Document / corpus truth | RAG vectorstore; Graph RAG (documents) | Strong |
| **Task-scoped** | Per-run scratch state | `TaskMemory` + `PolicyScopedMemoryView` | Strong |
| **Trace** | Audit, not agent-mutable | `RunTraceWriter` / `RuntimeEvent` | Strong |

### 4.1 `MemoryKind` tags (entry classification)

```python
# intergrax/memory/user_profile_memory.py
USER_FACT | PREFERENCE | SESSION_SUMMARY | ORG_FACT | POLICY
EPISODIC_EVENT | PROCEDURAL | OTHER
```

These tags classify **LTM entries** — semantic vs episodic vs procedural at the **entry** level (AUDIT-IDEAL-15.2 Done). Temporal validity (`valid_from` / `valid_until`) enforced on retrieval (MEM-DEPTH-5.2 **Done**, 2026-06-17).

---

## 5. Canon §27 → operational stores

Architecture canon §27 defines **five memory types**. Runtime implements **four mutable stores** plus knowledge retrieval and immutable trace:

```text
Canon §27 type              Runtime implementation
──────────────────────────────────────────────────────────────────
1. Task Memory              TaskMemory SQLite KV → MemoryView
2. Agent Local Memory       Same KV under agent/delegation namespaces
3. User / Organization      UserProfileManager + OrganizationProfileManager
4. Long-Term Knowledge      RAG vectorstore (+ Graph RAG for documents)
5. Execution Trace          RunTraceWriter (immutable — not agent memory)

+ Short-term (operational):  SessionManager + SessionStorage (turn-by-turn chat)
```

### 5.1 Store catalog

| Store | Scope key | Module | Agent access | Default persistence |
|-------|-----------|--------|--------------|---------------------|
| **Session (STM)** | `session_id` | `runtime/nexus/session/` | Via Nexus history steps only | SQLite bundle, in-memory fallback, or Mongo `DocumentStoreSessionStorage` (MEM-DEPTH-2.1) |
| **Task KV** | `tenant_id` + `task_id` + namespace | `runtime/task_memory/` | `memory.*` tools via `MemoryView` | SQLite (`INTERGRAX_TASK_MEMORY_DB`) |
| **User LTM** | `tenant_id` + `user_id` | `intergrax/memory/` | Nexus `UserLongtermMemoryStep` | SQLite bundle / Mongo document_store |
| **Org profile** | `org_id` | `runtime/organization/` | Nexus profile steps | SQLite bundle |
| **Shared handoff** | `task_id` | `SharedTaskContext` | `ContextManager` | Task metadata + KV bridge |
| **Knowledge (RAG)** | collection + metadata filters | `intergrax/rag/` | `rag.retrieve` tool / `rag.retrieve` (catalog) | Vector store per integration profile |
| **Session episodic index** | `tenant_id` + `session_id` (+ `user_id`) | `intergrax/memory/` (`SessionTurnIndexService`) | Nexus `run_session_semantic_recall_context` + CE `SessionSemanticRecallProvider` | Vector store — index over session turns, not a replacement for `SessionStorage` |
| **Trace** | `run_id` | `runtime/nexus/tracing/` | Read-only debug APIs | SQLite |

### 5.2 Session vs checkpoint vs task KV

| Concept | Intergrax | Use when |
|---------|-----------|----------|
| **Thread / session** | `SessionManager` + `session_id` | Multi-turn chat in one conversation |
| **Checkpointer** | `SQLiteTaskCheckpointStore` | Resumable UAEP / long-running loops |
| **Scoped KV** | `TaskMemory` + `MemoryView` | Per-task agent scratch, delegation state |

Do **not** store conversational turns in task KV or checkpoint cursors in session storage.

### 5.3 Vector memory index catalog (normative)

Intergrax uses **one vector integration stack** (`EmbeddingManager`, `VectorstoreManager`, optional `RetrievalService` from [`RAG.md`](RAG.md)) but **three logical index domains**. Each domain has distinct metadata, write triggers, and CE read paths. Hosts MUST NOT mix domains in a single undifferentiated collection.

| Index domain | Indexed payload | Primary store (source of truth) | Metadata filter keys (minimum) | Write trigger |
|--------------|-----------------|--------------------------------|--------------------------------|---------------|
| **`knowledge`** | Document / corpus chunks | RAG ingest pipelines | `tenant_id`, collection, `workspace_id`, … | `rag.ingest`, attachment ingest |
| **`ltm`** | `UserProfileMemoryEntry.content` | `UserProfileStore` (SQLite / Mongo) | `user_id`, `entry_id`, `kind`, `deleted` | `add_memory_entry`, consolidation, `ltm.write_fact` |
| **`episodic`** | Session turn text (user / assistant) | `SessionStorage` (SQLite / Mongo) | `tenant_id`, `session_id`, `user_id`, `entry_id`, `role`, `deleted` | `SessionManager.append_message` (MEM-VEC-2.2) |

**Rules:**

1. **Knowledge ≠ LTM ≠ episodic** — document RAG must not silently absorb user facts; episodic turns must not replace LTM extraction for stable cross-session facts.
2. **Collection isolation** — `MemoryProfile.vector_index_namespace` or integration-profile defaults derive separate collection names per domain; lab hosts use `"{tenant_id}:ltm"` and `"{tenant_id}:episodic"` unless overridden.
3. **Tombstones** — logical deletes in the primary store MUST propagate vector tombstones (`deleted=1` or `delete(ids)`).
4. **Agents** — Tier-2 agents consume semantic memory only via Nexus steps, `ltm.search`, or future `memory.semantic_search` skill runtime — never via direct vector SDK calls.

**As-built (2026-06-17):** LTM and episodic vector indexes wired via `memory_vector_wiring.py`; `retrieval_service` injected into `UserProfileManager`; `vector_index_namespace` enforced via `collection_name` metadata; Tier-3 hosts inject RAG stack into profile manager and episodic index.

---

## 6. Write path — memory lifecycle

### 6.1 Write paths by store

```mermaid
flowchart LR
    subgraph agents [Tier-2 Agents]
        A[Agent UAEP step]
    end

    subgraph tier1 [Tier-1 Nexus]
        MV[PolicyScopedMemoryView]
        SM[SessionManager.append]
        CS[SessionMemoryConsolidationService]
        UPM[UserProfileManager]
    end

    subgraph stores [Stores]
        TK[Task KV SQLite]
        SS[Session Storage]
        UL[User LTM]
    end

    A -->|memory.write tool| MV
    MV -->|policy + hooks| TK
    A -->|chat turn| SM
    SM --> SS
    CS -->|LLM extract| UPM
    UPM --> UL
```

| Store | Who writes | Policy |
|-------|------------|--------|
| Task KV | Agent via `memory.write` / UAEP | `MemoryWritePolicy`, `BEFORE_MEMORY_WRITE` hook, `scope_boundary` |
| Session | Nexus after each turn | Session lifecycle coordinator |
| User LTM | `SessionMemoryConsolidationService` or explicit API | LLM extraction; **LTM vector index upsert when RAG stack wired** |
| Session episodic index | Nexus after `append_message` (MEM-VEC-2.2) | Embed + upsert turn; tombstone on message delete |
| Org profile | `OrganizationProfileManager` | Admin / consolidation paths |
| RAG knowledge | Ingest pipelines / tools | Not agent-silent mutation of user profile |

### 6.2 Consolidation flow

`SessionMemoryConsolidationCoordinator` triggers `SessionMemoryConsolidationService`:

| Trigger | Condition |
|---------|-----------|
| `MID_SESSION` | Every N user turns (configurable interval + cooldown) |
| `CLOSE_SESSION` | Session close when `user_id` present |

Extraction produces up to `max_facts` `USER_FACT`, `max_preferences` `PREFERENCE`, optional `SESSION_SUMMARY`. May regenerate `system_instructions` via `UserProfileInstructionsService`.

**As-built:** `MemoryConsolidationJob` + `consolidation_mode` (`manual` \| `scheduled` \| `auto`) on `MemoryProfile`. ADR: [`ADR-MEM-001`](../adr/entries/2026-06-08/ADR-MEM-001.md).

### 6.3 Retention and forget

| Mechanism | Applies to |
|-----------|------------|
| `MemoryProfile.retention_days` | Session + task stores |
| `should_forget_stm_record` | Task KV namespaces prefixed `stm:` |
| `UserProfileMemoryEntry.deleted` | Logical delete + vector index tombstone |
| Session purge | Storage backend TTL (when configured) |
| Episodic index purge | `retention_days` + session delete cascades tombstone all `entry_id` for `session_id` (MEM-VEC-2.2) |

### 6.4 LTM vector index write path (as-built — MEM-VEC-1)

When `UserProfileManager.is_longterm_rag_enabled()` is true, every durable LTM mutation upserts or tombstones the **`ltm`** index domain:

```text
add_memory_entry / update_memory_entry / remove_memory_entry
  → UserProfileStore (source of truth)
  → UserProfileManager._index_upsert_entry | _index_delete_entry
  → VectorstoreManager (metadata: user_id, entry_id, kind, deleted)
```

**Normative Tier-3 contract (MEM-VEC-1.1):** if `MemoryProfile.enable_long_term_memory` is true **and** the host integration profile resolves a vector store + embedding manager, `build_session_manager_from_environment()` **MUST** construct:

```python
UserProfileManager(
    store,
    embedding_manager=rag_stack.embedding_manager,
    vectorstore_manager=rag_stack.vectorstore_manager,
    retrieval_service=rag_stack.retrieval_service,  # preferred — shared metadata scope with RAG
)
```

The **same** `UserProfileManager` instance MUST be exposed on `ToolWiringContext.user_profile_manager` for `ltm.search` / `ltm.write_fact` (MEM-VEC-1.2).

### 6.5 Session turn vector index write path (as-built — MEM-VEC-2)

Session messages remain in `SessionStorage`. When `MemoryProfile.enable_session_vector_index` is true, Nexus additionally indexes each appended turn into the **`episodic`** domain:

```mermaid
flowchart LR
    SM[SessionManager.append_message]
    SS[(SessionStorage)]
  IDX[SessionTurnIndexService]
    VS[(Vector store episodic)]

    SM --> SS
    SM --> IDX
    IDX -->|embed + upsert| VS
```

| Field | Requirement |
|-------|-------------|
| Chunk unit | One `ChatMessage` per index row (no cross-turn merge at write time) |
| `entry_id` | Stable id from `session_messages.entry_id` — upsert key |
| Roles indexed | `user`, `assistant` by default; `system` / `tool` configurable via `MemoryProfile.session_index_roles` |
| Async | Indexing is **synchronous** on `append_message` in the default adapter; CE tolerates empty episodic hits with `session_vector_recall_reason=no_hits` |

Consolidation (`SessionMemoryConsolidationService`) remains the path for **cross-session semantic facts**; episodic index answers **“what was said in this or prior sessions?”** without waiting for consolidation.

---

## 7. Read path — context assembly (moved)

> **Canonical spec:** [`architecture/CONTEXT_ENGINEERING.md`](CONTEXT_ENGINEERING.md) §7–§14.  
> This section retains a **summary** for MEMORY↔CE navigation only.

## 7.1 Read path — context assembly today (summary)

### 7.1 Nexus session turn pipeline

```mermaid
sequenceDiagram
    participant Client
    participant Engine as AgentEngine
    participant HL as HistoryLayer
    participant Steps as Runtime Steps
    participant LLM

    Client->>Engine: RuntimeRequest
    Engine->>HL: build_base_history
    Note over HL: token budget, SUMMARIZE_OLDEST / TRUNCATE
    HL-->>Engine: state.base_history
    Engine->>Steps: run_longterm_memory_context
    Engine->>Steps: run_session_semantic_recall_context
    Engine->>Steps: RagStep / rag.retrieve
    Engine->>Steps: HistoryStep
    Note over Steps: CE SessionSemanticRecallProvider reads session_vector_hits handle
    Steps-->>Engine: messages[]
    Engine->>LLM: generate (budget check per layer)
```

| Runtime invocation | Source | Injection |
|------|--------|-----------|
| `run_longterm_memory_context` | `UserProfileManager.search_longterm_memory` — **`ltm`** index | LTM block via prompt builder; CE `LONGTERM_MEMORY` fragments |
| `run_session_semantic_recall_context` | `SessionManager.search_session_semantic_recall` — **`episodic`** index | Populates `session_vector_hits` for CE `SessionSemanticRecallProvider` |
| `rag.retrieve` (catalog) | `RetrievalService` — **`knowledge`** index | Evidence chunks |
| `HistoryStep` / `HistoryLayer` | `state.base_history` — chronological session store | Conversation turns (recent tail) |
| Websearch | `websearch.query` | Evidence (when enabled) |

Legacy doc names `UserLongtermMemoryStep` / `SessionSemanticRecallStep` map to the runtime functions above plus CE providers (CE-VEC-1).

### 7.1.1 Semantic session recall vs chronological history

| Mechanism | Data source | Selection strategy | When used |
|-----------|-------------|-------------------|-----------|
| **Chronological history** | `SessionStorage` via `HistoryLayer` | Recent turns + token budget; `SUMMARIZE_OLDEST` / `TRUNCATE_OLDEST` | Default; short sessions |
| **LTM semantic search** | `ltm` vector index | Query = current user message; top_k + score threshold | Cross-session facts; user returns after days |
| **Episodic semantic recall** | `episodic` vector index | Query = current user message; filter `session_id` and/or `user_id`; top_k | Long sessions; prior sessions when `include_cross_session_episodic` |
| **RAG knowledge** | `knowledge` vector index | Corpus retrieval with tenant/workspace filters | Document Q&A |

**Context Engineering contract:** semantic recall hits MUST enter assembly as attributable fragments (`ContextFragmentSource.SESSION_HISTORY_SEMANTIC` or `LONGTERM_MEMORY`) with `source_id` = `entry_id` — see [`CONTEXT_ENGINEERING.md`](CONTEXT_ENGINEERING.md) §7.2, §14.2. `HistoryLayer` remains the chronological fallback; CE degradation ladder drops lowest-scored optional fragments before mandatory user turn.

**Pipeline order (as-built):** `run_longterm_memory_context` → `run_session_semantic_recall_context` → `rag.retrieve` (catalog) → `HistoryStep` → `CompileContextStep` / CE collect.

### 7.2 Graph node context (`ContextManager`)

For multi-agent `ExecutionGraph` nodes, `ContextManager.build_agent_context()` assembles:

- Task message and node spec
- Prior dependency outputs (`TaskContextAssemblyOptions` summary tier)
- Shared context reads (`SharedTaskContext`)
- `ContextBudgetPolicy` trim on composed message
- Provenance in `AgentContextBundle`

### 7.3 Budgeting — global allocator vs per-step caps

**Global allocator (Done — ADR-MEM-001):** `ContextCompiler` + degradation ladder in [`CONTEXT_ENGINEERING.md`](CONTEXT_ENGINEERING.md) §10 before agent LLM step.

**Per-step local caps (still apply before CE collect):**

| Layer | Limit mechanism |
|-------|-----------------|
| `HistoryLayer` | ~⅔ input budget for history; `history_compression_strategy` |
| `run_longterm_memory_context` | `max_longterm_entries_per_query`, `max_longterm_tokens` |
| `ContextBudgetPolicy` | `max_chars` + token estimate; char-cut fallback |
| `TaskContextAssemblyOptions` | `max_prior_chars`, summary tiers |

**AUDIT-IDEAL-16.1 / 16.2** (drift monitoring, semantic compression profile flags) — owner: [`CONTEXT_ENGINEERING.md`](CONTEXT_ENGINEERING.md) §11; status **Done** in master register.

---

## 8. Compression and degradation strategies

### 8.1 Strategy matrix

| Strategy | When | Module | Trace |
|----------|------|--------|-------|
| `OFF` | Debug / short sessions | `HistoryLayer` | History diag |
| `TRUNCATE_OLDEST` | History over budget; summarization failed | `HistoryLayer` | `effective_strategy` |
| `SUMMARIZE_OLDEST` | Long sessions; tokenizer available | `HistoryLayer` + `HistorySummaryPromptBuilder` | `summary_used=true` |
| `HYBRID` | Aggressive long-horizon | `HistoryLayer` | Combined |
| Summary tiers | Graph prior outputs | `ContextManager` | `AgentContextBundle.summary_tier` |
| Char/token trim | Composed agent message | `context_budget.py` | `CONTEXT_TRIMMED` |
| LTM top_k + threshold | Query relevance | `UserLongtermMemoryStep` | `UserLongtermMemorySummaryDiagV1` |
| RAG top_k + rerank | Knowledge retrieval | `RetrievalService` | RAG trace |

### 8.2 Target degradation ladder (MEM-DEPTH)

Apply in order until invariant §1.3 is satisfied:

```text
1. FULL fidelity (all sources within budget)
2. Lower summary tier on graph priors (FULL → SUMMARY_ONLY → MINIMAL)
3. Reduce LTM/RAG top_k
4. SUMMARIZE_OLDEST on session history
5. TRUNCATE_OLDEST on session history
6. Drop lowest-scored context fragments (relevance order)
7. Tokenizer-aware hard trim (last resort — never silent char-cut)
```

Each step MUST emit diagnostics with `degradation_step` and bytes/tokens removed.

---

## 9. Intelligent strategy selection

### 9.1 Decision inputs

The Harness selects memory and compression strategies from:

| Input | Profile field / config |
|-------|------------------------|
| Host memory flags | `MemoryProfile` on `ApplicationEnvironmentProfile` |
| Context preferences | `ContextProfile.decision` (`ContextDecisionProfile`) |
| Per-request override | `RuntimeRequest.history_compression_strategy` |
| Model capacity | `llm_adapter.context_window_tokens` |
| Task shape | Single chat vs `ExecutionGraph` vs delegation |
| Query emptiness | LTM step skipped on empty query |

### 9.2 `ContextDecisionProfile` (declarative policy)

```python
# intergrax/applications/contracts/environment_profile.py
include_session_history: bool = True
prefer_longterm_memory: bool = True
prefer_rag_when_enabled: bool = True
max_memory_entries_in_context: int = 8
```

**Enforced** by `ContextCompiler` via `ContextDecisionProfile` on `RuntimeConfig`.

### 9.3 Selection matrix (normative)

| Situation | Primary memory source | Compression | Secondary |
|-----------|----------------------|-------------|-----------|
| Short chat (&lt; 30% context) | Full session history | `OFF` or light | LTM if `prefer_longterm_memory` |
| Long chat (&gt; 50% context) | Session summary + recent tail | `SUMMARIZE_OLDEST` | LTM top_k reduced |
| Document Q&A | RAG retrieval | RAG top_k within budget | Minimal history |
| Multi-agent graph node | Prior outputs + shared context | Summary tier per policy | Task KV via tools |
| Delegated explore child | Isolated namespace reads | Child budget; synthesis return | Parent gets summary only |
| User returns after days | LTM semantic search (`ltm` index) | Session history empty or short | Episodic cross-session recall when `include_cross_session_episodic` |
| Long active session (> 50% context) | Episodic semantic recall + recent tail | `SUMMARIZE_OLDEST` on remainder | LTM top_k reduced |
| Codebase-scale task | RAG + workspace tools | Retrieval-first; no full dump | Explore delegation (target) |
| Regulated / high-risk | Policy + minimal context | `MINIMAL` tier; explicit citations | HITL before LTM write |

### 9.4 When to use which store (authoring guide)

| Need | Store | Do not use |
|------|-------|------------|
| "Remember this for this run only" | Task KV (`memory.write`) | Session |
| "Remember across sessions for this user" | User LTM (consolidation or explicit entry) | Task KV |
| "Team tone and constraints" | Org profile | User LTM |
| "Facts from uploaded PDFs" | RAG collection | User LTM (unless extracted fact) |
| "What happened in run #4521" | Trace / debug journal | Mutable memory |
| "Child agent scratch pad" | Delegation namespace | Parent namespace |

---

## 10. Delegation and explore pattern

### 10.1 Delegation memory (Done — R-Delegate)

Child agents read/write under:

```text
task_id/delegation/{node_id}/
```

via `PolicyScopedMemoryView`. Parent receives bounded context via `ContextManager` — not raw child history.

### 10.2 Explore pattern (Done — MEM-DEPTH-4.2 + MEM-LC-8)

For wide codebase or corpus search:

```mermaid
flowchart TB
    Parent[Parent agent context]
    Explore[Explore delegation child]
    Search1[Parallel RAG / grep / workspace]
    Search2[Parallel RAG / grep / workspace]
    Synth[Synthesis-only return]
    Parent -->|DelegationSpec| Explore
    Explore --> Search1
    Explore --> Search2
    Search1 --> Synth
    Search2 --> Synth
    Synth -->|findings summary only| Parent
```

Properties:

- Child runs in **isolated context window**
- Parallel searches do not bloat parent history
- Return payload is **structured findings**, not raw file dumps

Target: MEM-DEPTH-4.1, MEM-DEPTH-4.2 — **wired** via `explore_integration.py` in `graph_executor`.

---

## 11. Configuration surfaces

### 11.1 `MemoryProfile`

```python
enable_user_memory: bool
enable_org_memory: bool
enable_long_term_memory: bool
enable_task_memory: bool
enable_session_vector_index: bool = False          # MEM-VEC-2 — episodic turn indexing (Done)
include_cross_session_episodic: bool = False       # episodic search across sessions for same user
session_index_top_k: int = 8
session_index_score_threshold: float | None = None
vector_index_namespace: str | None = None          # collection prefix; default derived from tenant_id
session_index_roles: tuple[str, ...] = ("user", "assistant")
retention_days: int | None
scope_boundary: str = "tenant"
consolidation_mode: Literal["manual", "scheduled", "auto"]
```

Mapped to `RuntimeConfig` via `memory_runtime_bridge.py` / `materialize_runtime_config`. Vector-index flags require a resolved integration vector store — hosts without vector backend MUST fail closed (`reason=vector_backend_unavailable`) rather than silently disabling semantic recall while flags are true (MEM-VEC-1.4 gate).

### 11.2 `ContextProfile`

```python
assembly_options: TaskContextAssemblyOptions  # max_prior_chars, summary tier defaults
budget_policy: ContextBudgetPolicy | None   # max_chars, max_tokens_estimate
decision: ContextDecisionProfile
enable_rag: bool
enable_websearch: bool
```

### 11.3 `RuntimeConfig` LTM limits

| Field | Role |
|-------|------|
| `enable_user_longterm_memory` | Gates `UserLongtermMemoryStep` |
| `max_longterm_entries_per_query` | top_k |
| `longterm_score_threshold` | Minimum relevance |
| `max_longterm_tokens` | Injection cap |

### 11.4 Tier-3 wiring entry points

| Function | Role |
|----------|------|
| `resolve_memory_platform_wiring()` | Session + profile stores from integration profile |
| `build_session_manager_from_environment()` | SessionManager + profile managers — **MUST accept optional `rag_stack` for vector-enabled `UserProfileManager`** (MEM-VEC-1.1) |
| `wire_task_memory_from_profile()` | Task KV database path |
| `materialize_runtime_config()` | Profile → RuntimeConfig bridge |
| `build_runtime_context_from_environment()` | Single entry — MUST pass shared RAG stack into memory + tool wiring (MEM-VEC-1.2) |

### 11.5 Memory store and vector index plugins

| Protocol | EP group | Factory | Replaces |
|----------|----------|---------|----------|
| `UserProfileStorePlugin` | `intergrax.memory_stores` | `create_user_profile_store(**kwargs)` | Default SQLite / Mongo / in-memory LTM store |
| `SessionStoragePlugin` | `intergrax.memory_stores` | `create_session_storage(**kwargs)` | Default session persistence |
| `SessionTurnIndexStore` (MEM-VEC-2.1; plugin EP MEM-VEC-3.1 **Done**) | `intergrax.memory_stores` | `create_session_turn_index(**kwargs)` | Default: `VectorSessionTurnIndexStore` over `VectorstoreManager` |

Vector **integration** providers (Chroma, pgvector, Qdrant, …) remain in the integrations catalog — memory plugins select **how** indexes are written, not which vendor SDK is used. Custom Tier-3 hosts register EP plugins; Tier-2 agents still use Nexus APIs and tools only.

---

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
