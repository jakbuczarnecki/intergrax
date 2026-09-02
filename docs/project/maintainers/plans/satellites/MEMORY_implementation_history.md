# MEMORY - implementation history + LC closeout

**Parent hub:** [`MEMORY.md`](../MEMORY.md)

> **Plan ownership:** Implementation phases and LC closeout below. Historical audit findings/verdicts archived at [docs/audit_results/legacy/plan-audit-history/MEMORY_implementation_history.md](../../../../audit_results/legacy/plan-audit-history/MEMORY_implementation_history.md).


## Phase MEM-VEC - Vector memory integration (Band 2aw)

**Status:** **Done** (2026-06-17) - MEMV1/MEMV2 **Done**; MEMV3 plugin + skill runtime **Done**.
**Architecture:** [`architecture/MEMORY.md`](../architecture/MEMORY.md) §5.3, §6.4–6.5, §7.1.1, §11.5
**Cross-plan:** [`plan/CONTEXT_ENGINEERING.md`](CONTEXT_ENGINEERING.md) - `SESSION_HISTORY_SEMANTIC` fragment source (CE-VEC-1)
**ADR:** [`ADR-MEM-002`](../adr/entries/2026-06-14/ADR-MEM-002.md) - three-domain vector catalog (Accepted 2026-06-14)
**Goal:** Semantic vector recall for **LTM facts** and **session turns** wired end-to-end into Tier-3 hosts and Nexus CE pipeline - without agents touching vector SDKs.

**Success gate:** P0 **Done**; lab + reference hosts pass `test_memory_vector_wiring_gate`; `enable_long_term_memory` + vector integration ⇒ `UserLongtermMemoryStep` returns hits in integration test.

**Delivery rule:** One `MEM-VEC-*` ID per PR → update master table + paydown log → `pytest -m gate` green.

### 6.2aw Phase MEM-VEC execution order (Band 2aw)

| Wave | IDs | Count | Focus |
|------|-----|-------|--------|
| MEMV0 | MEM-VEC-0.1–0.2 | 2 | Canon + ADR |
| MEMV1 | MEM-VEC-1.1–1.4 | 4 | **P0** - LTM vector wiring + tool binding + gate |
| MEMV2 | MEM-VEC-2.1–2.4 | 4 | **P1** - Episodic session index + Nexus recall step |
| MEMV3 | MEM-VEC-3.1–3.2 | 2 | **P2** - Plugin contract + `memory.semantic_search` runtime |
| **Total** | | **12** | |

Work **MEM-VEC-1.*** before MEM-VEC-2.* - LTM wiring is a small diff that unblocks existing `UserProfileManager` code.

### MEM-VEC - Master deliverables register

| ID | Deliverable | Priority | Status | Module / test |
|----|-------------|----------|--------|---------------|
| MEM-VEC-0.1 | **Architecture canon** - §5.3 vector index catalog, §6.4–6.5 write paths, §7.1.1 semantic recall, §11.5 plugins | P0 | **Done** | `architecture/MEMORY.md` |
| MEM-VEC-0.2 | **ADR-MEM-002** - three-domain vector catalog, episodic vs LTM vs knowledge, collection isolation, tombstone rules | P0 | **Done** | `docs/project/technical/adr/entries/2026-06-14/ADR-MEM-002.md` |
| MEM-VEC-1.1 | **`build_session_manager_from_environment` accepts `rag_stack`** - inject `embedding_manager`, `vectorstore_manager`, `retrieval_service` into `UserProfileManager` | **P0 Critical** | **Done** | `applications/_shared/memory_wiring.py`, `memory_vector_wiring.py` |
| MEM-VEC-1.2 | **`build_runtime_context_from_environment` passes shared RAG stack** into memory wiring + sets `ToolWiringContext.user_profile_manager` to same manager instance | **P0 Critical** | **Done** | `runtime_config_bridge.py`, `environment_wiring.py` |
| MEM-VEC-1.3 | **Integration gate** - lab profile with sqlite + inmemory vector: consolidation write → `search_longterm_memory` hits | **P0** | **Done** | `tests/integration/applications/test_memory_vector_ltm_wiring.py` |
| MEM-VEC-1.4 | **`MemoryProfile` vector flags** - `vector_index_namespace`; fail-closed when flags true but no vector backend | P0 | **Done** | `environment_profile.py`, `memory_runtime_bridge.py`, `memory_vector_wiring.py` |
| MEM-VEC-2.1 | **`SessionTurnIndexStore` protocol** + default adapter over `VectorstoreManager` (`episodic` metadata schema) | P1 | **Done** | `intergrax/memory/contracts/session_turn_index.py`, `session_turn_index_service.py` |
| MEM-VEC-2.2 | **Write path** - `SessionTurnIndexService` on `append_message`; tombstone on delete; `enable_session_vector_index` on `MemoryProfile` | P1 | **Done** | `session_manager.py`, `memory_runtime_bridge.py` |
| MEM-VEC-2.3 | **`SessionSemanticRecallStep`** - semantic search episodic index; inject in CE provider before history layer; trace diag | P1 | **Done** | `memory_context_invocation.py`, `tool_runtime.py`, `uaep.py` |
| MEM-VEC-2.4 | **CE fragment source** - `SESSION_HISTORY_SEMANTIC` in `ContextCompiler` classification + degradation ladder | P1 | **Done** (CE-VEC-1) | `context/contracts.py`, `runtime_state_handle_bridge.py` |
| MEM-VEC-3.1 | **`SessionTurnIndexStorePlugin`** EP + fixture package | P2 | **Done** | `intergrax.memory_stores`, `tests/fixtures/plugin_packages/session_turn_index_plugin` |
| MEM-VEC-3.2 | **`memory.semantic_search` skill runtime** - delegates to `ltm.search` + episodic recall (not prompt-only) | P2 | **Done** | `tools/providers/memory`, `skills/providers/memory` |

### MEM-VEC - Paydown log

| Date | ID | Summary |
|------|-----|---------|
| 2026-06-12 | MEM-VEC-0.1 | Canon: three-domain vector catalog, wiring contract, episodic recall target state |
| 2026-06-14 | MEM-VEC-0.2 | ADR-MEM-002: three-domain vector catalog accepted |
| 2026-06-14 | MEM-VEC-1.1–1.4 | LTM vector wiring, fail-closed gate, integration test |
| 2026-06-17 | MEM-VEC-3.1–3.2 | SessionTurnIndexStorePlugin EP + memory.semantic_search tool runtime |
| 2026-06-14 | MEM-VEC-2.1–2.4 | Episodic index write/recall + CE handle population |

**Explicitly out of NOW:** merging `knowledge` + `ltm` + `episodic` into one collection; Mem0 SaaS replacement; Neo4j as default episodic backend.

---

### 6.2bf Phase CTX execution order (Band 2n - closed 2026-06-02)

**Status:** **Done** · register: [Phase CTX](plan/MEMORY.md) · queue: [§6.1f](.#61f-harness-implementation-queue--context-engineering-closeout-closed)

| Step | ID | Deliverable | Priority |
|------|-----|-------------|----------|
| 1 | CTX-1 | `context_runtime_bridge` | Critical |
| 2 | CTX-2 | `context_wiring` + Nexus factory wire | High |
| 3 | CTX-DOC.1–2 | Appendix L + plan sync | Low |

**RAG closeout (Band 2m):** canonical register in [`plan/RAG.md`](plan/RAG.md).

---

### 6.2aa Phase MEM execution order (Band 2h - closed)

**Status:** **Done** (2026-06-02) · **48/48 Done** · canonical register: [Phase MEM - Master deliverables register](.#mem--master-deliverables-register-all-48-tasks).

Work **one MEM ID per PR**; after each step update the MEM master table + paydown log; keep §6.1 scripts green. **Start with MEM-1.*** before MEM-3/MEM-7 - bridge must exist before plugins/hooks.

| Wave | IDs | Count | Focus |
|------|-----|-------|--------|
| MEM0 | MEM-0.1–MEM-0.4, MEM-PAR.1, MEM-CHk.1, MEM-PERS.1, MEM-ST.1, MEM-OBS.2 | 9 | Register, audit baseline, parity tables (MEM-OBS.2 Done) |
| MEM1 | MEM-1.1–MEM-1.6, MEM-2.1–MEM-2.3 | 9 | **P0** - H-APP bridge + SQLite user LTM |
| MEM2 | MEM-3.*, MEM-4.*, MEM-5.*, MEM-CTX.1, MEM-DOC.1–6, MEM-GRAPH.1, MEM-TASK.* | 18 | **P1** - gates, plugins, context docs |
| MEM3 | MEM-6.*, MEM-7.*, MEM-OBS.1, MEM-DOC.5, MEM-CTX.2, MEM-PERS.2, MEM-ST.4 | 9 | **P2** - retention, hooks, metrics |
| MEM4 | MEM-8.*, MEM-9.1, MEM-PERS.3 | 4 | **P3** - product RFCs |
| **Total** | | **48** | |

**Success gate:** P0 + P1 **Done**; H-APP.4.3 **Done**; user LTM durable on sqlite lab profile; `MemoryProfile` drives all reference hosts.

**Explicitly out of NOW:** K.1/K.2, Mem0 auto-ingest ship (MEM-8.2), entity graph implementation (MEM-9.1 beyond RFC).

### 6.2ab Phase MEM-DEPTH execution order (Band 2am - closed)

**Status:** **Done** (2026-06-08) · **26/26 Done** · canonical register: [Phase MEM-DEPTH - Master deliverables register](.#mem-depth--master-deliverables-register-all-26-tasks).

Work **one MEM-DEPTH ID per PR**; after each step update the MEM-DEPTH master table + paydown log; keep §6.1 scripts green. **Start with MEM-DEPTH-0.3 (ADR) then MEM-DEPTH-1.*** - architecture decision before Context Compiler code.

| Wave | IDs | Count | Focus |
|------|-----|-------|--------|
| MEMD0 | MEM-DEPTH-0.1–0.4 | 4 | Canon doc, plan register, ADR, cross-links |
| MEMD1 | MEM-DEPTH-1.1–1.6 | 6 | **P0** - Context Compiler + never-overflow |
| MEMD2 | MEM-DEPTH-2.1–2.2 | 2 | **P1** - Mongo session persistence parity |
| MEMD3 | MEM-DEPTH-3.1–3.5 | 5 | **P1** - Lifecycle automation |
| MEMD4 | MEM-DEPTH-4.1–4.3 | 3 | **P1** - Explore / discovery |
| MEMD5 | MEM-DEPTH-5.1–5.6 | 6 | **P2/P3** - Entity graph, temporal validity, quality gates |
| **Total** | | **26** | |

**Success gate:** P0 + P1 **Done**; never-overflow acceptance green; Memory Layer **L3+** on FAUDIT re-run.

**Explicitly out of NOW:** K.1/K.2, Mem0 SaaS, entity graph ship without §6.3 decision (MEM-DEPTH-5.1).

---

## Phase MEM - Memory Platform Completion

**Status:** **Done** (2026-06-02) - **48/48** deliverables; gate **571 passed**.
**Prerequisites:** Phases **I** (TaskMemory), **R-Context**, **H-APP** (profile models), **DX-5.7** (ops:memory hints) **Done**; **H-APP.4.3** closed via **MEM-1.***.
**Goal:** Close every gap from the **memory platform audit** - short-term session, user/org LTM, task KV, context compression, H-APP→runtime wiring, persistence, recovery, observability, developer hooks, and market-parity documentation - **without** Band 3 product agents (K.1/K.2) or Mem0-like SaaS product layer (MEM-8 deferred P3).
**Priority ladder:** **Band 2h** (§4.0) - **closed**; default queue = §6.1 maintenance.
**Execution order:** [§6.2aa](.#62aa-phase-mem-execution-order-band-2h--closed).
**Canon refs:** §27 Memory model · §28.1 Context assembly · §42.35 MemoryView · Appendix G in [`guides/AGENT_CREATION_GUIDE.md`](guides/AGENT_CREATION_GUIDE.md).

**Delivery rule:** One `MEM-*` ID per PR → update status in tables below + paydown log → `pytest -m gate` + §6.1 audit scripts green.

**Audit verdict (baseline - preserve as acceptance context):**

| Area | Maturity (1–5) | Audit comment | Close via |
|------|----------------|---------------|-----------|
| Task memory (KV, delegation, handoff) | **4/5** | Best-in-repo; SQLite, policy, events | MEM-DOC.*, MEM-TASK.* (docs + lab policy) |
| Context / LLM window | **3,5/5** | Budget + assembly + history summarization; weak tests; in-memory session default | MEM-1.*, MEM-5.*, MEM-CTX.* |
| Short-term session (STM) | **3/5** | Model OK; production path often in-memory | MEM-1.3, MEM-4.1, MEM-DOC.1 |
| User LTM | **2,5/5** | Logic exists; no durable store in repo | MEM-2.*, MEM-4.2 |
| Organization memory | **2,5/5** | SQLite org profile; not full org memory product | MEM-1.4, MEM-DOC.3 |
| Consolidation / fact extraction | **2/5** | LLM consolidation; notebooks; few gate tests | MEM-4.2, MEM-8.* (P3) |
| Graph memory (agent sense) | **1/5** | Graph RAG ≠ agent memory | MEM-GRAPH.*, MEM-9.* |
| Developer hooks | **2/5** | MemoryView + events; no memory lifecycle hooks / EP | MEM-3.*, MEM-7.* |
| H-APP → runtime config | **4/5** | Bridge **Done** via MEM-1.* | MEM-DOC.* maintenance |
| Declarative env config | **4/5** | MemoryProfile wired via H-APP bridge | MEM-1.* |
| Memory observability | **4/5** | MEMORY_* / CONTEXT_* events + memory SLO metrics baseline | MEM-OBS.* |

**Overall platform memory score: ~3,5/5** - Tier-1 architecture closed for harness; product Mem0/Zep layer remains **§6.3** optional.

**Out of scope (explicit):** K.1/K.2 business memory; hosted Mem0/Zep replacement SaaS; Neo4j entity graph as default user memory (MEM-9 = design RFC only); Redis/Postgres session backends as shipped defaults (MEM-PERS.3 spike P3).

```text
Wave MEM0 - Register, audit baseline, conceptual + parity docs (9 tasks)
Wave MEM1 - P0 bridge + SQLite user LTM (9 tasks) - closes H-APP.4.3 gap
Wave MEM2 - P1 plugins, gates, context docs, graph clarification (18 tasks)
Wave MEM3 - P2 retention, hooks, SLO metrics, optional backends (9 tasks)
Wave MEM4 - P3 product memory layer + entity graph RFC (4 tasks)
Total: 48
```

### MEM - Conceptual model (canon §27 vs runtime)

Canon §27 defines **5 memory types**:

1. Task Memory
2. Agent Local Memory
3. User / Organization Memory
4. Long-Term Knowledge Memory
5. Execution Trace Memory

Runtime maps these to **four operational stores** (+ trace + RAG - not memory layers):

```text
Short-term:     SessionManager + SessionStorage; optional ConversationalMemory (FIFO)
Task-scoped:    TaskMemory SQLite KV → PolicyScopedMemoryView → SharedTaskContext handoff
User / Org LTM: UserProfileManager + entries; OrganizationProfile SQLite
Knowledge:      RAG vectorstore; Graph RAG (document graph - NOT agent entity memory)
Trace:          RunTraceWriter / RuntimeEvents (immutable audit, not agent-mutable memory)
```

**MemoryKind taxonomy (as-built):** LTM entries use `MemoryKind` tags including `EPISODIC_EVENT` and `PROCEDURAL` (AUDIT-IDEAL-15.2 Done). Cognitive episodic/semantic/procedural **stores** remain mapped to session index + LTM + `system_instructions` - not separate Mem0-style modules. Temporal validity on facts deferred: MEM-DEPTH-5.2.

### MEM - Persistence backend matrix (as-built)

| Layer | In-memory | SQLite | Postgres | Redis | Mongo |
|-------|-----------|--------|----------|-------|-------|
| Task KV | test | prod path (`INTERGRAX_TASK_MEMORY_DB`) | - | - | - |
| Session | lab SQLite via bridge | bundle path | - | - | `DocumentStoreSessionStorage` (MEM-DEPTH-2.1) |
| User profile LTM | test | bundle (`SQLiteUserProfileStore`) | - | - | optional `DocumentStoreUserProfileStore` (MEM-PERS.2) |
| Org profile | test | bundle | - | - | - |
| Checkpoints (≠ memory) | - | yes | - | - | - |
| Trace / events | test | yes | - | - | - |

**SQLite integration bundle** (`create_sqlite_integration`) = lab hub (trace, events, task_memory, session, org profile, checkpoints) - coherent for dev; **not** multi-tenant production scale.

**Recovery semantics (target documentation - MEM-DOC.4):**

| Layer | Recovery key | Works when | Broken today |
|-------|--------------|------------|--------------|
| Task memory | `tenant_id` + `task_id` + namespace | SQLite enabled | - |
| Session | `session_id` | SQLite SessionStorage when relational_store=sqlite | - |
| User LTM | user id | SQLite bundle or Mongo document_store | - |
| Long-running | checkpoint store | SQLite | separate from conversational memory |
| Org profile | org id | SQLite bundle | - |

### MEM - Market parity traceability (MEM-PAR.1)

| Capability | LangGraph | Mem0 / Zep | Intergrax today | Target ID |
|------------|-----------|------------|-----------------|-----------|
| Thread / session persistence | Checkpointer | Session + graph | Session + checkpoint **separate** | MEM-DOC.1, MEM-1.3 |
| Scoped KV per run | Store API | - | TaskMemory + MemoryView ✅ | MEM-DOC.2 |
| Auto fact extraction | - | core | Consolidation service, **manual trigger** | MEM-4.2, MEM-8.* |
| Entity graph memory | - | Zep ✅ | ❌ (RAG graph only) | MEM-GRAPH.1, MEM-9.* |
| Vector semantic memory | Optional | ✅ | User LTM via RAG index | MEM-2.* |
| Subagent namespace isolation | Subgraph state | - | delegation namespace ✅ | documented |
| Memory hooks / plugins | Checkpointer swap | API | Event bus only | MEM-3.*, MEM-7.* |
| Declarative env config | Partial | SaaS | MemoryProfile **Done** | MEM-1.* |
| Observability | LangSmith | Dashboard | Trace events ✅; no memory SLO | MEM-OBS.* |

### MEM - User audit checklist → deliverables (MEM-CHk.1)

| Audit question | Answer (as-built) | Deliverable IDs |
|----------------|-------------------|-----------------|
| How is memory **managed**? | Nexus Tier-1; agents via UAEP/MemoryView; profiles via runtime steps | MEM-DOC.2, MEM-1.* |
| **Limited context** handling? | Budget trim + history summarization + LTM limits + summary tiers - no single end-to-end policy | MEM-1.2, MEM-CTX.1, MEM-5.* |
| **Strategy** (summarize, trim)? | `HistoryLayer` + `ContextBudgetPolicy`; trim often char-cut | MEM-5.* |
| **Developer handlers**? | MemoryView, policy, events - no formal memory hooks / plugin catalog | MEM-3.*, MEM-7.* |
| **Where persisted**? | Task/org/session: SQLite (lab); user LTM: in-memory; Redis: not memory layer | MEM-2.*, MEM-PERS.1 |
| **Configuration**? | Env + `MemoryProfile` partial; H-APP bridge incomplete | MEM-1.* |
| **Recovery**? | Task/session by ID if SQLite; user LTM weak | MEM-1.3, MEM-2.*, MEM-DOC.4 |
| **Tests**? | Task/context OK; consolidation/history gaps | MEM-4.*, MEM-5.1, MEM-TEST.* |
| **Observation**? | MEMORY_* / CONTEXT_* events; no product memory metrics | MEM-OBS.* |
| **Graph memory**? | **No** as agent memory - Graph RAG only | MEM-GRAPH.1 |

### MEM - Architecture inventory (existing code - do not rewrite)

| Module | Tier | Role |
|--------|------|------|
| `intergrax/memory` | 0 | ConversationalMemory, UserProfileManager |
| `intergrax/runtime/task_memory` | 1 | Coordinator, MemoryView, SQLite store, delegation |
| `intergrax/runtime/nexus/session` | 1 | SessionManager, InMemory/SQLite storage |
| `intergrax/runtime/nexus/context` | 1 | ContextManager, context_budget, engine_history_layer |
| `intergrax/runtime/user_profile/session_memory_consolidation_service.py` | 1 | LLM session → LTM extraction |
| `intergrax/applications/_shared/runtime_config_bridge.py` | 3 | **Gap:** line ~112 always `InMemorySessionStorage()` |
| `intergrax/applications/_shared/task_memory_wiring.py` | 3 | Enables task DB from profile flags; does not enable LTM/org Nexus steps |
| `intergrax/applications/contracts/environment_profile.py` | 3 | `MemoryProfile`, `ContextProfile` |
| `intergrax/rag/graph` | 0 | Graph RAG retrieval - **not** agent memory |
| `integrations/examples/custom_memory_kv` | 0 | KV plugin example - not wired to TaskMemory |

**Existing gate tests:** `tests/unit/runtime/task_memory`, `tests/acceptance/.../test_acceptance_08_memory_handoff`, context budget, profile steps, sqlite session integration.

**Known gaps in tests:** `engine_history_layer` summarization; E2E LTM consolidation; `MemoryProfile` → runtime wiring; external memory backends; graph-as-memory.

### MEM - Traceability (audit section → task IDs)

| Audit § | Topic | Task IDs |
|---------|--------|----------|
| §1 | Conceptual model (5 types vs 4 stores) | MEM-0.3, MEM-0.4 |
| §2 | Short-term session; InMemory default in bridge; Redis not memory layer | MEM-1.3, MEM-4.1, MEM-DOC.1, MEM-ST.1, MEM-PERS.3 |
| §3 | User LTM; InMemoryUserProfileStore only | MEM-2.*, MEM-4.2 |
| §4 | Org memory; enable_org_memory not mapped | MEM-1.4, MEM-DOC.3 |
| §5 | Task memory strengths; lab default off | MEM-TASK.*, MEM-DOC.2 |
| §6 | Context strategy; budget_policy not mapped | MEM-1.2, MEM-5.*, MEM-CTX.* |
| §7 | Developer hooks; no memory EP | MEM-3.*, MEM-7.*, MEM-DOC.5, MEM-DOC.6 |
| §8 | Persistence matrix; H-APP.4.3 divergence | MEM-PERS.1, MEM-1.*, MEM-REC → MEM-DOC.4 |
| §9 | Observability + test gaps | MEM-OBS.*, MEM-4.*, MEM-5.1 |
| §10 | Graph memory ≠ Graph RAG | MEM-GRAPH.* |
| §11 | Market comparison | MEM-PAR.1 (table above) |
| §12 | Recommended MEM-1..9 backlog | MEM-1.* … MEM-9.* |
| §13 | User checklist | MEM-CHk.1 (table above) |

### MEM - Master deliverables register (all 48 tasks)

#### Wave MEM0 - Register & audit baseline

| ID | Deliverable | Status | Priority | Location / acceptance |
|----|-------------|--------|----------|------------------------|
| MEM-0.1 | **Phase MEM register** in this plan + §6.2aa + §6.1aa + doc model row | **Done** | Low | This section |
| MEM-0.2 | **Audit maturity baseline table** preserved (§Audit verdict above) | **Done** | Low | Do not delete on paydown |
| MEM-0.3 | **Canon §27 → 4 stores** mapping + flow diagram in `AGENT_CREATION_GUIDE` Appendix G | **Done** | Medium | Guide + cross-link §27 |
| MEM-0.4 | Document **`MemoryKind` tags** vs episodic/semantic/procedural (IDEAL vision vs runtime) | **Done** | Low | Guide or canon footnote |
| MEM-PAR.1 | **Market parity traceability table** (LangGraph / Mem0 / Zep) | **Done** | Low | This section §MEM - Market parity |
| MEM-CHk.1 | **User audit checklist** → deliverable mapping (10 questions) | **Done** | Low | This section §MEM - User audit checklist |
| MEM-PERS.1 | **Persistence backend matrix** synced to Appendix G | **Done** | Low | Guide Appendix G + this section |
| MEM-ST.1 | **Document:** Redis `KeyValueCache` = integration cache only - **not** session/LTM memory layer | **Done** | Low | `architecture/INTEGRATIONS.md` or guide |
| MEM-OBS.2 | Baseline: `MEMORY_READ`/`MEMORY_WRITE`, `CONTEXT_*`, `ops:memory` filter | **Done** | - | DX-5.7 · `phase_coverage.py` |

#### Wave MEM1 - P0: H-APP bridge + durable user LTM (closes H-APP.4.3)

| ID | Deliverable | Status | Priority | Location / acceptance |
|----|-------------|--------|----------|------------------------|
| MEM-1.1 | **`materialize_runtime_config` reads `MemoryProfile`** - map `enable_user_longterm_memory`, `enable_task_memory`, retention, scope flags to `RuntimeConfig` | **Done** | **P0 Critical** | `runtime_config_bridge.py` |
| MEM-1.2 | Map **`ContextProfile.budget_policy`** → `RuntimeConfig` context budget fields | **Done** | **P0 Critical** | `runtime_config_bridge.py` |
| MEM-1.3 | **`SessionManager` from integration bundle** - resolve `SQLiteSessionStorage` when sqlite profile active; remove hardcoded `InMemorySessionStorage()` in `build_runtime_context_from_environment` | **Done** | **P0 Critical** | `runtime_config_bridge.py` |
| MEM-1.4 | Map **`MemoryProfile.enable_org_memory`** → `RuntimeConfig.enable_org_profile_memory` | **Done** | **P0** | `runtime_config_bridge.py` |
| MEM-1.5 | **Gate test:** `ApplicationEnvironmentProfile` memory + context → `RuntimeConfig` round-trip | **Done** | **P0** | `tests/unit/applications/test_memory_profile_runtime_bridge.py` |
| MEM-1.6 | **Reconcile H-APP.4.3** - mark **Done** only when MEM-1.1–MEM-1.4 **Done** | **Done** | **P0** | H-APP register row |
| MEM-2.1 | **`SQLiteUserProfileStore`** - mirror `SQLiteOrganizationProfileStore` pattern | **Done** | **P0 Critical** | `intergrax/memory` or `runtime/user_profile` |
| MEM-2.2 | **Wire `SQLiteUserProfileStore`** in sqlite integration bundle + lab/legal/research profiles | **Done** | **P0** | integration bundle wiring |
| MEM-2.3 | **Unit tests:** `UserProfileManager` CRUD + search with SQLite backend (fake RetrievalService) | **Done** | **P0** | `tests/unit/memory` |

#### Wave MEM2 - P1: gates, plugins prep, context docs, graph clarification

| ID | Deliverable | Status | Priority | Location / acceptance |
|----|-------------|--------|----------|------------------------|
| MEM-3.1 | **`UserProfileStore` / `SessionStorage` plugin Protocol** (typed, no Tier-2 imports) | **Done** | P1 | `intergrax/memory/contracts` |
| MEM-3.2 | **Entry point group `intergrax.memory_stores`** + `bootstrap_memory_stores()` | **Done** | P1 | Mirror P-Ext pattern |
| MEM-3.3 | **Reference external memory store** + gate (fixture package) | **Done** | P1 | `tests/fixtures` + unit test |
| MEM-4.1 | **Gate:** session SQLite persist + resume round-trip via H-APP host | **Done** | P1 | `tests/integration` |
| MEM-4.2 | **Gate:** LTM consolidation E2E with deterministic fake LLM | **Done** | P1 | `tests/acceptance` (not notebook-only) |
| MEM-4.3 | **Gate:** full memory stack on lab profile (task + session + LTM + org) | **Done** | P1 | acceptance or integration |
| MEM-5.1 | **Unit tests:** `engine_history_layer` - `SUMMARIZE_OLDEST` + truncate fallback | **Done** | P1 | `tests/unit/runtime/nexus/context` |
| MEM-5.2 | **Document context compression strategy matrix** (FULL / SUMMARY / SUMMARIZE_OLDEST / hard trim) | **Done** | P1 | Guide + canon §28.1 |
| MEM-CTX.1 | **`ContextDecisionProfile`** (or extend `ContextProfile`) - unified memory vs context vs RAG assembly policy for Tier-3 authors | **Done** | P1 | `environment_profile.py` |
| MEM-DOC.1 | **Author cookbook:** session vs checkpoint vs task KV mental model (LangGraph thread analogy) | **Done** | P1 | `guides/AGENT_CREATION_GUIDE.md` |
| MEM-DOC.2 | Document **`wire_task_memory_from_profile` vs Nexus LTM/org steps** gap | **Done** | P1 | Guide + this plan |
| MEM-DOC.3 | **Org memory scope** - profile + instructions vs shared episodic / team knowledge | **Done** | P1 | Guide |
| MEM-DOC.4 | **Recovery semantics** per memory layer (table in guide) | **Done** | P1 | Guide or `guides/HARNESS_ENVIRONMENT.md` |
| MEM-DOC.6 | Clarify **`custom_memory_kv` example** - integration KV vs Nexus TaskMemory | **Done** | P1 | `integrations/examples` README |
| MEM-GRAPH.1 | **Document:** Graph RAG (`intergrax/rag/graph`) ≠ agent entity graph memory | **Done** | P1 | Canon + RAG docs |
| MEM-TASK.1 | **Lab profile:** explicit task memory enable policy (replace silent default-off + log warning only) | **Done** | P1 | `lab_application` environment profile |
| MEM-TASK.2 | **Author cookbook:** MemoryView namespaces + delegation paths | **Done** | P1 | Guide Appendix G |

#### Wave MEM3 - P2: policy enforcement, hooks, observability, optional backends

| ID | Deliverable | Status | Priority | Location / acceptance |
|----|-------------|--------|----------|------------------------|
| MEM-6.1 | **Enforce `MemoryProfile.retention_days`** on session + task stores (TTL / purge job or read filter) | **Done** | P2 | session + task_memory |
| MEM-6.2 | **Enforce `scope_boundary`** on `PolicyScopedMemoryView` writes | **Done** | P2 | `memory_view.py` + policy |
| MEM-7.1 | **HookPoint `BEFORE_MEMORY_WRITE`** (+ optional `AFTER_MEMORY_WRITE`) | **Done** | P2 | `runtime/nexus/hooks` |
| MEM-7.2 | **Gate:** hook can deny or mutate memory write | **Done** | P2 | unit test |
| MEM-OBS.1 | **Memory SLO metrics** - LTM hit rate, retention violations, memory write volume | **Done** | P2 | observability / Prometheus hooks |
| MEM-DOC.5 | **Cookbook:** swap `UserProfileStore` to external backend via EP | **Done** | P2 | `guides/EXTENSION_AUTHOR_GUIDE.md` |
| MEM-CTX.2 | **Token-aware context trim** evaluation (vs char-cut only) - spike + recommendation | **Done** | P2 | `context_budget.py` RFC or impl |
| MEM-PERS.2 | **Optional:** Mongo `document_store` path for user memory artifacts | **Done** | P2 | Tier-0 integration wiring |
| MEM-ST.4 | **Optional:** `ConversationalMemoryStore` backend beyond in-memory | **Done** | P2 | `intergrax/memory` |

#### Wave MEM4 - P3: product memory layer (Band 3 option) + entity graph RFC

| ID | Deliverable | Status | Priority | Location / acceptance |
|----|-------------|--------|----------|------------------------|
| MEM-8.1 | **Design RFC:** unified memory product layer (Mem0-like auto-ingest, dedup, temporal validity) | **Done** | P3 | §6.3 decision gate |
| MEM-8.2 | **Background consolidation job** - auto fact extraction (optional product) | **Done** | P3 | Deferred with MEM-8.1 |
| MEM-9.1 | **Design RFC:** entity graph memory for user entities (separate from Graph RAG) | **Done** | P3 | Canon §53 follow-up |
| MEM-PERS.3 | **Spike:** Postgres memory backend for session/LTM (multi-tenant) | **Done** | P3 | RFC only; no default ship |

### MEM - Paydown log

| Date | ID | Notes |
|------|-----|-------|
| 2026-06-02 | MEM-1.*–MEM-9.* | Phase MEM **48/48 Done**; H-APP.4.3 **Done**; gate **571** |
| 2026-06-02 | §6.1 reference hosts | `with_harness_memory()` on legal/research; gate `test_reference_hosts_memory_bridge`; W-OPS memory_platform_gate |
| 2026-06-02 | MEM-0.1–MEM-0.2, MEM-PAR.1, MEM-CHk.1, MEM-PERS.1 | Memory audit → Phase MEM register in plan |
| 2026-06-02 | MEM-OBS.2 | Baseline already **Done** (DX-5.7) |

**Suggested PR order (P0 first):** MEM-1.1 → MEM-1.2 → MEM-1.3 → MEM-1.4 → MEM-1.5 → MEM-2.1 → MEM-2.2 → MEM-2.3 → MEM-1.6 → MEM-4.1 → MEM-5.1 → MEM-4.2 → MEM-3.1 → MEM-3.2 → MEM-0.3 → remaining MEM2 → MEM3 → MEM4.

**Success gate for Phase MEM closeout:** All **P0 + P1** rows **Done** or **Won't fix** with rationale; gate green; H-APP.4.3 **Done**; user LTM survives process restart on sqlite lab profile; `MemoryProfile` fully drives `RuntimeConfig` on all four reference hosts.

**Explicitly out of NOW:** K.1/K.2 memory features, Mem0 SaaS parity (MEM-8.2), entity graph implementation (MEM-9.1 beyond RFC), Redis session as default.

---

---

## Phase MEM-DEPTH - Memory Intelligence Depth

**Status:** **Done** (2026-06-08) - **26/26** deliverables; canonical architecture **Done** ([`architecture/MEMORY.md`](architecture/MEMORY.md)).
**Prerequisites:** Phase **MEM** (**Done**), Phase **CTX** (**Done**), Phase **R-Delegate** (**Done**), Phase **H-APP** (**Done**).
**Goal:** Raise Memory Layer from **L2 → L4** and Context Compiler from fragmented steps to a **unified, never-overflow** pipeline - context compiler, memory lifecycle automation, explore delegation, entity intelligence - **without** Band 3 business agents or Mem0 SaaS product.
**Priority ladder:** **Band 2am** (§4.0) - **closed** (2026-06-08); default queue = §6.1 maintenance.
**Execution order:** [§6.2ab](.#62ab-phase-mem-depth-execution-order-band-2am--active).
**Canon refs:** [`architecture/MEMORY.md`](architecture/MEMORY.md) · architecture §27–§28.1 · IDEAL §3.7, §16 · audit map §15–16.

**Delivery rule:** One `MEM-DEPTH-*` ID per PR → update status in tables below + paydown log → `pytest -m gate` + §6.1 audit scripts green.

**Audit verdict (target acceptance context):**

| Area | Maturity today | Target after MEM-DEPTH |
|------|----------------|------------------------|
| Task KV | 4/5 | 4/5 (maintain) |
| Context / never-overflow | 3/5 | 4.5/5 |
| STM persistence parity | 3/5 | 4/5 |
| User LTM lifecycle | 2.5/5 | 4/5 |
| Consolidation automation | 2/5 | 4/5 |
| Explore / discovery pattern | 1.5/5 | 4/5 |
| Entity graph agent memory | 1/5 | 3/5 (P2) or RFC+ship (P3 decision) |
| **Overall memory platform** | **~3.5/5** | **~4.5/5** |

**Out of scope (explicit):** K.1/K.2 business memory; hosted Mem0/Zep replacement; Redis as default session backend; autonomous prompt mutation without Prompt Registry.

```text
Wave MEMD0 - Canon doc + plan register + ADR (4 tasks)
Wave MEMD1 - Context Compiler + never-overflow (6 tasks) - P0
Wave MEMD2 - Persistence parity (2 tasks) - P0/P1
Wave MEMD3 - Memory lifecycle automation (5 tasks) - P1
Wave MEMD4 - Explore / discovery pattern (3 tasks) - P1
Wave MEMD5 - Entity intelligence + quality gates (6 tasks) - P2/P3
Total: 26
```

### MEM-DEPTH - Master deliverables register (all 26 tasks)

#### Wave MEMD0 - Canon & register

| ID | Deliverable | Status | Priority | Location / acceptance |
|----|-------------|--------|----------|------------------------|
| MEM-DEPTH-0.1 | **`architecture/MEMORY.md`** - canonical memory + context compiler spec | **Done** | **P0** | `docs/project/architecture/MEMORY.md` |
| MEM-DEPTH-0.2 | **Plan register** - Phase MEM-DEPTH, §4.0 Band 2am, §6.2ab, §6.1am; cross-links README + canon §27 | **Done** | **P0** | This section |
| MEM-DEPTH-0.3 | **ADR-MEM-001** - Context Compiler architecture decision (global budget allocator, degradation ladder) | **Done** | **P0** | `docs/project/technical/adr/entries/2026-06-08/ADR-MEM-001.md` |
| MEM-DEPTH-0.4 | **Sync** `AGENT_CREATION_GUIDE` Appendix G + audit map §15 pointers to MEMORY_ARCHITECTURE | **Done** | Low | Guide + AUDIT_MAP |

#### Wave MEMD1 - Context Compiler + never-overflow (P0)

| ID | Deliverable | Status | Priority | Location / acceptance |
|----|-------------|--------|----------|------------------------|
| MEM-DEPTH-1.1 | **`ContextCompiler`** - collect candidates from STM/LTM/RAG/task/profile; rank; allocate global token budget | **Done** | **P0 Critical** | `runtime/nexus/context/context_compiler.py` |
| MEM-DEPTH-1.2 | **`DegradationLadder`** - ordered steps per MEMORY_ARCHITECTURE §8.2; trace `degradation_step` on each apply | **Done** | **P0 Critical** | `degradation_ladder.py` + events |
| MEM-DEPTH-1.3 | **Tokenizer-aware trim** - replace char-cut happy path in `trim_message_to_budget` | **Done** | **P0** | `context_budget.py` |
| MEM-DEPTH-1.4 | **Wire `ContextDecisionProfile`** - enforce `include_session_history`, `prefer_*`, `max_memory_entries_in_context` in compiler | **Done** | **P0** | `CompileContextStep`, `context_compiler.py` |
| MEM-DEPTH-1.5 | **Pre-flight invariant** - `assembled_tokens + max_output ≤ context_window − margin` before every LLM call | **Done** | **P0** | `context_preflight.py + on_next_step`, `context_preflight.py` |
| MEM-DEPTH-1.6 | **Gate:** synthetic long session (10k turns fixture) completes without overflow; degradation trace present | **Done** | **P0** | `tests/acceptance/test_acceptance_context_compiler_long_session.py` |

#### Wave MEMD2 - Persistence parity (P0/P1)

| ID | Deliverable | Status | Priority | Location / acceptance |
|----|-------------|--------|----------|------------------------|
| MEM-DEPTH-2.1 | **MongoDB profile:** durable `SessionStorage` (not in-memory fallback) | **Done** | **P1** | `document_store_session_storage.py`, `memory_wiring.py` |
| MEM-DEPTH-2.2 | **Gate:** session persist + resume on Mongo document_store profile | **Done** | **P1** | `tests/unit/memory/test_mem_depth_modules.py` |

#### Wave MEMD3 - Memory lifecycle automation (P1)

| ID | Deliverable | Status | Priority | Location / acceptance |
|----|-------------|--------|----------|------------------------|
| MEM-DEPTH-3.1 | **Background consolidation job** - ship MEM-8.2 (scheduler hook + `SessionMemoryConsolidationService`) | **Done** | **P1** | `memory_consolidation_job.py` |
| MEM-DEPTH-3.2 | **LTM dedup + merge policy** on consolidate write (near-duplicate facts, preference updates) | **Done** | **P1** | `user_profile_dedup.py`, consolidation service |
| MEM-DEPTH-3.3 | **`MemoryProfile.consolidation_mode`** - `manual` \| `scheduled` \| `auto` → runtime config | **Done** | **P1** | `environment_profile.py`, `session_consolidation.py`, bridge |
| MEM-DEPTH-3.4 | **Episodic memory** - `MemoryKind.EPISODIC_EVENT` + retrieval for few-shot recall | **Done** | **P1** | `user_profile_memory.py`, consolidation service |
| MEM-DEPTH-3.5 | **Structured session summary schema** - facts / open tasks / decisions (not plain text blob) | **Done** | **P1** | `session_summary_schema.py` |

#### Wave MEMD4 - Explore / discovery pattern (P1)

| ID | Deliverable | Status | Priority | Location / acceptance |
|----|-------------|--------|----------|------------------------|
| MEM-DEPTH-4.1 | **`ExploreDelegationProfile`** on `DelegationSpec` - parallel search budget, synthesis-only return | **Done** | **P1** | `contracts/delegation.py` |
| MEM-DEPTH-4.2 | **Explore runner** - child context isolation + parallel retrieval + parent summary handoff | **Done** | **P1** | `runtime/nexus/delegation/explore_runner.py` |
| MEM-DEPTH-4.3 | **Hybrid retrieval orchestrator spike** - vector + keyword + graph doc in one ranked result set | **Done** | **P1** | `rag/retrieval/hybrid_retrieval_orchestrator.py` |

#### Wave MEMD5 - Entity intelligence + quality (P2/P3)

| ID | Deliverable | Status | Priority | Location / acceptance |
|----|-------------|--------|----------|------------------------|
| MEM-DEPTH-5.1 | **Entity graph user memory** - implement MEM-9 beyond RFC (separate from Graph RAG docs) | **Done** | **P2** | `intergrax/memory/entity_graph_memory.py` |
| MEM-DEPTH-5.2 | **Temporal validity** - `valid_from` / `valid_until` on `UserProfileMemoryEntry`; supersede old facts | **Done** | **P2** | `user_profile_memory.py`, dedup policy |
| MEM-DEPTH-5.3 | **Procedural memory versioning** - link `system_instructions` to Prompt Registry versions | **Done** | **P2** | `user_profile_instructions_service.py` |
| MEM-DEPTH-5.4 | **Context quality regression harness** - compression fidelity + retrieval@k benchmarks (IDEAL §16.5) | **Done** | **P2** | `tests/unit/runtime/architecture/test_context_regression_harness_mem_depth.py` |
| MEM-DEPTH-5.5 | **Workspace incremental index spike** - Merkle + AST chunking for codebase-scale hosts | **Done** | **P3** | `workspace_index_spike.py` |
| MEM-DEPTH-5.6 | **Postgres session/LTM backend** - ship MEM-PERS.3 beyond spike when multi-tenant required | **Done** | **P3** | `stores/postgres_memory_backend_rfc.py` (spike; §6.3 ship gate) |

### MEM-DEPTH - Paydown log

| Date | ID | Notes |
|------|-----|-------|
| 2026-06-08 | MEM-DEPTH-0.1, MEM-DEPTH-0.2, MEM-DEPTH-0.4 | Canonical `architecture/MEMORY.md` + plan register + cross-links |
| 2026-06-08 | MEM-DEPTH-0.3–5.6 | Phase MEM-DEPTH **26/26 Done**; Context Compiler + lifecycle + explore + entity graph |

**Suggested PR order (P0 first):** MEM-DEPTH-0.3 → MEM-DEPTH-1.1 → MEM-DEPTH-1.2 → MEM-DEPTH-1.3 → MEM-DEPTH-1.4 → MEM-DEPTH-1.5 → MEM-DEPTH-1.6 → MEM-DEPTH-2.1 → MEM-DEPTH-3.1 → MEM-DEPTH-3.2 → remaining MEMD3 → MEMD4 → MEMD5.

**Success gate for Phase MEM-DEPTH closeout:** All **P0 + P1** rows **Done**; Memory Layer audit **L3+**; never-overflow gate green; user LTM auto-consolidation optional on lab profile; `ContextDecisionProfile` enforced end-to-end.

**Explicitly out of NOW:** K.1/K.2, Mem0 SaaS, Redis session default, entity graph without §6.3 decision (MEM-DEPTH-5.1).

---

---

## Phase CTX - Context engineering control plane closeout

**Status:** **Done** (2026-06-02) - **4/4** deliverables Done (CTX-DOC.* + CTX-1–2); gate **612 passed**


**Priority ladder:** **Band 2n** (§4.0) - closed; default queue = **§6.1** maintenance.

**Execution order:** [§6.2bf](.#62bf-phase-ctx-execution-order-band-2n--closed) · queue: [§6.1f](.#61f-harness-implementation-queue--context-engineering-closeout-closed)

### CTX - Master register

| ID | Area | Deliverable | Status | Priority | Modules | Acceptance |
|----|------|-------------|--------|----------|---------|------------|
| CTX-DOC.1 | CTX0 | **Appendix L** - context engineering control plane (§L.1–L.6) | **Done** | High | `guides/AGENT_CREATION_GUIDE.md` | TOC + verification table |
| CTX-DOC.2 | CTX0 | **Cross-ref sync** - plan, README, AUDIT_MAP §16, audit prompt ref #9 | **Done** | Medium | `docs/*` | Links resolve |
| CTX-1 | CTX1 | **`context_runtime_bridge.py`** - dedicated context profile → `RuntimeConfig` | **Done** | **Critical** | `context_runtime_bridge.py`, `runtime_config_bridge.py` | `test_context_runtime_bridge.py` |
| CTX-2 | CTX2 | **`context_wiring.py`** - `ContextManager` + task options from environment; `nexus_factory` wire | **Done** | High | `context_wiring.py`, `nexus_factory.py`, `harness_host_runtime.py` | `test_context_wiring.py` |

### CTX - Paydown log

| Date | CTX ID | Summary |
|------|--------|---------|
| 2026-06-02 | CTX-DOC.1, CTX-DOC.2 | Appendix L + cross-refs; AUDIT_MAP §16 |
| 2026-06-02 | CTX-1, CTX-2 | Context runtime bridge + Nexus ContextManager wiring; gate **608** |

**Phase CTX complete when:** CTX-1–2 + CTX-DOC.* **Done**; §6.1f queue closed. **Status: complete (2026-06-02).**

---

---

### Phase I - Memory & Context (§27–28)

| # | Deliverable | Status | Canon | Notes |
|---|-------------|--------|-------|-------|
| I.1 | `TaskMemory` store | **Done** | §27 | Contract + coordinator; `store.py` (`open_task_memory_store`, env `INTERGRAX_TASK_MEMORY_DB` only) |
| I.2 | `MemoryView` gateway | **Done** | §42.35 | `PolicyScopedMemoryView` + UAEP wiring + `MEMORY_*` events |
| I.3 | `SharedTaskContext` | **Done** | §42.14 | Contract + `ContextManager` + graph merge + memory bridge |
| I.4 | Agent handoff | **Done** | §42.15 | `AgentHandoff` + `HandoffCoordinator` + graph path + `HANDOFF_*` events |
| I.5 | ContextManager v2 | **Done** | §28 | Provenance + summary tiers + `TaskContextAssemblyOptions` on `TaskExecutionOptions.context` |

---

---

## Phase MEMORY-LC - Full Harness Layer Completion closeout (2026-06-17)

**Status:** **Done** (2026-06-17) - re-validates 2026-06-17 layer completion + MEM-VEC/MEM-DEPTH; no open P0/P1
**Prerequisites:** MEM-VEC **Done** · MEM-DEPTH **Done** · MEM-OBS.1 **Done**
**Goal:** Formal Full Harness LC closeout - gate verification, journal
**ADR:** **No ADR needed**

| ID | Deliverable | Status | Priority | Acceptance |
|----|-------------|--------|----------|------------|
| MEMORY-LC-S1 | **Re-audit** - MEM register + tier-0/1 verdict | **Done** | High | No P0/P1 |
| MEMORY-LC-S2 | **Plan/architecture sync** - Full Harness LC note | **Done** | High | Domain pair consistent |
| MEMORY-LC-S3 | **Gate verification** | **Done** | High | 41 unit tests + integration · `check_entity_graph_memory_wiring` |
| MEMORY-LC-S4 | **Journal + progress tracker** | **Done** | High | `layer_completion_progress.json` mature |

**Deferred P2–P4:** procedural memory depth · org memory maturity · LangMem/Zep parity on entity graph

### 6.1av Harness implementation queue - Memory audit maintenance (planned)

**Source:** Layer 13 audit (2026-06-18) - `MEMORY` layer 15 · [`../audit_results/2026-06-18/MEMORY.md`](../audit_results/2026-06-18/MEMORY.md)
**Priority ladder:** **Band 1** (§6.1) - depth backlog only; **one ID per PR**

| Order | ID | Type | Priority | Status | Deliverable | Acceptance |
|-------|-----|------|----------|--------|-------------|------------|
| 1 | **MEM-MAINT-01** | Code | P3 | **Done** | Procedural memory depth - cognitive store mapping + tests beyond minimal `PROCEDURAL` kind | `cognitive_store_mapping.py` + procedural write/read tests |
| 2 | **MEM-MAINT-02** | Code | P3 | **Done** | Org memory maturity - promotion criteria + integration tests | `org_memory_maturity.py` + maturity checklist tests |
| 3 | **MEM-MAINT-03** | Backlog | P4 | **Done** | LangMem/Zep entity graph parity - backlog register + explicit scope boundary (not Mem0 SaaS) | Plan row; audit prompt boundary; no Phase K coupling |
| 4 | **MEM-MAINT-04** | Code | P3 | **Done** | MEM-DEPTH-5.2 - temporal validity on LTM facts | `memory_temporal.py` tests for valid-from/until retrieval window |

**Suggested PR order:** MEM-MAINT-04 → MEM-MAINT-01 → MEM-MAINT-02 → MEM-MAINT-03.

**Cross-domain:** Knowledge vs LTM boundary - CONTEXT_ENGINEERING; entity graph wiring **Done** (MEM-DEPTH-5.1).

**MEM-MAINT-03 scope boundary:** LangMem/Zep-style entity graph parity remains **backlog** - harness ships `EntityGraphMemoryStore` RFC spike; full Zep/LangMem SaaS parity is **out of scope** and **not** Phase K coupled.

---

*End of Memory Implementation Plan.*
