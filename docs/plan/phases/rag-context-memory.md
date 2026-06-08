# Implementation Phases — Rag Context Memory

**Hub:** [`INTERGRAX_IMPLEMENTATION_PLAN.md`](../INTERGRAX_IMPLEMENTATION_PLAN.md)

---

## Phase MEM — Memory Platform Completion

**Status:** **Done** (2026-06-02) — **48/48** deliverables; gate **571 passed**.  
**Prerequisites:** Phases **I** (TaskMemory), **R-Context**, **H-APP** (profile models), **DX-5.7** (ops:memory hints) **Done**; **H-APP.4.3** closed via **MEM-1.***.  
**Goal:** Close every gap from the **memory platform audit** — short-term session, user/org LTM, task KV, context compression, H-APP→runtime wiring, persistence, recovery, observability, developer hooks, and market-parity documentation — **without** Band 3 product agents (K.1/K.2) or Mem0-like SaaS product layer (MEM-8 deferred P3).  
**Priority ladder:** **Band 2h** (§4.0) — **default implementation queue** after §6.1 maintenance.  
**Execution order:** [§6.2aa](#62aa-phase-mem-execution-order-band-2h--active).  
**Canon refs:** §27 Memory model · §28.1 Context assembly · §42.35 MemoryView · Appendix G in [`guides/AGENT_CREATION_GUIDE.md`](guides/AGENT_CREATION_GUIDE.md).

**Delivery rule:** One `MEM-*` ID per PR → update status in tables below + paydown log → `pytest -m gate` + §6.1 audit scripts green.

**Audit verdict (baseline — preserve as acceptance context):**

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

**Overall platform memory score: ~3,5/5** — Tier-1 architecture closed for harness; product Mem0/Zep layer remains **§6.3** optional.

**Out of scope (explicit):** K.1/K.2 business memory; hosted Mem0/Zep replacement SaaS; Neo4j entity graph as default user memory (MEM-9 = design RFC only); Redis/Postgres session backends as shipped defaults (MEM-PERS.3 spike P3).

```text
Wave MEM0 — Register, audit baseline, conceptual + parity docs (9 tasks)
Wave MEM1 — P0 bridge + SQLite user LTM (9 tasks) — closes H-APP.4.3 gap
Wave MEM2 — P1 plugins, gates, context docs, graph clarification (18 tasks)
Wave MEM3 — P2 retention, hooks, SLO metrics, optional backends (9 tasks)
Wave MEM4 — P3 product memory layer + entity graph RFC (4 tasks)
Total: 48
```

### MEM — Conceptual model (canon §27 vs runtime)

Canon §27 defines **5 memory types**:

1. Task Memory  
2. Agent Local Memory  
3. User / Organization Memory  
4. Long-Term Knowledge Memory  
5. Execution Trace Memory  

Runtime maps these to **four operational stores** (+ trace + RAG — not memory layers):

```text
Short-term:     SessionManager + SessionStorage; optional ConversationalMemory (FIFO)
Task-scoped:    TaskMemory SQLite KV → PolicyScopedMemoryView → SharedTaskContext handoff
User / Org LTM: UserProfileManager + entries; OrganizationProfile SQLite
Knowledge:      RAG vectorstore; Graph RAG (document graph — NOT agent entity memory)
Trace:          RunTraceWriter / RuntimeEvents (immutable audit, not agent-mutable memory)
```

**Gap (document, do not implement as separate modules in MEM0):** no first-class **episodic / semantic / procedural** taxonomy in code — only `MemoryKind` entry tags (`USER_FACT`, `PREFERENCE`, `SESSION_SUMMARY`, `ORG_FACT`, `POLICY`). IDEAL harness doc describes episodic/semantic as **vision only**.

### MEM — Persistence backend matrix (as-built)

| Layer | In-memory | SQLite | Postgres | Redis | Mongo |
|-------|-----------|--------|----------|-------|-------|
| Task KV | test | prod path (`INTERGRAX_TASK_MEMORY_DB`) | — | — | — |
| Session | lab SQLite via bridge | bundle path | — | — | — |
| User profile LTM | test | bundle (`SQLiteUserProfileStore`) | — | — | optional `DocumentStoreUserProfileStore` (MEM-PERS.2) |
| Org profile | test | bundle | — | — | — |
| Checkpoints (≠ memory) | — | yes | — | — | — |
| Trace / events | test | yes | — | — | — |

**SQLite integration bundle** (`create_sqlite_integration`) = lab hub (trace, events, task_memory, session, org profile, checkpoints) — coherent for dev; **not** multi-tenant production scale.

**Recovery semantics (target documentation — MEM-DOC.4):**

| Layer | Recovery key | Works when | Broken today |
|-------|--------------|------------|--------------|
| Task memory | `tenant_id` + `task_id` + namespace | SQLite enabled | — |
| Session | `session_id` | SQLite SessionStorage when relational_store=sqlite | — |
| User LTM | user id | SQLite bundle or Mongo document_store | — |
| Long-running | checkpoint store | SQLite | separate from conversational memory |
| Org profile | org id | SQLite bundle | — |

### MEM — Market parity traceability (MEM-PAR.1)

| Capability | LangGraph | Mem0 / Zep | Intergrax today | Target ID |
|------------|-----------|------------|-----------------|-----------|
| Thread / session persistence | Checkpointer | Session + graph | Session + checkpoint **separate** | MEM-DOC.1, MEM-1.3 |
| Scoped KV per run | Store API | — | TaskMemory + MemoryView ✅ | MEM-DOC.2 |
| Auto fact extraction | — | core | Consolidation service, **manual trigger** | MEM-4.2, MEM-8.* |
| Entity graph memory | — | Zep ✅ | ❌ (RAG graph only) | MEM-GRAPH.1, MEM-9.* |
| Vector semantic memory | Optional | ✅ | User LTM via RAG index | MEM-2.* |
| Subagent namespace isolation | Subgraph state | — | delegation namespace ✅ | documented |
| Memory hooks / plugins | Checkpointer swap | API | Event bus only | MEM-3.*, MEM-7.* |
| Declarative env config | Partial | SaaS | MemoryProfile **Done** | MEM-1.* |
| Observability | LangSmith | Dashboard | Trace events ✅; no memory SLO | MEM-OBS.* |

### MEM — User audit checklist → deliverables (MEM-CHk.1)

| Audit question | Answer (as-built) | Deliverable IDs |
|----------------|-------------------|-----------------|
| How is memory **managed**? | Nexus Tier-1; agents via UAEP/MemoryView; profiles via runtime steps | MEM-DOC.2, MEM-1.* |
| **Limited context** handling? | Budget trim + history summarization + LTM limits + summary tiers — no single end-to-end policy | MEM-1.2, MEM-CTX.1, MEM-5.* |
| **Strategy** (summarize, trim)? | `HistoryLayer` + `ContextBudgetPolicy`; trim often char-cut | MEM-5.* |
| **Developer handlers**? | MemoryView, policy, events — no formal memory hooks / plugin catalog | MEM-3.*, MEM-7.* |
| **Where persisted**? | Task/org/session: SQLite (lab); user LTM: in-memory; Redis: not memory layer | MEM-2.*, MEM-PERS.1 |
| **Configuration**? | Env + `MemoryProfile` partial; H-APP bridge incomplete | MEM-1.* |
| **Recovery**? | Task/session by ID if SQLite; user LTM weak | MEM-1.3, MEM-2.*, MEM-DOC.4 |
| **Tests**? | Task/context OK; consolidation/history gaps | MEM-4.*, MEM-5.1, MEM-TEST.* |
| **Observation**? | MEMORY_* / CONTEXT_* events; no product memory metrics | MEM-OBS.* |
| **Graph memory**? | **No** as agent memory — Graph RAG only | MEM-GRAPH.1 |

### MEM — Architecture inventory (existing code — do not rewrite)

| Module | Tier | Role |
|--------|------|------|
| `intergrax/memory/` | 0 | ConversationalMemory, UserProfileManager |
| `intergrax/runtime/task_memory/` | 1 | Coordinator, MemoryView, SQLite store, delegation |
| `intergrax/runtime/nexus/session/` | 1 | SessionManager, InMemory/SQLite storage |
| `intergrax/runtime/nexus/context/` | 1 | ContextManager, context_budget, engine_history_layer |
| `intergrax/runtime/user_profile/session_memory_consolidation_service.py` | 1 | LLM session → LTM extraction |
| `intergrax/applications/_shared/runtime_config_bridge.py` | 3 | **Gap:** line ~112 always `InMemorySessionStorage()` |
| `intergrax/applications/_shared/task_memory_wiring.py` | 3 | Enables task DB from profile flags; does not enable LTM/org Nexus steps |
| `intergrax/applications/contracts/environment_profile.py` | 3 | `MemoryProfile`, `ContextProfile` |
| `intergrax/rag/graph/` | 0 | Graph RAG retrieval — **not** agent memory |
| `integrations/examples/custom_memory_kv/` | 0 | KV plugin example — not wired to TaskMemory |

**Existing gate tests:** `tests/unit/runtime/task_memory/`, `tests/acceptance/.../test_acceptance_08_memory_handoff`, context budget, profile steps, sqlite session integration.

**Known gaps in tests:** `engine_history_layer` summarization; E2E LTM consolidation; `MemoryProfile` → runtime wiring; external memory backends; graph-as-memory.

### MEM — Traceability (audit section → task IDs)

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

### MEM — Master deliverables register (all 48 tasks)

#### Wave MEM0 — Register & audit baseline

| ID | Deliverable | Status | Priority | Location / acceptance |
|----|-------------|--------|----------|------------------------|
| MEM-0.1 | **Phase MEM register** in this plan + §6.2aa + §6.1aa + doc model row | **Done** | Low | This section |
| MEM-0.2 | **Audit maturity baseline table** preserved (§Audit verdict above) | **Done** | Low | Do not delete on paydown |
| MEM-0.3 | **Canon §27 → 4 stores** mapping + flow diagram in `AGENT_CREATION_GUIDE` Appendix G | **Done** | Medium | Guide + cross-link §27 |
| MEM-0.4 | Document **`MemoryKind` tags** vs episodic/semantic/procedural (IDEAL vision vs runtime) | **Done** | Low | Guide or canon footnote |
| MEM-PAR.1 | **Market parity traceability table** (LangGraph / Mem0 / Zep) | **Done** | Low | This section §MEM — Market parity |
| MEM-CHk.1 | **User audit checklist** → deliverable mapping (10 questions) | **Done** | Low | This section §MEM — User audit checklist |
| MEM-PERS.1 | **Persistence backend matrix** synced to Appendix G | **Done** | Low | Guide Appendix G + this section |
| MEM-ST.1 | **Document:** Redis `KeyValueCache` = integration cache only — **not** session/LTM memory layer | **Done** | Low | `architecture/INTEGRATIONS.md` or guide |
| MEM-OBS.2 | Baseline: `MEMORY_READ`/`MEMORY_WRITE`, `CONTEXT_*`, `ops:memory` filter | **Done** | — | DX-5.7 · `phase_coverage.py` |

#### Wave MEM1 — P0: H-APP bridge + durable user LTM (closes H-APP.4.3)

| ID | Deliverable | Status | Priority | Location / acceptance |
|----|-------------|--------|----------|------------------------|
| MEM-1.1 | **`materialize_runtime_config` reads `MemoryProfile`** — map `enable_user_longterm_memory`, `enable_task_memory`, retention, scope flags to `RuntimeConfig` | **Done** | **P0 Critical** | `runtime_config_bridge.py` |
| MEM-1.2 | Map **`ContextProfile.budget_policy`** → `RuntimeConfig` context budget fields | **Done** | **P0 Critical** | `runtime_config_bridge.py` |
| MEM-1.3 | **`SessionManager` from integration bundle** — resolve `SQLiteSessionStorage` when sqlite profile active; remove hardcoded `InMemorySessionStorage()` in `build_runtime_context_from_environment` | **Done** | **P0 Critical** | `runtime_config_bridge.py` |
| MEM-1.4 | Map **`MemoryProfile.enable_org_memory`** → `RuntimeConfig.enable_org_profile_memory` | **Done** | **P0** | `runtime_config_bridge.py` |
| MEM-1.5 | **Gate test:** `ApplicationEnvironmentProfile` memory + context → `RuntimeConfig` round-trip | **Done** | **P0** | `tests/unit/applications/test_memory_profile_runtime_bridge.py` |
| MEM-1.6 | **Reconcile H-APP.4.3** — mark **Done** only when MEM-1.1–MEM-1.4 **Done** | **Done** | **P0** | H-APP register row |
| MEM-2.1 | **`SQLiteUserProfileStore`** — mirror `SQLiteOrganizationProfileStore` pattern | **Done** | **P0 Critical** | `intergrax/memory/` or `runtime/user_profile/` |
| MEM-2.2 | **Wire `SQLiteUserProfileStore`** in sqlite integration bundle + lab/legal/research profiles | **Done** | **P0** | integration bundle wiring |
| MEM-2.3 | **Unit tests:** `UserProfileManager` CRUD + search with SQLite backend (fake RetrievalService) | **Done** | **P0** | `tests/unit/memory/` |

#### Wave MEM2 — P1: gates, plugins prep, context docs, graph clarification

| ID | Deliverable | Status | Priority | Location / acceptance |
|----|-------------|--------|----------|------------------------|
| MEM-3.1 | **`UserProfileStore` / `SessionStorage` plugin Protocol** (typed, no Tier-2 imports) | **Done** | P1 | `intergrax/memory/contracts/` |
| MEM-3.2 | **Entry point group `intergrax.memory_stores`** + `bootstrap_memory_stores()` | **Done** | P1 | Mirror P-Ext pattern |
| MEM-3.3 | **Reference external memory store** + gate (fixture package) | **Done** | P1 | `tests/fixtures/` + unit test |
| MEM-4.1 | **Gate:** session SQLite persist + resume round-trip via H-APP host | **Done** | P1 | `tests/integration/` |
| MEM-4.2 | **Gate:** LTM consolidation E2E with deterministic fake LLM | **Done** | P1 | `tests/acceptance/` (not notebook-only) |
| MEM-4.3 | **Gate:** full memory stack on lab profile (task + session + LTM + org) | **Done** | P1 | acceptance or integration |
| MEM-5.1 | **Unit tests:** `engine_history_layer` — `SUMMARIZE_OLDEST` + truncate fallback | **Done** | P1 | `tests/unit/runtime/nexus/context/` |
| MEM-5.2 | **Document context compression strategy matrix** (FULL / SUMMARY / SUMMARIZE_OLDEST / hard trim) | **Done** | P1 | Guide + canon §28.1 |
| MEM-CTX.1 | **`ContextDecisionProfile`** (or extend `ContextProfile`) — unified memory vs context vs RAG assembly policy for Tier-3 authors | **Done** | P1 | `environment_profile.py` |
| MEM-DOC.1 | **Author cookbook:** session vs checkpoint vs task KV mental model (LangGraph thread analogy) | **Done** | P1 | `guides/AGENT_CREATION_GUIDE.md` |
| MEM-DOC.2 | Document **`wire_task_memory_from_profile` vs Nexus LTM/org steps** gap | **Done** | P1 | Guide + this plan |
| MEM-DOC.3 | **Org memory scope** — profile + instructions vs shared episodic / team knowledge | **Done** | P1 | Guide |
| MEM-DOC.4 | **Recovery semantics** per memory layer (table in guide) | **Done** | P1 | Guide or `guides/HARNESS_ENVIRONMENT.md` |
| MEM-DOC.6 | Clarify **`custom_memory_kv` example** — integration KV vs Nexus TaskMemory | **Done** | P1 | `integrations/examples/` README |
| MEM-GRAPH.1 | **Document:** Graph RAG (`intergrax/rag/graph/`) ≠ agent entity graph memory | **Done** | P1 | Canon + RAG docs |
| MEM-TASK.1 | **Lab profile:** explicit task memory enable policy (replace silent default-off + log warning only) | **Done** | P1 | `lab_application` environment profile |
| MEM-TASK.2 | **Author cookbook:** MemoryView namespaces + delegation paths | **Done** | P1 | Guide Appendix G |

#### Wave MEM3 — P2: policy enforcement, hooks, observability, optional backends

| ID | Deliverable | Status | Priority | Location / acceptance |
|----|-------------|--------|----------|------------------------|
| MEM-6.1 | **Enforce `MemoryProfile.retention_days`** on session + task stores (TTL / purge job or read filter) | **Done** | P2 | session + task_memory |
| MEM-6.2 | **Enforce `scope_boundary`** on `PolicyScopedMemoryView` writes | **Done** | P2 | `memory_view.py` + policy |
| MEM-7.1 | **HookPoint `BEFORE_MEMORY_WRITE`** (+ optional `AFTER_MEMORY_WRITE`) | **Done** | P2 | `runtime/nexus/hooks/` |
| MEM-7.2 | **Gate:** hook can deny or mutate memory write | **Done** | P2 | unit test |
| MEM-OBS.1 | **Memory SLO metrics** — LTM hit rate, retention violations, memory write volume | **Done** | P2 | observability / Prometheus hooks |
| MEM-DOC.5 | **Cookbook:** swap `UserProfileStore` to external backend via EP | **Done** | P2 | `guides/EXTENSION_AUTHOR_GUIDE.md` |
| MEM-CTX.2 | **Token-aware context trim** evaluation (vs char-cut only) — spike + recommendation | **Done** | P2 | `context_budget.py` RFC or impl |
| MEM-PERS.2 | **Optional:** Mongo `document_store` path for user memory artifacts | **Done** | P2 | Tier-0 integration wiring |
| MEM-ST.4 | **Optional:** `ConversationalMemoryStore` backend beyond in-memory | **Done** | P2 | `intergrax/memory/` |

#### Wave MEM4 — P3: product memory layer (Band 3 option) + entity graph RFC

| ID | Deliverable | Status | Priority | Location / acceptance |
|----|-------------|--------|----------|------------------------|
| MEM-8.1 | **Design RFC:** unified memory product layer (Mem0-like auto-ingest, dedup, temporal validity) | **Done** | P3 | §6.3 decision gate |
| MEM-8.2 | **Background consolidation job** — auto fact extraction (optional product) | **Done** | P3 | Deferred with MEM-8.1 |
| MEM-9.1 | **Design RFC:** entity graph memory for user entities (separate from Graph RAG) | **Done** | P3 | Canon §53 follow-up |
| MEM-PERS.3 | **Spike:** Postgres memory backend for session/LTM (multi-tenant) | **Done** | P3 | RFC only; no default ship |

### MEM — Paydown log

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

## Phase MEM-DEPTH — Memory Intelligence Depth

**Status:** **Planned** (0/26) — canonical architecture **Done** ([`architecture/MEMORY.md`](architecture/MEMORY.md), 2026-06-08).  
**Prerequisites:** Phase **MEM** (**Done**), Phase **CTX** (**Done**), Phase **R-Delegate** (**Done**), Phase **H-APP** (**Done**).  
**Goal:** Raise Memory Layer from **L2 → L4** and Context Compiler from fragmented steps to a **unified, never-overflow** pipeline — context compiler, memory lifecycle automation, explore delegation, entity intelligence — **without** Band 3 business agents or Mem0 SaaS product.  
**Priority ladder:** **Band 2am** (§4.0) — **recommended next harness band** after §6.1 gate (parallel-safe slices).  
**Execution order:** [§6.2ab](#62ab-phase-mem-depth-execution-order-band-2am--active).  
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
Wave MEMD0 — Canon doc + plan register + ADR (4 tasks)
Wave MEMD1 — Context Compiler + never-overflow (6 tasks) — P0
Wave MEMD2 — Persistence parity (2 tasks) — P0/P1
Wave MEMD3 — Memory lifecycle automation (5 tasks) — P1
Wave MEMD4 — Explore / discovery pattern (3 tasks) — P1
Wave MEMD5 — Entity intelligence + quality gates (6 tasks) — P2/P3
Total: 26
```

### MEM-DEPTH — Master deliverables register (all 26 tasks)

#### Wave MEMD0 — Canon & register

| ID | Deliverable | Status | Priority | Location / acceptance |
|----|-------------|--------|----------|------------------------|
| MEM-DEPTH-0.1 | **`architecture/MEMORY.md`** — canonical memory + context compiler spec | **Done** | **P0** | `docs/architecture/MEMORY.md` |
| MEM-DEPTH-0.2 | **Plan register** — Phase MEM-DEPTH, §4.0 Band 2am, §6.2ab, §6.1am; cross-links README + canon §27 | **Done** | **P0** | This section |
| MEM-DEPTH-0.3 | **ADR-MEM-001** — Context Compiler architecture decision (global budget allocator, degradation ladder) | **Planned** | **P0** | `docs/adr/ADR-MEM-001.md` |
| MEM-DEPTH-0.4 | **Sync** `AGENT_CREATION_GUIDE` Appendix G + audit map §15 pointers to MEMORY_ARCHITECTURE | **Done** | Low | Guide + AUDIT_MAP |

#### Wave MEMD1 — Context Compiler + never-overflow (P0)

| ID | Deliverable | Status | Priority | Location / acceptance |
|----|-------------|--------|----------|------------------------|
| MEM-DEPTH-1.1 | **`ContextCompiler`** — collect candidates from STM/LTM/RAG/task/profile; rank; allocate global token budget | **Planned** | **P0 Critical** | `runtime/nexus/context/context_compiler.py` |
| MEM-DEPTH-1.2 | **`DegradationLadder`** — ordered steps per MEMORY_ARCHITECTURE §8.2; trace `degradation_step` on each apply | **Planned** | **P0 Critical** | `context_compiler.py` + events |
| MEM-DEPTH-1.3 | **Tokenizer-aware trim** — replace char-cut happy path in `trim_message_to_budget` | **Planned** | **P0** | `context_budget.py` |
| MEM-DEPTH-1.4 | **Wire `ContextDecisionProfile`** — enforce `include_session_history`, `prefer_*`, `max_memory_entries_in_context` in compiler | **Planned** | **P0** | `memory_runtime_bridge.py`, runtime steps |
| MEM-DEPTH-1.5 | **Pre-flight invariant** — `assembled_tokens + max_output ≤ context_window − margin` before every LLM call | **Planned** | **P0** | AgentEngine / LLM step |
| MEM-DEPTH-1.6 | **Gate:** synthetic long session (10k turns fixture) completes without overflow; degradation trace present | **Planned** | **P0** | `tests/acceptance/` or integration |

#### Wave MEMD2 — Persistence parity (P0/P1)

| ID | Deliverable | Status | Priority | Location / acceptance |
|----|-------------|--------|----------|------------------------|
| MEM-DEPTH-2.1 | **MongoDB profile:** durable `SessionStorage` (not in-memory fallback) | **Planned** | **P1** | `memory_wiring.py` + integration bundle |
| MEM-DEPTH-2.2 | **Gate:** session persist + resume on Mongo document_store profile | **Planned** | **P1** | `tests/integration/` |

#### Wave MEMD3 — Memory lifecycle automation (P1)

| ID | Deliverable | Status | Priority | Location / acceptance |
|----|-------------|--------|----------|------------------------|
| MEM-DEPTH-3.1 | **Background consolidation job** — ship MEM-8.2 (scheduler hook + `SessionMemoryConsolidationService`) | **Planned** | **P1** | `runtime/user_profile/` or Tier-3 wiring |
| MEM-DEPTH-3.2 | **LTM dedup + merge policy** on consolidate write (near-duplicate facts, preference updates) | **Planned** | **P1** | `session_memory_consolidation_service.py` |
| MEM-DEPTH-3.3 | **`MemoryProfile.consolidation_mode`** — `manual` \| `scheduled` \| `auto` → runtime config | **Planned** | **P1** | `environment_profile.py`, bridge |
| MEM-DEPTH-3.4 | **Episodic memory** — `MemoryKind.EPISODIC_EVENT` + retrieval for few-shot recall | **Planned** | **P1** | `user_profile_memory.py`, manager |
| MEM-DEPTH-3.5 | **Structured session summary schema** — facts / open tasks / decisions (not plain text blob) | **Planned** | **P1** | consolidation service + prompt |

#### Wave MEMD4 — Explore / discovery pattern (P1)

| ID | Deliverable | Status | Priority | Location / acceptance |
|----|-------------|--------|----------|------------------------|
| MEM-DEPTH-4.1 | **`ExploreDelegationProfile`** on `DelegationSpec` — parallel search budget, synthesis-only return | **Planned** | **P1** | `contracts/delegation.py` |
| MEM-DEPTH-4.2 | **Explore runner** — child context isolation + parallel retrieval + parent summary handoff | **Planned** | **P1** | `runtime/nexus/delegation/` or graph runner |
| MEM-DEPTH-4.3 | **Hybrid retrieval orchestrator spike** — vector + keyword + graph doc in one ranked result set | **Planned** | **P1** | `rag/retrieval/` RFC or impl |

#### Wave MEMD5 — Entity intelligence + quality (P2/P3)

| ID | Deliverable | Status | Priority | Location / acceptance |
|----|-------------|--------|----------|------------------------|
| MEM-DEPTH-5.1 | **Entity graph user memory** — implement MEM-9 beyond RFC (separate from Graph RAG docs) | **Planned** | **P2** | `intergrax/memory/` or new module · §6.3 gate for scope |
| MEM-DEPTH-5.2 | **Temporal validity** — `valid_from` / `valid_until` on `UserProfileMemoryEntry`; supersede old facts | **Planned** | **P2** | `user_profile_memory.py`, stores |
| MEM-DEPTH-5.3 | **Procedural memory versioning** — link `system_instructions` to Prompt Registry versions | **Planned** | **P2** | `user_profile_instructions_service.py` |
| MEM-DEPTH-5.4 | **Context quality regression harness** — compression fidelity + retrieval@k benchmarks (IDEAL §16.5) | **Planned** | **P2** | `tests/` + fixture corpus |
| MEM-DEPTH-5.5 | **Workspace incremental index spike** — Merkle + AST chunking for codebase-scale hosts | **Planned** | **P3** | RFC + spike; optional Tier-3 |
| MEM-DEPTH-5.6 | **Postgres session/LTM backend** — ship MEM-PERS.3 beyond spike when multi-tenant required | **Planned** | **P3** | integration bundle · §6.3 gate |

### MEM-DEPTH — Paydown log

| Date | ID | Notes |
|------|-----|-------|
| 2026-06-08 | MEM-DEPTH-0.1, MEM-DEPTH-0.2, MEM-DEPTH-0.4 | Canonical `architecture/MEMORY.md` + plan register + cross-links |

**Suggested PR order (P0 first):** MEM-DEPTH-0.3 → MEM-DEPTH-1.1 → MEM-DEPTH-1.2 → MEM-DEPTH-1.3 → MEM-DEPTH-1.4 → MEM-DEPTH-1.5 → MEM-DEPTH-1.6 → MEM-DEPTH-2.1 → MEM-DEPTH-3.1 → MEM-DEPTH-3.2 → remaining MEMD3 → MEMD4 → MEMD5.

**Success gate for Phase MEM-DEPTH closeout:** All **P0 + P1** rows **Done**; Memory Layer audit **L3+**; never-overflow gate green; user LTM auto-consolidation optional on lab profile; `ContextDecisionProfile` enforced end-to-end.

**Explicitly out of NOW:** K.1/K.2, Mem0 SaaS, Redis session default, entity graph without §6.3 decision (MEM-DEPTH-5.1).

---

### Phase P-Ext — Plugin Catalogs (Integrations, Tools, Skills)

**Status:** **Done** (2026-06-02) — MVP + production closure (Appendix I).  
**Prerequisites:** Phases **M** (Integration Library), **O** (Tool Library), **R** (Skill Library MVP) **Done**; open integration slug model (no closed `IntegrationSlug` enum in registry) **Done**.  
**Goal:** Make all three Tier-0 catalogs **plugin-native** and aligned with market patterns (hexagonal adapters, MCP-style tools, capability packs) — including **pip-installable** extensions without editing Intergrax core.  
**Tracker:** **Appendix I** (task-level status). **Author guide:** [`guides/EXTENSION_AUTHOR_GUIDE.md`](guides/EXTENSION_AUTHOR_GUIDE.md).

**Delivered (2026-06-02):** `load_plugins` + `bootstrap_catalogs()` · three plugin protocols · lazy presets/bundle ids · EP fixture package · `warn_override` conflict policy · scaffold CLI · integrations **manifest+factory** (**135** full) + `IntegrationPlugin` for externals · tools **13/13** `ToolPlugin` · skills **3/3** `SkillPlugin` · `resolve_typed` (6 categories) · health API · `CatalogSnapshot` · expanded `check_plugin_catalog.py` · canon §7.1.5.1 + author guide.

**Principle:** Integration → Tool → Skill → Agent (unchanged) · explicit first-party bootstrap + optional entry points · one P-Ext.* ID per PR · gate green.

**Production-path reality (do not confuse with MVP):**

| Layer | Shipped catalog | External extension | Runtime materialization |
|-------|-----------------|--------------------|-------------------------|
| **Integrations** | **135** slugs (`preset="full"`) / **12** core (`preset="core"`) via `register_from_manifest` + `create_*` — **0** shipped `register.py` use `register_integration_plugin` | `IntegrationPlugin` + EP `intergrax.integrations` | `IntegrationProfile.resolve(category, config=…)` → backend instance |
| **Tools** | **13** bundles / **~29** `tool_id` — **13/13** via `ToolPlugin` (`shipped_plugins.py`) | `ToolPlugin` + EP `intergrax.tools` | `bootstrap_catalogs` → `build_registry_from_profile(ToolProfile, ctx)` → `ToolRegistry` → `RuntimeToolInvoker` / MCP |
| **Skills** | **3** bundles / **8** `skill_id` — **3/3** via `SkillPlugin` (`harness`×6, `legal`×1, `research`×1) | `SkillPlugin` + EP `intergrax.skills` | `build_registry_from_profile(SkillProfile)` → `SkillRegistry` → `SkillResolver` → `allowed_tools` |

**Out of scope for Phase P-Ext:**

- Online plugin marketplace UI / central registry service
- Runtime hot-reload of catalogs without process restart
- Skill as executable workflow graph (LangGraph pack) — separate initiative
- Replacing `ToolWiringContext` with a generic DI framework
- Migrating all **135** shipped integrations to `IntegrationPlugin` classes (optional long-term; manifest path remains supported)

#### P-Ext.0 — Shared plugin foundation

**Goal:** One plugin loader and one Tier-3 bootstrap entry point.

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| P-Ext.0.1 | **`load_plugins(group, …)`** — entry point discovery | **Done** | **Critical** | `intergrax/core/plugins/discovery.py` | Idempotent; `on_conflict=error\|skip` |
| P-Ext.0.2 | **Plugin errors** — `PluginConflictError`, `PluginLoadError` | **Done** | High | `intergrax/core/plugins/errors.py` | Unit tests |
| P-Ext.0.3 | **`bootstrap_catalogs()`** — unified Tier-3 composition | **Done** | **Critical** | `intergrax/core/catalog_bootstrap.py` | tool/skill wiring + idempotent shipped |
| P-Ext.0.4 | **`docs/guides/EXTENSION_AUTHOR_GUIDE.md`** | **Done** | High | `docs/` | pip package walkthrough |
| P-Ext.0.5 | **Fixture pip package** in tests | **Done** | High | `tests/fixtures/plugin_packages/` | editable install; registers integration + tool + skill |
| P-Ext.0.6 | **EP discovery tests** via fixture (all three groups) | **Done** | High | `tests/unit/core/plugins/` | `bootstrap_catalogs(discover_entry_points=True)` loads fixture |
| P-Ext.0.7 | **`INTERGRAX_DISCOVER_PLUGINS`** env + Tier-3 wiring | **Done** | Medium | `catalog_bootstrap.py`, `applications/_shared/platform_wiring.py` | lab opt-in; default `false` in prod hosts |

**DoD:** Fixture package registers via entry point; discovery unit tests green.

**Entry point groups (canonical names):**

```toml
[project.entry-points."intergrax.integrations"]
[project.entry-points."intergrax.tools"]
[project.entry-points."intergrax.skills"]
```

---

#### P-Ext.1 — Integrations: plugin closure

**Baseline:** `IntegrationManifest`, `IntegrationPlugin`, `register_from_manifest`, per-provider `manifest.py` (open slug catalog).

**Audit snapshot (2026-06-02 — integrations only; counts synced post M.6 P5 closeout):**

| Area | Finding | Prod? |
|------|---------|-------|
| **Shipped catalog** | `bootstrap_core` **12** slugs + `bootstrap_extended` **~123** → **135** full; all `register.py` call `register_from_manifest(MANIFEST, create_*)` | **Yes** — primary harness path |
| **`IntegrationPlugin` shipped** | **0/135** providers register via `register_integration_plugin` in shipped code | N/A — external / explicit only |
| **Reference plugin class** | `SqliteIntegrationPlugin` in `sqlite/plugin.py`; `register.py` still uses manifest path | Doc pattern only (P-Ext.1.12) |
| **External example** | `integrations/examples/custom_memory_kv/` + `test_external_plugin.py` (explicit register) | **Yes** API; EP not tested |
| **`IntegrationProfile.resolve`** | Manifest, plugin class, slug `str`, or pre-built instance via `IntegrationBinding` | **Yes** — Tier-3 prod |
| **`resolve_typed.py`** | Six typed helpers incl. vector_store, notification_channel, object_storage | **Done** |
| **`IntegrationSlug` enum** | **0** references in `intergrax/**/*.py` and provider `USAGE.md`; legacy mention only in plan + migration scripts | **Done** (P-Ext.1.5) |
| **Tier-3 bootstrap** | `integration_wiring` / `tool_wiring` / `skill_wiring` → `bootstrap_catalogs()` + lazy bundle ids | **Done** |
| **Entry points** | Fixture pip package + EP tests; `INTERGRAX_DISCOVER_PLUGINS` for lab | **Done** |
| **`on_conflict`** | `bootstrap_catalogs(on_conflict=…)` — `error`, `skip`, `override`, `warn_override` for catalog slugs + EP names | **Done** (P-Ext.4.3) |
| **Health API** | `integrations/registry/health.py` — `ping_integration` / `integration_registered` | **Done** |
| **Unit tests** | Per-provider tests + `test_profile` + `test_external_plugin` + lazy `preset="core"` in `test_lazy_catalog_bootstrap` | **Strong**; no full-count assertion in CI |

**Verdict:** Shipped integrations are **production-ready** on the **manifest + factory** path. `IntegrationPlugin` is **production-ready for third-party** extensions; parity with tools (all shipped as plugin classes) is **explicitly out of scope**.

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| P-Ext.1.1 | Wire **`intergrax.integrations`** entry points in `bootstrap_catalogs()` | **Done** | **Critical** | `catalog_bootstrap.py` | `discover_entry_points=True` |
| P-Ext.1.2 | Split **`register_default_integrations()`** → core + optional | **Done** | High | `integrations/registry/bootstrap_core.py` | `preset="core"` (12) \| `"full"` (135) |
| P-Ext.1.3 | **Typed resolve** helpers (top categories) | **Done** | Medium | `integrations/registry/resolve_typed.py` | 3 categories today |
| P-Ext.1.3a | Expand **`resolve_typed`** + unit tests | **Done** | Medium | `resolve_typed.py`, `tests/unit/integrations/test_resolve_typed.py` | +`vector_store`, `notification_channel`, `object_storage`; used in lab docs |
| P-Ext.1.4 | **Health check** API per slug (optional) | **Done** | Low | `integrations/registry/health.py` | `ping(slug) -> bool` smoke helper |
| P-Ext.1.5 | Remove **`IntegrationSlug`** from docs/scripts | **Done** | Medium | `**/USAGE.md`, `README.md`, `scripts/`, `docs/guides/AGENT_CREATION_GUIDE.md` | `intergrax/**/*.py` already clean |
| P-Ext.1.6 | **EP integration test** via fixture | **Done** | High | `tests/unit/integrations/` | `discover_entry_points=True` loads fixture slug |
| P-Ext.1.7 | **Dual-model docs** — manifest+factory vs `IntegrationPlugin` | **Done** | Medium | `architecture/INTEGRATIONS.md`, `guides/EXTENSION_AUTHOR_GUIDE.md` | decision table + when to migrate |
| P-Ext.1.8 | **CI smoke** — integration slug counts | **Done** | Medium | `scripts/check_plugin_catalog.py` | `core` ≥12, `full` ≥95 (or exact snapshot) |
| P-Ext.1.9 | **`test_resolve_typed.py`** | **Done** | Low | `tests/unit/integrations/` | type errors on wrong contract |
| P-Ext.1.10 | **Tier-3** lab/poc use `bootstrap_catalogs(integration_preset=…)` | **Done** | High | `applications/*/host/integration_wiring.py` | replace bare `register_default_integrations()` |
| P-Ext.1.11 | **`applications/_shared/integration_wiring.py`** helper | **Done** | Medium | `applications/_shared/` | mirror `tool_wiring` — bootstrap + profile factory |
| P-Ext.1.12 | **`SqliteIntegrationPlugin`** — document or wire one shipped slug | **Done** | Low | `sqlite/register.py` or `architecture/INTEGRATIONS.md` | either `register_integration_plugin` in sqlite **or** “reference only” in docs |

**DoD:** 364+ integration unit tests green; external integration via entry point **and** via pip entry point (fixture); Tier-3 hosts use unified `bootstrap_catalogs()` for integrations.

---

#### P-Ext.2 — Tools: ToolPlugin + MCP export

**Baseline:** `ToolContract`, `ToolBundleEntry`, `ToolProfile`, `ToolWiringContext`, `RuntimeToolInvoker`.

**Audit snapshot (2026-06-02 — tools only):**

| Area | Finding | Prod? |
|------|---------|-------|
| **Shipped catalog** | **13/13** bundles on `ToolPlugin` via `shipped_plugins.py` + `define_tool_plugin` | **Yes** — full plugin parity |
| **Tool count** | **~29** `tool_id` across bundles (RAG, websearch, jira, sandbox, vision, speech, …) | **Yes** |
| **Legacy register path** | No shipped bundle bypasses `register_tool_plugin`; `register_from_tool_manifest` is internal only | **Yes** |
| **External example** | `intergrax/tools/examples/` + `test_external_tool_plugin.py` | **Yes** |
| **EP `intergrax.tools`** | Fixture package + EP discovery tests (P-Ext.0.5 / 2.11) | **Yes** |
| **Tier-3 wiring** | `tool_wiring.build_application_tool_wiring` → `bootstrap_catalogs(register_shipped=True)` | **Yes** |
| **Lazy catalog** | `tool_wiring` passes `tool_bundle_ids` from `ToolProfile` | **Done** |
| **Runtime materialization** | Two-phase: catalog → `ToolWiringContext` + integrations → `ToolRegistry` handlers | **Yes** |
| **MCP / standalone LLM** | `export_mcp_tools`, `ToolsAgent`, `RuntimeToolInvoker` trace | **Yes** — strongest market path |
| **Unit tests** | Per-bundle tests + `test_external_tool_plugin` + EP fixture | **Yes** |

**Verdict:** Shipped tools are **production-ready** on **`ToolPlugin`**; P-Ext.2 closure complete (external example, EP test, lazy `tool_wiring`).

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| P-Ext.2.1 | **`ToolPlugin` Protocol** | **Done** | **Critical** | `intergrax/tools/core/plugin.py` | `tool_bundle_manifest()`, `register_tools(registry, ctx)` |
| P-Ext.2.2 | **`ToolManifest`** (bundle metadata) | **Done** | **Critical** | `intergrax/tools/core/manifest.py` | bundle_id, tool_ids, status |
| P-Ext.2.3 | **`register_tool_plugin()`** | **Done** | **Critical** | `intergrax/tools/registry/plugin_register.py` | Mirror integrations |
| P-Ext.2.4 | **Pilot migration** — RAG bundle → `ToolPlugin` | **Done** | High | `tools/providers/rag/` | Pattern for other bundles |
| P-Ext.2.5 | Entry point group **`intergrax.tools`** | **Done** | High | `catalog_bootstrap.py` | opt-in `discover_entry_points` |
| P-Ext.2.6 | **`export_mcp_tools(registry)`** | **Done** | High | `intergrax/tools/exporters/mcp.py` | alias of `to_mcp_tools` |
| P-Ext.2.7 | **`ToolContract.version`** field (semver) | **Done** | Medium | `tools/core/contracts.py` | Default `1.0.0` |
| P-Ext.2.8 | **Migrate all shipped tool bundles** → `ToolPlugin` | **Done** | High | `tools/registry/shipped_plugins.py`, `providers/*/register.py` | 13/13 bundles |
| P-Ext.2.9 | **Reference external tool** — `tools/examples/` | **Done** | High | `intergrax/tools/examples/` | mirror `integrations/examples/custom_memory_kv` |
| P-Ext.2.10 | **`test_external_tool_plugin.py`** | **Done** | High | `tests/unit/tools/` | catalog → `build_registry_from_profile` → `RuntimeToolInvoker.invoke` |
| P-Ext.2.11 | **EP tool test** via fixture | **Done** | High | `tests/unit/tools/` | depends on P-Ext.0.5 |
| P-Ext.2.12 | **`tool_wiring` lazy bootstrap** — pass `tool_bundle_ids` from profile | **Done** | Medium | `applications/_shared/tool_wiring.py` | `bootstrap_catalogs(..., tool_bundle_ids=profile.enabled_bundles)` |

**DoD:** External tool executes via `RuntimeToolInvoker` after entry-point registration (test proves it); Tier-3 `tool_wiring` supports lazy bundle bootstrap.

---

#### P-Ext.3 — Skills: SkillPlugin

**Baseline:** `SkillManifest`, `SkillBundleEntry`, `SkillResolver`, `AgentRegistry` merge to `allowed_tools`.

**Audit snapshot (2026-06-02 — skills only):**

| Area | Finding | Prod? |
|------|---------|-------|
| **Shipped catalog** | **3/3** bundles on `SkillPlugin` via `shipped_plugins.py` + `register_default_skills()` | **Yes** — best plugin parity of Tier-0 |
| **Skill count** | **8** `skill_id`: `harness` (6), `legal` (1), `research` (1) | **Yes** |
| **Legacy `register_skill_bundle`** | Only in `plugin_register.py` + **outdated** `scaffold new-skill` output | Scaffold **not** prod (P-Ext.3.10) |
| **`register_from_skill_manifest`** | Internal helper; all shipped paths use `register_skill_plugin` | **Yes** |
| **External example** | `intergrax/skills/examples/` + external plugin tests | **Yes** |
| **EP `intergrax.skills`** | Fixture package + EP discovery tests (P-Ext.0.5 / 3.8) | **Yes** |
| **Tier-3 wiring** | `skill_wiring.build_application_skill_wiring` → `bootstrap_catalogs(register_shipped=True)` — **better than integrations** | **Yes** |
| **Lazy catalog** | `skill_wiring` passes `skill_bundle_ids` from `SkillProfile` | **Done** |
| **Runtime materialization** | Two-phase like tools: catalog bundle rows → `build_registry_from_profile` → `SkillRegistry` | **Yes** |
| **`requires_skills`** | Resolver + `test_requires_skills.py`; **0** shipped manifests use it | Feature **Done**; adoption open (P-Ext.3.12) |
| **Cursor `SKILL.md` importer** | `CursorSkillImporter` — parallel path, not `SkillPlugin` | **Yes** for import; document vs plugin (P-Ext.3.11) |
| **Agent merge** | `AgentRegistry.register(..., skill_registry=, tool_registry=)` + `test_agent_registry_skills.py` | **Yes** |
| **Unit tests** | Harness + resolver + `test_external_skill_plugin` + EP fixture | **Yes** |

**Verdict:** Shipped skills are **production-ready** on **`SkillPlugin`**; P-Ext.3 closure complete (external example, EP test, lazy `skill_wiring`, scaffold alignment).

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| P-Ext.3.1 | **`SkillPlugin` Protocol** | **Done** | **Critical** | `intergrax/skills/core/plugin.py` | `skill_bundle_manifest()`, `skill_manifests()`, `register_skills(registry)` |
| P-Ext.3.2 | **`register_skill_plugin()`** | **Done** | **Critical** | `intergrax/skills/registry/plugin_register.py` | Wraps `register_from_skill_manifest` |
| P-Ext.3.3 | Entry point group **`intergrax.skills`** | **Done** | High | `catalog_bootstrap.py` | opt-in `discover_entry_points` |
| P-Ext.3.4 | Migrate **`harness`** + **`research`** + **`legal`** → `SkillPlugin` | **Done** | High | `skills/providers/*/plugin.py`, `shipped_plugins.py` | **3/3** bundles |
| P-Ext.3.5 | **`requires_skills`** on `SkillManifest` + resolver DFS | **Done** | Low | `skills/resolver.py`, `test_requires_skills.py` | Cycle + unknown dep errors |
| P-Ext.3.6 | **Reference external skill** — `skills/examples/` | **Done** | High | `intergrax/skills/examples/` | mirror `integrations/examples/custom_memory_kv` |
| P-Ext.3.7 | **`test_external_skill_plugin.py`** | **Done** | High | `tests/unit/skills/` | explicit `register_skill_plugin` → `SkillResolver` → tool merge |
| P-Ext.3.8 | **EP skill test** via fixture | **Done** | High | `tests/unit/skills/` | depends on P-Ext.0.5 |
| P-Ext.3.9 | **`skill_wiring` lazy bootstrap** — pass `skill_bundle_ids` from profile | **Done** | Medium | `applications/_shared/skill_wiring.py` | `bootstrap_catalogs(..., skill_bundle_ids=profile.enabled_bundles)` |
| P-Ext.3.10 | **Scaffold `new-skill`** emits `SkillPlugin` + `plugin.py` | **Done** | Medium | `intergrax/scaffold/new_skill.py` | remove legacy `register_skill_bundle` template |
| P-Ext.3.11 | **Docs: SkillPlugin vs Cursor importer** | **Done** | Medium | `architecture/SKILLS.md`, `guides/EXTENSION_AUTHOR_GUIDE.md` | when to use pip plugin vs `SKILL.md` import |
| P-Ext.3.12 | **`requires_skills` in shipped harness** (optional demo) | **Done** | Low | `skills/providers/harness/manifests.py` | one derived skill depending on `harness.tool_smoke` |

**DoD:** External skill merges `allowed_tools` on `AgentRegistry.register` (test proves it); Tier-3 `skill_wiring` supports lazy bundle bootstrap.

---

#### P-Ext.4 — Operational scale

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| P-Ext.4.1 | **Lazy bootstrap** — register only bundles in active `*Profile` | **Done** | High | `catalog_bootstrap.py`, bootstrap modules | `tool_bundle_ids`, `skill_bundle_ids`, `integration_preset` |
| P-Ext.4.2 | **`CatalogSnapshot` API** (read-only) | **Done** | Medium | `intergrax/core/catalog_snapshot.py` | list slugs for docs/UI |
| P-Ext.4.3 | Slug conflict policy in bootstrap | **Done** | Medium | `catalog_bootstrap.py` | `error` / `warn_override` |
| P-Ext.4.4 | CI **`check_plugin_catalog.py`** | **Done** | High | `scripts/` | smoke: shipped bundles present |
| P-Ext.4.5 | **Expand CI smoke** — all three catalog counts | **Done** | Medium | `scripts/check_plugin_catalog.py` | tools **13** bundles / ~**29** tool_id; skills **3** bundles / **8** skill_id; integrations **core≥12**, **full≥95** (see also P-Ext.1.8) |

---

#### P-Ext.5 — Docs, scaffold, canon

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| P-Ext.5.1 | Scaffold **`new_integration` / `new_tool_bundle` / `new_skill_bundle`** | **Done** | Medium | `intergrax/scaffold/` | manifest + plugin + register |
| P-Ext.5.2 | **External plugins** sections in INTEGRATIONS/TOOLS/SKILLS | **Done** | Medium | `docs/` | Cross-link Appendix I |
| P-Ext.5.3 | **Canon §7.1.5.1** — entry points + plugin protocols | **Done** | High | `intergrax_runtime_architecture.md` | §7.1.5.1 Tier-0 Plugin Catalogs |
| P-Ext.5.4 | Remove duplicate `PLUGIN_CATALOG_PLAN.md` | **Done** | Low | — | tracking only in this plan + Appendix I |
| P-Ext.5.5 | **Prod path matrix** in author guide (integration vs tool vs skill) | **Done** | Medium | `guides/EXTENSION_AUTHOR_GUIDE.md` | two-phase tool bootstrap documented |
| P-Ext.5.6 | **Lab wiring recipe** for external plugins | **Done** | Medium | `applications/lab_application/`, `TIER3_READINESS.md` | `discover_entry_points` + profile example |

---

#### P-Ext.6 — Production closure (paydown)

**Goal:** Close gaps between **MVP** (API + shipped catalogs) and **production-ready extensibility** (tested pip install, parity across three layers, ops hooks).

| # | Deliverable | Status | Priority | Depends on | Acceptance |
|---|-------------|--------|----------|------------|------------|
| P-Ext.6.1 | **Fixture pip package** (unblocks EP tests) | **Done** | **Critical** | — | same as P-Ext.0.5 |
| P-Ext.6.2 | **External tool + skill examples + tests** | **Done** | **Critical** | 6.1 | P-Ext.2.9–2.11, P-Ext.3.6–3.8, 3.7 green |
| P-Ext.6.8 | **Skill Tier-3 + scaffold** (rollup) | **Done** | Medium | — | P-Ext.3.9–3.12, scaffold overlap P-Ext.5.1 |
| P-Ext.6.9 | **Tool Tier-3 lazy wiring** (rollup) | **Done** | Medium | — | P-Ext.2.12 (symmetric with P-Ext.3.9) |
| P-Ext.6.10 | **Tier-3 lazy wiring** (all catalogs rollup) | **Done** | Medium | — | P-Ext.2.12 + P-Ext.3.9 + optional `integration_preset` in shared helpers |
| P-Ext.6.3 | **EP discovery** in tests + lab env flag | **Done** | High | 6.1 | P-Ext.0.6–0.7, P-Ext.1.6 |
| P-Ext.6.4 | **IntegrationSlug cleanup** in docs/scripts | **Done** | Medium | — | P-Ext.1.5 |
| P-Ext.6.5 | **Scaffold** `new_tool_bundle` / `new_skill_bundle` / `new_integration` | **Done** | Medium | — | P-Ext.5.1 |
| P-Ext.6.6 | **Integration Tier-3** + typed resolve + health (rollup) | **Done** | Medium | — | P-Ext.1.3a, 1.4, 1.8–1.11 |
| P-Ext.6.7 | **Conflict policy** + expanded CI smoke | **Done** | Medium | — | P-Ext.4.3, P-Ext.4.5, P-Ext.1.8 |

**DoD (phase closure):** Appendix I has no **Planned** P0/P1 rows; external integration, tool, and skill each proven via **entry point** (fixture package), not only explicit in-process registration.

---

#### Phase P-Ext — Definition of done

**MVP (met 2026-06-02):**

1. `bootstrap_catalogs()` + three plugin protocols + lazy presets.
2. All shipped tool/skill bundles on `ToolPlugin` / `SkillPlugin`.
3. Integration example `custom_memory_kv` + `test_external_plugin.py`.
4. Canon §7.1.5.1 + `guides/EXTENSION_AUTHOR_GUIDE.md` (EN).
5. Gate: `tests/unit/core/plugins`, integrations/tools/skills plugin tests green.

**Production closure (P-Ext.6 — open):**

1. **Fixture pip package** registers integration + tool + skill without Intergrax core edits.
2. **EP discovery tests** for all three groups (`discover_entry_points=True`).
3. **External tool test** — `RuntimeToolInvoker` after EP registration.
4. **External skill test** — `allowed_tools` merge after EP registration.
5. **Tier-3** documents/env for optional discovery; default remains explicit bootstrap.
6. **Tier-3 lazy wiring** — `tool_wiring` and `skill_wiring` pass profile bundle ids to `bootstrap_catalogs()` (P-Ext.2.12, P-Ext.3.9).
7. **No central slug enum** in new code/docs (string slugs); `IntegrationSlug` removed from author-facing examples.
8. **MCP export** from active `ToolRegistry` (already met).
9. Appendix I: all P-Ext.* rows **Done** or **Won't fix** with reason.

#### Phase P-Ext — Recommended execution order

```text
MVP (Done):               P-Ext.0.1–0.4 | P-Ext.1.1–1.2 | P-Ext.2.1–2.8 | P-Ext.3.* | P-Ext.4.1–4.2,4.4 | P-Ext.5.2–5.4

Paydown Wave P1 (critical):
  P-Ext.0.5 → P-Ext.0.6 → P-Ext.1.6 → P-Ext.1.10
           → P-Ext.2.9 → P-Ext.2.10 → P-Ext.2.11
           → P-Ext.3.6 → P-Ext.3.7 → P-Ext.3.8

Paydown Wave P2 (ops + docs):
  P-Ext.0.7 → P-Ext.4.3 → P-Ext.4.5 → P-Ext.1.8 → P-Ext.1.5 → P-Ext.1.7 → P-Ext.5.5 → P-Ext.5.6
           → P-Ext.2.12 → P-Ext.3.9 → P-Ext.3.10 → P-Ext.3.11

Paydown Wave P3 (optional polish):
  P-Ext.1.3a → P-Ext.1.4 → P-Ext.5.1 → P-Ext.3.12
```

**Effort estimate:** MVP ~21–32 person-days (**spent**); paydown **~12–18** person-days incl. integration + tool + skill closure (Appendix I).

**Priority ladder:** **Band 2c** (§4.0) — harness Tier-0 extensibility; **not** Band 3 product work.

---

## 4. Priority Order

### 4.0 Implementation priority ladder (canonical)

**Read this before §6.** The plan has three bands. Implement **top to bottom**. **Never** pull items from band 3 into “next step” summaries while band 1–2 are the active policy.

| Band | What | Status (2026-06-05) | Examples |
|------|------|---------------------|----------|
| **1 — Harness platform** | Tier-0/1/3 lab wiring, security, policy, typing, legacy removal, gate audits | **Maintenance** (§4.1 **Done**; keep green) | `pytest -m gate`, `check_harness_*`, `check_legacy_modules_removed.py`, regression fixes |
| **2 — Harness architecture hardening** | Capability graph, lifecycle governance, prompt/eval/context/security/cost/metrics hardening — **no** business domain | **Done** (2026-06-05) | V-CG … V-KG, V-V6 closeout · V-REM |
| **2i — Phase V runtime remediation (V-REM)** | Close 9 Partial Phase V + EvalRunner gate gaps — runtime enforcement, not new OS features | **Done** (2026-06-05) | [Phase V-REM](#phase-v-rem--phase-v-runtime-remediation-audit-closeout) · Appendix J |
| **2b — Modality plane (optional parallel)** | Vision CV, speech, classical ML — harness Tier-0 only | **Done** | W-ML complete; optional Celery bus wiring for Tier-3 scale-out |
| **2c — Plugin catalogs (P-Ext)** | Entry points + `ToolPlugin` + `SkillPlugin` + `bootstrap_catalogs()` | **Done** (2026-06-02) | Appendix I · [guides/EXTENSION_AUTHOR_GUIDE.md](guides/EXTENSION_AUTHOR_GUIDE.md) |
| **2d — Operational L3 (W-OPS)** | Reliability, identity, SLO/ops evidence, online eval — **no** business agents | **Done** (2026-06-06) | [Phase W-OPS](#phase-w-ops--operational-harness-maturity-ideal-l3-ops) · `phase_w_ops_evidence.py` |
| **2e — Application environment (H-APP)** | `ApplicationEnvironmentProfile`, unified Tier-3 wiring, host migration — **no** business agents | **Done** (2026-06-03) | [Phase H-APP](#phase-h-app--tier-3-application-environment-full-configurability) · [`HARNESS_APPLICATION_LAYER_AUDIT.md`](HARNESS_APPLICATION_LAYER_AUDIT.md) · **§6.2x** |
| **2f — Developer authoring UX (DX)** | LangGraph-like facades, minimal scaffold, CLI run/doctor, TTFRun gates, UI spec export — **no** business agents | **Done** (2026-06-03) | [Phase DX](#phase-dx--developer-authoring-experience-fast-environment--agent-builds) · **§6.2y** |
| **2g — Agents & applications conformance (AA)** | Scaffold alignment, per-agent/app `ARCHITECTURE.md`, deploy triad, legal **scaffold** reset (domain steps → Band 3) | **Mostly Done** (2026-06-02) | [Phase AA](#phase-aa--agents--applications-conformance-scaffold-docs-deploy) · **§6.2z** · [§4.0a](#40a-implementation-scope-split-infrastructure-vs-business) |
| **2h — Memory platform (MEM)** | H-APP→runtime bridge, durable user LTM, session SQLite, gates, hooks, memory docs — **no** business agents | **Done** (2026-06-02) | [Phase MEM](#phase-mem--memory-platform-completion) · **§6.2aa** |
| **2j — Orchestration closeout (ORCH)** | Wire `planner_kind`/`classifier_kind`, `ApplicationGraphSpec`→plan, graph concurrency cap — **no** business agents | **Done** (2026-06-05) | [Phase ORCH](#phase-orch--orchestration-control-plane-closeout) · **§6.1b** · **§6.2bb** |
| **2k — Tools/skills closeout (TS)** | Catalog→`RuntimeConfig` bridge, harness LLM wiring, `SkillResolverProtocol`, Appendix J — **no** business agents | **Done** (2026-06-02) | [Phase TS](#phase-ts--tools--skills-control-plane-closeout) · **§6.1c** · **§6.2bc** |
| **2l — Integration closeout (INT)** | `integration_runtime_bridge`, bootstrap health probes, Appendix K — **no** business agents | **Done** (2026-06-02) | [Phase INT](#phase-int--integration-control-plane-closeout) · **§6.1d** · **§6.2bd** |
| **2m — RAG closeout (RAG)** | `rag_runtime_bridge`, RAG stack on environment wire — **no** business agents | **Done** (2026-06-02) | [Phase RAG](#phase-rag--rag-retrieval-control-plane-closeout) · **§6.1e** · **§6.2be** |
| **2n — Context engineering closeout (CTX)** | `context_runtime_bridge`, `context_wiring`, Nexus `ContextManager` wire — **no** business agents | **Done** (2026-06-02) | [Phase CTX](#phase-ctx--context-engineering-control-plane-closeout) · **§6.1f** · **§6.2bf** |
| **2o — Legacy tool plan closeout (LEG)** | `tool_ids` canonical path; gateway/engine planner migration — **no** business agents | **Done** (2026-06-02) | [Phase LEG](#phase-leg--legacy-tool-plan-boolean-closeout) · **§6.1h** |
| **2p — Prompt registry closeout (PE)** | `PromptProfile`, `prompt_runtime_bridge`, `prompt_wiring`, Appendix M — **no** business agents | **Done** (2026-06-02) | [Phase PE](#phase-pe--prompt-registry-control-plane-closeout) · **§6.1i** |
| **2q — Agent assembly closeout (AS)** | Agent contract conformance, capability/skill resolution, lifecycle state — **no** business agents | **Done** (2026-06-02) | [Phase AS](#phase-as--agent-assembly-control-plane-closeout) · **§6.1k** · **Appendix N** |
| **2r — Registry architecture closeout (REG)** | Registry snapshot, assembly resolver, host resolution CI — **no** business agents | **Done** (2026-06-02) | [Phase REG](#phase-reg--registry-architecture-control-plane-closeout) · **§6.1l** · **Appendix O** |
| **2s — Capability graph closeout (CG)** | Environment graph slice, wire-time validation, CI audit — **no** business agents | **Done** (2026-06-02) | [Phase CG](#phase-cg--capability-graph-control-plane-closeout) · **§6.1m** · **Appendix P** |
| **2t — Observability closeout (OBS)** | Profile bridge, assembly resolver, host wiring CI — **no** business agents | **Done** (2026-06-02) | [Phase OBS](#phase-obs--observability-control-plane-closeout) · **§6.1n** · **Appendix Q** |
| **2u — Reliability closeout (REL)** | Idempotency bridge, circuit breaker wire, assembly resolver CI — **no** business agents | **Done** (2026-06-02) | [Phase REL](#phase-rel--reliability-control-plane-closeout) · **§6.1o** · **Appendix R** |
| **2v — Security closeout (SEC)** | V-SEC bridge, middleware assembly resolver, host CI — **no** business agents | **Done** (2026-06-02) | [Phase SEC](#phase-sec--security-control-plane-closeout) · **§6.1q** · **Appendix S** |
| **2w — Cost governance closeout (COST)** | Budget bridge, policy bundle merge, assembly resolver CI — **no** business agents | **Done** (2026-06-02) | [Phase COST](#phase-cost--cost-governance-control-plane-closeout) · **§6.1r** · **Appendix T** |
| **2x — Evaluation closeout (EVAL)** | Registry bridge, policy bundle merge, assembly resolver CI — **no** business agents | **Done** (2026-06-02) | [Phase EVAL](#phase-eval--evaluation-control-plane-closeout) · **§6.1s** · **Appendix U** |
| **2y — Adaptive Harness Intelligence (W-ADAPT)** | L4 **runtime** closed loop — SignalCollector, AdaptationEngine, ProfileVersionStore, verify/rollback — **no** business agents | **Done** (2026-06-02) — **70/70 Done** | [Phase W-ADAPT](#phase-w-adapt--adaptive-harness-intelligence-l4-runtime) · [`architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md`](architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md) · **§6.1t** · **§6.2ac** · **Appendix K** |
| **2z — LLM completion envelope (M-LLM-R)** | Typed `LLMAdapterResponse` replaces `str`/`dict` adapter returns; full consumer refactor — **no** business agents | **Done** (2026-06-06) — **39/39** | [Phase M-LLM-R](#phase-m-llm-r--llm-completion-response-envelope-audit-2026-06-06) · **§6.1v** · **§6.2ad** · **Appendix L** |
| **2aa — Integration expansion (M.6 P4)** | 28 harness-ROI provider slugs (secrets, observability stack, OLAP, feature flags, prod deploy) — **no** business agents | **Done** (2026-06-02) — **28/28** | [M.6 P4 register](#m6-p4--harness-platform-expansion-done) · **§6.1w** · **§6.2ae** |
| **2ab — Integration depth (M.6 P5)** | Harden 25 beta + 8 greenfield harness slugs (metrics, CI/CD, eval, async, data plane) — **no** business agents | **Done** (2026-06-02) — **33/34** | [M.6 P5 register](#m6-p5--harness-integration-depth-done--3334) · **§6.1x** · **§6.2af** |
| **2ac — Integration expansion (M.6 P6)** | 32 harness slugs + post-catalog wiring (tools, bridges, promote gate, infra `p6`) — **no** business agents | **Done** (2026-06-02) — **32/32 + M-P6-WIRE** | [M.6 P6 register](#m6-p6--harness-integration-expansion-planned) · **§6.1y** · **§6.2ag** |
| **2ad — FAUDIT-32 remediation** | Close 32-layer audit residuals (tier gate, intake, observability taxonomy, registry depth, eval release gate) — **no** business agents | **Done** (2026-06-06) — **23/23 + §6.1ai follow-up** | [Phase FAUDIT-32](#phase-faudit-32--full-architecture-audit-closeout) · **§6.1ah** · **§6.1ai** · **Appendix M** |
| **2aj — Nexus execution depth (FLOW)** | Close `FLOW-GAP.*` (01–16) — delegation, SubtaskContract, backpressure profile, LLM planner, merge, eval, graph hardening — **no** K.1/K.2 | **Done** (2026-06-07) — **17/18** (**FLOW-8 Deferred**) | [Phase FLOW](#phase-flow--nexus-execution-depth) · **§6.1aj** · **§6.2aj** · **Appendix N (FLOW)** |
| **2ak — Critic & Verification Layer (CRIT-V)** | PEV verify depth — `CriticOrchestrator`, `eval.judge`, `eval.trajectory`, evaluator-loop, semantic offline runner — **no** business agents | **Done** | [Phase CRIT-V](#phase-crit-v--critic--verification-layer) · [`architecture/CRITIC_VERIFICATION.md`](architecture/CRITIC_VERIFICATION.md) · **§6.1ak** · **§6.2ak** · canon §55 · [ADR-CRITIC-001](adr/ADR-CRITIC-001.md) |
| **2al — Unified Observability Spine (OBS-BUS)** | Full HOS — typed payloads, `ObservabilityEmitter`, emission coverage, extension SDK, L4 §21 — **no** business agents | **Done** | [Phase OBS-BUS](#phase-obs-bus--unified-observability-spine) · [`architecture/OBSERVABILITY.md`](architecture/OBSERVABILITY.md) · **§6.1al** · [ADR-OBS-001](adr/ADR-OBS-001.md) |
| **2am — Memory intelligence depth (MEM-DEPTH)** | Context Compiler, never-overflow invariant, lifecycle automation, explore delegation, entity memory — **no** business agents | **Planned** (0/26) | [Phase MEM-DEPTH](#phase-mem-depth--memory-intelligence-depth) · [`architecture/MEMORY.md`](architecture/MEMORY.md) · **§6.2ab** |
| **3 — END OF PLAN (product)** | Business agents, new product Tier-3 apps, domain skills, Legal live E2E | **Deferred** — **[§6.3](#63-end-of-plan--deferred-product-work-only)** | K.1, K.2, `applications/<product>/`, K.6, B.15, S-Ops.4 · FLOW-8 |

**Hard rule:** Band 3 is **not** “next after harness.” It runs only after an **explicit product prioritization decision** (Appendix A for agents; separate decision for new applications). Until then, **do not** implement, extend, or schedule K.1/K.2 waves, new product hosts, or product-only E2E in implementation cadence (§6.1–§6.2).

**Policy (2026-06-07):** Harness completion in §4.1 is **Done**. Band 1 = keep gate green on every PR. Bands **2j–2ad** platform closeouts = **Done**. **Band 2aj (Phase FLOW)** = **Done** (17/18; FLOW-8 Deferred). **Band 2ak (Phase CRIT-V)** = **Done** (24/24). Band 3 = **frozen** unless leadership reprioritizes.

```text
BAND 1:  Harness maintenance — gate + audit scripts (§6.1) — every PR
BAND 2y: Adaptive Harness Intelligence — Phase W-ADAPT (§6.1t) — DONE (70/70)
BAND 2z: LLM completion envelope — Phase M-LLM-R (§6.1v) — DONE (2026-06-06)
BAND 2j: Orchestration closeout — Phase ORCH (§6.1b) — DONE (2026-06-05)
BAND 2:  Harness architecture hardening — Phase V + V-REM — DONE (2026-06-05)
BAND 2i: Phase V runtime remediation — V-REM — DONE (2026-06-05)
BAND 2d: Operational L3 — Phase W-OPS (§6.2w) — DONE
BAND 2e: Application environment — Phase H-APP (§6.2x) — DONE (43 tasks)
BAND 2f: Developer authoring UX — Phase DX (§6.2y) — DONE (47 tasks)
BAND 2g: Agents & applications conformance — Phase AA (§6.2z) — MOSTLY DONE (platform); domain → Band 3
BAND 2h: Memory platform — Phase MEM (§6.2aa) — DONE (48/48)
BAND 2j: Orchestration closeout — Phase ORCH (§6.1b) — DONE (ORCH-1 → ORCH-4)
BAND 2k: Tools/skills closeout — Phase TS (§6.1c) — DONE (TS-1 → TS-3)
BAND 2l: Integration closeout — Phase INT (§6.1d) — DONE (INT-1 → INT-2)
BAND 2m: RAG closeout — Phase RAG (§6.1e) — DONE (RAG-1)
BAND 2n: Context engineering closeout — Phase CTX (§6.1f) — DONE (CTX-1 → CTX-2)
BAND 2o: Legacy tool plan closeout — Phase LEG (§6.1h) — DONE (LEG-1 → LEG-3)
BAND 2p: Prompt registry closeout — Phase PE (§6.1i) — DONE (PE-1 → PE-3)
BAND 2q: Agent assembly closeout — Phase AS (§6.1k) — DONE (AS-1 → AS-3)
BAND 2r: Registry architecture closeout — Phase REG (§6.1l) — DONE (REG-1 → REG-3)
BAND 2s: Capability graph closeout — Phase CG (§6.1m) — DONE (CG-1 → CG-3)
BAND 2t: Observability closeout — Phase OBS (§6.1n) — DONE (OBS-1 → OBS-3)
BAND 2u: Reliability closeout — Phase REL (§6.1o) — DONE (REL-1 → REL-3)
BAND 2v: Security closeout — Phase SEC (§6.1q) — DONE (SEC-1 → SEC-3)
BAND 2w: Cost governance closeout — Phase COST (§6.1r) — DONE (COST-1 → COST-3)
BAND 2x: Evaluation closeout — Phase EVAL (§6.1s) — DONE (EVAL-1 → EVAL-3)
BAND 2y: Adaptive Harness Intelligence — Phase W-ADAPT (§6.1t) — DONE (70/70, Wave 0–7 Done)
BAND 2z: LLM completion envelope — Phase M-LLM-R (§6.1v) — DONE (39/39)
BAND 2aa: Integration expansion — Phase M.6 P4 (§6.1w) — DONE (28/28)
BAND 2ab: Integration depth — Phase M.6 P5 (§6.1x) — DONE (33/34)
BAND 2ac: Integration expansion — Phase M.6 P6 (§6.1y) — DONE (32/32 + M-P6-WIRE)
BAND 2ad: FAUDIT-32 remediation — DONE (2026-06-06)
BAND 2aj: Nexus execution depth — Phase FLOW (§6.1aj) — DONE (17/18; FLOW-8 Deferred)
BAND 2ak: Critic & Verification Layer — Phase CRIT-V (§6.1ak) — **Done** (incl. CRIT-V-FOLLOWUP)
BAND 2al: Unified Observability Spine — Phase OBS-BUS (§6.1al) — **Done**
DONE:    Phase CLEAN — legacy module closeout (§6.1j) — 2026-06-02
BAND 3:  END OF PLAN — product agents & applications (§6.3) — DO NOT SCHEDULE AS DEFAULT NEXT

DONE:    Harness completion backlog (§4.1) — 2026-06-02
DONE:    Phase U — Harness production hardening (2026-06-01)
DONE:    Phase T — Harness cleanliness (2026-06-01)
DONE:    Phase S — Harness environment GA (2026-06-01)
DONE:    Phase Q+ — Harness Hardening (Appendix D)
DONE:    Phase R (MVP) — Harness AI alignment (Appendix E)
DONE:    Phase Q — Harness Quality (audit #1) — Waves 1–9
DONE:    Phase L, M, M-LLM, M-RAG, N, O — harness GA (functional)
DONE:    Phase K hardening K.3–K.5; Appendix B paydown (except B.15)

PARALLEL (harness-only): M.6 P6 integration expansion (§6.1y, **32 planned**); M.6 P5 residual `trivy` absorbed into P6 M-P6.1; legacy M.6 on-demand slugs; R-Skill catalog expansion (platform packs)

BAND 3 — END OF PLAN (see §6.3; not default “next”):
  • K.1 Problem Radar / K.2 Vendor Discovery (business agents)
  • K.6 / B.15 / S-Ops.4 — Legal live LLM E2E (product/CI)
  • New Tier-3 **product** applications (beyond lab + existing reference hosts)
  • Domain skill packs for product agents (until K.* started)
  • Problem Radar wave 2+ (`agents/problem_radar/` frozen)

RULE:    Strategy → canon → plan → code; Tier-1 via §0.6; four layers Integration → Tool → Skill → Agent
```

**Rationale:** Phases S/T/U + §4.1 delivered a production-configurable **harness**. Band 1–2 preserve and extend that platform. **Band 3 (product) is intentionally last** so business agents and new applications do not drive Tier-1 evolution (canon §52, [INTERGRAX_DEVELOPMENT_STRATEGY.md](INTERGRAX_DEVELOPMENT_STRATEGY.md)).

### 4.0a Implementation scope split (infrastructure vs business)

**Canonical rule:** Default implementation queue = **infrastructure only** (Bands 1–2g + §6.1). **Business** work runs only after explicit product prioritization — **[§6.3](#63-end-of-plan--deferred-product-work-only)**.

**Documentation rule:** This plan and [`intergrax_runtime_architecture.md`](intergrax_runtime_architecture.md) document **platform** delivery (Harness / Agent OS). They do **not** subsume `applications/<product>/IMPLEMENTATION_PLAN.md` or `agents/<name>/` product roadmaps — each business environment and business agent owns its architecture and deployment narrative.

| Layer | Bands / phases | What it includes | Default queue |
|-------|----------------|------------------|---------------|
| **Infrastructure (Intergrax Harness)** | 1, 2, 2b–2j (platform rows) | `intergrax/runtime/`, Tier-0 catalogs, H-APP, DX, MEM, ORCH, scaffold, CI audits, reference hosts | **Active** — §6.1 maintenance only |
| **Conformance shells (platform)** | 2g AA | `legal` / `legal_application` **scaffold** + deploy triad + tier hygiene (no domain UAEP steps) | **Done** (shell) |
| **Business agents & product apps** | 3, §6.3, AA-LEG.2.*, K.* | K.1/K.2, Legal UAEP steps, research/org domain tests, new `applications/<product>/`, live LLM E2E | **Deferred** — not default next |

**Module classification (repo inventory):**

| Module | Role | Queue |
|--------|------|-------|
| `agents/echo`, `agents/signoff_probe` | Harness reference Tier-2 | Infrastructure — **Done** |
| `agents/lab` | Lab mocks (not product agents) | Infrastructure — AA-LABAG.* optional |
| `applications/poc_template_application`, `applications/lab_application` | Reference Tier-3 hosts | Infrastructure — **Done** |
| `agents/legal`, `applications/legal_application` | Product shell on scaffold | Platform **Done**; domain logic **Deferred** (AA-LEG.2.2+) |
| `agents/research`, `applications/research_application` | Research prototype host | Platform **Done**; domain tests **Deferred** (AA-RES.4–5, AA-RESAPP.6) |
| `agents/organization_worker` | HITL / long-running demo | Docs **Done**; full scaffold + lab flag **Deferred** (AA-ORG.3–4) |
| `agents/problem_radar` | K.1 placeholder | **Frozen** — Band 3 (K.1) |
| New `applications/<product>/` beyond four hosts | Customer/product deploy | **Deferred** — §6.3 |

**Where to look for open work:**

| Topic | Section |
|-------|---------|
| **Canonical implementation queue (infrastructure)** | [§6.1](#61-harness-platform-maintenance-default--band-1) (**active** — maintenance) · [§6.1b](#61b-harness-implementation-queue--orchestration-closeout-closed) · [§6.1c](#61c-harness-implementation-queue--toolsskills-closeout-closed) · [§6.1d](#61d-harness-implementation-queue--integration-closeout-closed) · [§6.1e](#61e-harness-implementation-queue--rag-closeout-closed) (all closed) · [§6.1z](#61z-harness-implementation-queue-consolidated) (closed) |
| Integration catalog expansion (Done) | [M.6 P4](#m6-p4--harness-platform-expansion-done) · [§6.1w](#61w-harness-implementation-queue--integration-expansion-m6-p4-closed) — **28/28 Done** |
| Integration harness depth (Done) | [M.6 P5](#m6-p5--harness-integration-depth-done--3334) · [§6.1x](#61x-harness-implementation-queue--integration-depth-m6-p5-done) — **33/34 Done** |
| Integration harness expansion | [M.6 P6](#m6-p6--harness-integration-expansion-planned) · [§6.1y](#61y-harness-implementation-queue--integration-expansion-m6-p6-planned) — **Done** (32/32 + wiring) |
| Ongoing gate + audit scripts | [§6.1](#61-harness-platform-maintenance-default--band-1) |
| Memory platform wiring (Done) | [Phase MEM](#phase-mem--memory-platform-completion) · [§6.2aa](#62aa-phase-mem-execution-order-band-2h--active) |
| **Memory intelligence depth (active)** | [Phase MEM-DEPTH](#phase-mem-depth--memory-intelligence-depth) · [`architecture/MEMORY.md`](architecture/MEMORY.md) · [§6.1am](#61am-harness-implementation-queue--memory-intelligence-depth-active) · [§6.2ab](#62ab-phase-mem-depth-execution-order-band-2am--active) |
| All business / domain work | [§6.3](#63-end-of-plan--deferred-product-work-only) · [Business backlog register](#63a-business-backlog-register-consolidated) |

### 4.1 Harness completion backlog (execution order)

Work **one ID per PR**; gate green after each step. Map fixes to Appendix G where applicable.

| Order | ID | Deliverable | Priority | Notes |
|-------|-----|-------------|----------|-------|
| 1 | U-Leg.2 | Remove or archive `intergrax/rag/answers/`; migrate tests to `RetrievalService` | **Done** | `intergrax/legacy/rag_answers/`; import guard |
| 2 | U-Leg.1 | Freeze `ToolsAgent.run` — docs + `check_tools_agent_run.py` | **Done** | Deprecation + CI audit |
| 3 | U-Leg.3 | Sunset legacy plan booleans (`from_legacy`, `uses_legacy_booleans_only`) | **Done** | Warnings + `check_legacy_tool_plan_booleans.py` |
| 4 | U-Typ.4 | `profile.slug_for_category` + sandbox `session_id` typing | **Done** | No getattr on integration profile |
| 5 | U-Arch.2 | Typed `LabIntegrationWiring` — sqlite bundle types | **Done** | Removed `# type: ignore` on lab wiring |
| 6 | U-CI.3 | CI job: `LAB_STRICT_HARNESS` + API key | **Done** | `harness-strict` workflow job |
| 7 | R-Skill.* | `harness.skill_registry` platform skill | **Done** | Harness bundle + gate test |
| 8 | U-Con.* | `ResearchAgent` / `SummaryAgent` → `HarnessReferenceAgent` | **Done** | Lab `requires_uaep` when research enabled |

**Explicitly out of NOW:** K.1, K.2, Legal product E2E, new `applications/<product>/`, Problem Radar wave 2+.



---



## 5. Definition of Done (Global)



1. **Contract** — Pydantic / Protocol public API

2. **Trace** — state transitions emit `TraceEvent` (+ `RuntimeEvent` where wired)

3. **Test** — unit + integration, deterministic, no network

4. **Documentation** — update this plan + [`guides/AGENT_CREATION_GUIDE.md`](guides/AGENT_CREATION_GUIDE.md) when workflow changes

5. **No regression** — `pytest tests/ -m gate` green; Echo through NexusLoop

6. **Reuse Tier-0** — extend existing modules; no parallel LLM/log/trace stacks (§5.2)
7. **Architecture governance** — for Phase V streams, update compatibility/evaluation evidence (graph impact + score deltas)
8. **Security/cost controls** — hardening changes include policy-enforced tests for deny/degrade paths
9. **No product scope creep** — harness phases MUST NOT implicitly include K.1/K.2 or new product hosts



---

## Phase RAG — RAG retrieval control plane closeout

**Status:** **Done** (2026-06-02) — **3/3** deliverables Done (RAG-DOC.* + RAG-1); gate **612 passed**

**Audit basis:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](INTEGRAX_HARNESS_AUDIT_MAP.md) §14; author map: **Appendix K** §K.5.

**Priority ladder:** **Band 2m** (§4.0) — closed; default queue = **§6.1** maintenance.

**Execution order:** [§6.2be](#62be-phase-rag-execution-order-band-2m--closed) · queue: [§6.1e](#61e-harness-implementation-queue--rag-closeout-closed)

### RAG — Master register

| ID | Area | Deliverable | Status | Priority | Modules | Acceptance |
|----|------|-------------|--------|----------|---------|------------|
| RAG-DOC.1 | RAG0 | **Appendix K** §K.5 + AUDIT_MAP §14 cross-ref | **Done** | High | `docs/*` | RAG bridge documented |
| RAG-1 | RAG1 | **`rag_runtime_bridge.py`** + RAG stack on `wire_application_environment` | **Done** | **Critical** | `rag_runtime_bridge.py`, `environment_wiring.py`, `runtime_config_bridge.py` | `test_rag_runtime_bridge.py` |

### RAG — Paydown log

| Date | RAG ID | Summary |
|------|--------|---------|
| 2026-06-02 | RAG-DOC.1 | Appendix K §K.5 + plan sync |
| 2026-06-02 | RAG-1 | RAG runtime bridge + environment wire; gate **600** |

**Phase RAG complete when:** RAG-1 + RAG-DOC.* **Done**; §6.1e queue closed. **Status: complete (2026-06-02).**

---

## Phase CTX — Context engineering control plane closeout

**Status:** **Done** (2026-06-02) — **4/4** deliverables Done (CTX-DOC.* + CTX-1–2); gate **612 passed**

**Audit basis:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](INTEGRAX_HARNESS_AUDIT_MAP.md) §16; author map: `guides/AGENT_CREATION_GUIDE.md` **Appendix L**.

**Priority ladder:** **Band 2n** (§4.0) — closed; default queue = **§6.1** maintenance.

**Execution order:** [§6.2bf](#62bf-phase-ctx-execution-order-band-2n--closed) · queue: [§6.1f](#61f-harness-implementation-queue--context-engineering-closeout-closed)

### CTX — Master register

| ID | Area | Deliverable | Status | Priority | Modules | Acceptance |
|----|------|-------------|--------|----------|---------|------------|
| CTX-DOC.1 | CTX0 | **Appendix L** — context engineering control plane (§L.1–L.6) | **Done** | High | `guides/AGENT_CREATION_GUIDE.md` | TOC + verification table |
| CTX-DOC.2 | CTX0 | **Cross-ref sync** — plan, README, AUDIT_MAP §16, audit prompt ref #9 | **Done** | Medium | `docs/*` | Links resolve |
| CTX-1 | CTX1 | **`context_runtime_bridge.py`** — dedicated context profile → `RuntimeConfig` | **Done** | **Critical** | `context_runtime_bridge.py`, `runtime_config_bridge.py` | `test_context_runtime_bridge.py` |
| CTX-2 | CTX2 | **`context_wiring.py`** — `ContextManager` + task options from environment; `nexus_factory` wire | **Done** | High | `context_wiring.py`, `nexus_factory.py`, `harness_host_runtime.py` | `test_context_wiring.py` |

### CTX — Paydown log

| Date | CTX ID | Summary |
|------|--------|---------|
| 2026-06-02 | CTX-DOC.1, CTX-DOC.2 | Appendix L + cross-refs; AUDIT_MAP §16 |
| 2026-06-02 | CTX-1, CTX-2 | Context runtime bridge + Nexus ContextManager wiring; gate **608** |

**Phase CTX complete when:** CTX-1–2 + CTX-DOC.* **Done**; §6.1f queue closed. **Status: complete (2026-06-02).**

---

