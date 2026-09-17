# MEM-FINAL-AUDIT-1 — Complete Architecture & Surface Inventory

**Stage:** inventory / classification / evidence mapping only — **NOT** enterprise certification.  
**Baseline HEAD (audit execution):** `aecc235ecba0559cb523992d49aa82539926896c` · branch `development` · clean working tree.  
**Auditor:** Cursor agent session (must be independently verified on GitHub before MEM-FINAL-AUDIT-2).

---

## 1. Repo state

| Field | Value |
| ----- | ----- |
| HEAD | `aecc235ecba0559cb523992d49aa82539926896c` |
| Branch | `development` (tracking `origin/development`) |
| Working tree | clean |
| Foreign WIP | none at audit time |
| Conflicts | none |

---

## 2. Audit scope

- `intergrax/memory/**` (103 Python modules under package tree)
- Memory contracts, stores, strategies, resolver, provider qualification, recall pipeline
- Runtime consumers: `intergrax/runtime/**`, `intergrax/applications/_shared/*memory*`, `intergrax/tools/providers/memory/**`, CE/Nexus context recall
- Organization memory: `intergrax/runtime/organization/**` + `intergrax/memory/org_memory_*.py`
- Task memory: `intergrax/runtime/task_memory/**` (adjacent to `TaskMemoryCapability`)
- Tests: `tests/unit/memory/**`, `tests/integration/memory/**`, related integration (MEM-XINT, applications wiring)
- Docs: `docs/project/architecture/MEMORY*.md`, plans, plugin guides, qualification boundary doc
- **Out of scope for Memory proofs:** `platform_proofs/**` (no Memory-domain qualification scenarios found; GPU “memory” strings only)

---

## 3. Master Memory inventory

Category key: **A** canonical enterprise · **B** derived/projection · **C** specialized · **D** composition/integration · **E** legacy/compat · **F** test/qual only · **G** documented unproven · **H** bypass/debt · **I** dead/unused candidate.

| ID | Element | Path | Cat | Contract | Owner | RT reachable | Pluggable | Persist | Vendor proof | E2E | Docs | Status |
| -- | ------- | ---- | --- | -------- | ----- | ------------ | --------- | ------- | ------------ | --- | ---- | ------ |
| M-001 | MemoryControlPlane | `contracts/memory_control.py` | A | Protocol | memory | yes | via plane inject | n/a | n/a | MEM-ENT-15 e2e | MEMORY_ARCH | ENTERPRISE-READY NOT FULLY PROVEN |
| M-002 | DefaultMemoryControlPlane | `default_memory_control_plane.py` | A | impl | memory | yes | partial | n/a | n/a | yes | yes | PARTIAL |
| M-003 | UserProfileMemoryCapability | `contracts/memory_control.py` | A | Protocol | memory | yes | adapter | n/a | n/a | yes | yes | ENTERPRISE-READY NOT FULLY PROVEN |
| M-004 | UserProfileManagerMemoryCapability | `default_memory_control_plane.py` | D | adapter | memory | yes | no | n/a | n/a | yes | yes | PARTIAL |
| M-005 | TaskMemoryCapability | `contracts/memory_control.py` | A | Protocol | memory/runtime | host-dependent | inject | host store | gap | partial | yes | GAP |
| M-006 | EpisodicMemoryCapability | `contracts/memory_control.py` | C | Protocol (minimal) | memory | optional | inject | STI/RAG | gap | gap | yes | GAP |
| M-007 | UserProfileStore | `user_profile_store.py` | A | Protocol | memory | yes | EP plugin | yes | SQLite qual V3–V4 | restart e2e | yes | ENTERPRISE-READY NOT FULLY PROVEN |
| M-008 | InMemoryUserProfileStore | `stores/in_memory_user_profile_store.py` | F | impl | memory | dev/test | plugin | no | V2 | synth | yes | NOT IN SCOPE production |
| M-009 | SQLiteUserProfileStore | `stores/sqlite_user_profile_store.py` | A | impl | memory | yes | plugin | yes | V4 ref | MEM-ENT-15 restart | yes | ENTERPRISE-READY NOT FULLY PROVEN |
| M-010 | DocumentStoreUserProfileStore | `stores/document_store_user_profile_store.py` | C | impl | memory | composable | plugin | yes | audit gap | unit | yes | PARTIAL |
| M-011 | UserProfileManager | `user_profile_manager.py` | A | service | memory | yes | store inject | via store | via store | yes | yes | ENTERPRISE-READY NOT FULLY PROVEN |
| M-012 | UserProfile / UserProfileMemoryEntry | `user_profile_memory.py` | A | models | memory | yes | n/a | via store | via store | yes | yes | ENTERPRISE-READY NOT FULLY PROVEN |
| M-013 | UserProfileMemoryLifecycleCoordinator | `user_profile_memory_lifecycle.py` | A | service | memory | yes | projection inject | n/a | n/a | yes | yes | PARTIAL |
| M-014 | UserProfileMemoryProjection | `contracts/memory_lifecycle.py` | B | Protocol | memory | yes | custom | derived | n/a | reconcile tests | yes | ENTERPRISE-READY NOT FULLY PROVEN |
| M-015 | UserProfileLtmVectorProjection | `user_profile_ltm_vector_projection.py` | B | impl | memory | host-dependent | ports | vector | LTM/RAG split | reconcile | yes | PARTIAL |
| M-016 | Enterprise memory record types | `contracts/enterprise_memory_record.py` | A | dataclasses | memory | yes | n/a | serialized | n/a | unit | yes | PARTIAL |
| M-017 | Memory lifecycle / reconciliation | `contracts/memory_lifecycle.py` | A | types+Protocol | memory | yes | n/a | n/a | n/a | e2e | yes | ENTERPRISE-READY NOT FULLY PROVEN |
| M-018 | Memory recall public types | `contracts/memory_recall.py` | A | types | memory | yes | n/a | n/a | n/a | yes | yes | PARTIAL |
| M-019 | MemoryRecallStrategySet | `recall_strategy_bundle.py` | A | bundle | memory | yes | replaceable | no | n/a | MEM-ENT-6 | yes | PARTIAL |
| M-020 | Memory recall pipeline | `recall/pipeline.py` | A | fn | memory | yes | strategies | no | n/a | unit | partial | PARTIAL |
| M-021 | EntityTemporalMemoryStore | `contracts/entity_temporal_memory.py` | C | Protocol | memory | yes | EP | yes | in-mem qual V2 | unit/e2e partial | yes | GAP durable vendor |
| M-022 | EntityTemporalMemoryCapability | `contracts/entity_temporal_memory.py` | C | Protocol | memory | yes | service | yes | gap | partial | yes | GAP |
| M-023 | EntityMemoryIndexer | `contracts/entity_temporal_memory.py` | B | Protocol | memory | yes | DefaultEntityMemoryIndexer | derived | gap | yes | yes | PARTIAL |
| M-024 | DefaultEntityMemoryIndexer | `entity_memory_indexing.py` | B | impl | memory | yes | replace | derived | gap | yes | yes | PARTIAL |
| M-025 | EntityTemporalMemoryService | `entity_temporal_memory_service.py` | C | service | memory | apps wiring | store inject | yes | gap | partial | yes | GAP |
| M-026 | InMemoryEntityTemporalMemoryStore | `stores/in_memory_entity_temporal_memory_store.py` | F | impl | memory | test/ref | plugin | no | V2 qual | synth | yes | NOT production |
| M-027 | Entity temporal plugin | `stores/in_memory_entity_temporal_memory_plugin.py` | F | plugin | memory | test | EP | no | V2 | synth | yes | qual reference |
| M-028 | EntityTemporalNoop | `entity_temporal_noop.py` | E | noop | memory | optional | yes | no | n/a | n/a | yes | LEGACY/compat |
| M-029 | EntityGraphMemoryStore | `entity_graph_memory.py` | E | legacy store | memory | guarded | no | optional | n/a | bypass tests | partial | LEGACY |
| M-030 | EntityGraphMemoryService | `entity_graph_service.py` | E | service | memory | apps | no | optional | n/a | partial | partial | LEGACY |
| M-031 | ProcedureMemoryStore | `contracts/procedural_memory.py` | C | Protocol | memory | yes | EP | yes | in-mem qual | unit | yes | GAP durable vendor |
| M-032 | ProcedureMemoryCapability | `contracts/procedural_memory.py` | C | Protocol | memory | yes | service | yes | gap | partial | yes | GAP |
| M-033 | ProceduralMemoryService | `procedural_memory_service.py` | C | service | memory | composable | store inject | yes | gap | MEM-ENT-8 | yes | GAP |
| M-034 | DefaultProceduralMemoryIndexer | `procedural_memory_indexing.py` | B | impl | memory | yes | replace | derived | gap | yes | yes | PARTIAL |
| M-035 | InMemoryProceduralMemoryStore | `stores/in_memory_procedural_memory_store.py` | F | impl | memory | test | plugin | no | V2 qual | synth | yes | ref |
| M-036 | ProceduralMemoryNoop | `procedural_memory_noop.py` | E | noop | memory | optional | yes | no | n/a | n/a | yes | compat |
| M-037 | LongHorizonMemoryStore | `contracts/long_horizon_memory.py` | C | Protocol | memory | yes | EP | yes | in-mem qual | unit | yes | GAP durable vendor |
| M-038 | LongHorizonMemoryService | `long_horizon_memory_service.py` | C | service | memory | composable | store inject | yes | gap | MEM-ENT-9 | yes | GAP |
| M-039 | InMemoryLongHorizonMemoryStore | `stores/in_memory_long_horizon_memory_store.py` | F | impl | memory | test | plugin | no | V2 qual | synth | yes | ref |
| M-040 | SessionTurnIndexStore | `contracts/session_turn_index.py` | C | Protocol | memory | host-dependent | EP | vector | gap | **P1 gap** | yes | GAP |
| M-041 | VectorSessionTurnIndexStore | `session_turn_index_service.py` | C | impl | memory | host-dependent | ports | vector | gap | partial unit | yes | GAP |
| M-042 | InMemorySessionTurnIndexStore | `stores/in_memory_session_turn_index_store.py` | F | impl | memory | test | ref | no | V2 qual | synth | yes | ref |
| M-043 | SessionTurnIndex recall metadata | `session_turn_index_recall_metadata.py` | B | helper | memory | CE path | n/a | n/a | gap | gap | partial | PARTIAL |
| M-044 | Memory store plugin contracts | `contracts/memory_store_plugin.py` | A | Protocol | memory | yes | EP `intergrax.memory_stores` | varies | fixture EP | plugin e2e | guide | PARTIAL |
| M-045 | Provider qualification | `provider_qualification/**` | F | framework | memory | CI/qual | yes | harness | V2–V3 ref | qual not full E2E | yes | qual only |
| M-046 | Resolver discovery/classifier/resolver | `resolver/**` | D | infra | memory | yes | EP | n/a | fixture | unit/e2e plugin | yes | PARTIAL |
| M-047 | MemorySecurityGovernanceService | `memory_security_governance_service.py` | A | service | memory | yes | policy inject | n/a | n/a | MEM-ENT-10 | yes | ENTERPRISE-READY NOT FULLY PROVEN |
| M-048 | Governance contracts | `contracts/memory_security_governance.py` | A | Protocol/types | memory | yes | strategies | n/a | n/a | yes | yes | PARTIAL |
| M-049 | CanonicalMemoryGovernanceSourceAuthority | `canonical_memory_governance_source_authority.py` | A | impl | memory | yes | n/a | n/a | n/a | yes | yes | PARTIAL |
| M-050 | Specialized mutation/disclosure gov | `memory_specialized_*_governance.py` | A | helpers | memory | yes | n/a | n/a | n/a | yes | yes | PARTIAL |
| M-051 | MemoryObservabilitySink | `contracts/memory_observability.py` | A | Protocol | memory | yes | inject | n/a | n/a | MEM-ENT-12 | yes | PARTIAL |
| M-052 | MemoryDiagnosticEmitter | `memory_diagnostic_emitter.py` | A | impl | memory | yes | sink | n/a | n/a | yes | yes | PARTIAL |
| M-053 | Memory strategies bundle | `strategies/**` | A | Protocols+defaults | memory | yes | replace | no mutate store | n/a | MEM-ENT-4/6 | yes | PARTIAL |
| M-054 | ConversationalMemory | `conversational_memory.py` | E | aggregate | memory | session/chat | store | optional SQLite | sqlite store | unit | partial | LEGACY parallel |
| M-055 | ConversationalMemoryStore | `conversational_store.py` | E | Protocol | memory | session | inject | yes | sqlite | unit | partial | LEGACY |
| M-056 | InMemory/Sqlite conversational stores | `stores/*conversational*` | E | impl | memory | session | yes | yes | V2 sqlite | unit | partial | LEGACY |
| M-057 | Summary compressor | `summary_compressor.py` | B | utility | memory | token opt | hooks | no | n/a | unit | partial | PARTIAL |
| M-058 | memory_temporal helpers | `memory_temporal.py` | A | functions | memory | yes | n/a | n/a | n/a | yes | yes | PARTIAL |
| M-059 | memory_projection_failure | `memory_projection_failure.py` | A | classifier | memory | yes | n/a | n/a | n/a | yes | yes | PARTIAL |
| M-060 | cognitive_store_mapping | `cognitive_store_mapping.py` | G | mapping | memory | audit gap | n/a | n/a | n/a | maint | partial | G |
| M-061 | workspace_index_spike | `workspace_index_spike.py` | I | spike | memory | no | n/a | n/a | n/a | none | no | DEAD candidate |
| M-062 | postgres_memory_backend_rfc | `stores/postgres_memory_backend_rfc.py` | G | RFC spike | memory | no | n/a | n/a | V0 | none | no | NOT IMPLEMENTED |
| M-063 | OrgMemoryScope / maturity | `org_memory_scope.py`, `org_memory_maturity.py` | G | taxonomy | memory | harness | n/a | via UserProfile | n/a | unit maint | partial | G |
| M-064 | OrganizationProfile (runtime) | `runtime/organization/**` | C | parallel domain | runtime/org | yes | SQLite/in-mem | yes | sqlite integ | partial | MEMORY.md gap | PARTIAL not on plane |
| M-065 | TaskMemoryCoordinator (runtime) | `runtime/task_memory/**` | C | parallel | runtime | host flag | store | yes | sqlite unit | uaep integ | partial | PARTIAL |
| M-066 | build_default_memory_control_plane | `applications/_shared/memory_control_wiring.py` | D | composition | applications | yes | inject | n/a | n/a | e2e | yes | PARTIAL |
| M-067 | memory_context_invocation | `runtime/nexus/context/memory_context_invocation.py` | D | CE recall | runtime | yes | plane | n/a | n/a | MEM-XINT | yes | PARTIAL |
| M-068 | session_memory_consolidation | `runtime/user_profile/session_memory_consolidation_service.py` | D | consolidation | runtime | yes | uses plane.remember | n/a | n/a | yes | yes | PARTIAL |
| M-069 | Tools memory provider | `tools/providers/memory/service.py` | D | tools | tools | yes | task view + plane semantic | task | n/a | unit | partial | PARTIAL |
| M-070 | Package `__init__.py` public API | `memory/__init__.py` | G | exports summary only | memory | n/a | n/a | n/a | n/a | n/a | mismatch | P3 doc/API |

*(Table continues in section inventories below for strategies, tests, vendors.)*

---

## 4. Public contract inventory

| Contract | Path | Kind | Public export via `memory/__init__` | Replaceable | Hard invariant | Implementations (primary) | Consumers (primary) |
| -------- | ---- | ---- | ----------------------------------- | ------------- | -------------- | ------------------------- | ------------------- |
| MemoryControlPlane | `contracts/memory_control.py` | Protocol | no | inject | single semantic API | DefaultMemoryControlPlane | CE recall, tools LTM, consolidation |
| UserProfileStore | `user_profile_store.py` | Protocol | no | EP | canonical user facts | SQLite, InMemory, DocumentStore | UserProfileManager |
| UserProfileMemoryCapability | `memory_control.py` | Protocol | no | adapter | routed by plane | UserProfileManagerMemoryCapability | DefaultMemoryControlPlane |
| EntityTemporalMemoryStore/Capability/Indexer | `entity_temporal_memory.py` | Protocol | no | EP | derived not canonical | InMemory+plugins, services | Apps entity wiring, plane indirect |
| ProcedureMemoryStore/Capability | `procedural_memory.py` | Protocol | no | EP | governance | InMemory+plugins, ProceduralMemoryService | Specialized recall gov tests |
| LongHorizonMemoryStore | `long_horizon_memory.py` | Protocol | no | EP | derived summaries | InMemory+plugins, LongHorizonMemoryService | Compaction service |
| SessionTurnIndexStore | `session_turn_index.py` | Protocol | no | EP | episodic vector | InMemory, VectorSessionTurnIndexStore | SessionManager (host) |
| UserProfileMemoryProjection | `memory_lifecycle.py` | Protocol | no | yes | never canonical | LTM vector projection | Lifecycle coordinator |
| MemoryObservabilitySink | `memory_observability.py` | Protocol | no | inject | diagnostics | app wiring | emitter |
| MemoryRecallStrategySet | `recall_strategy_bundle.py` | dataclass bundle | no | yes | no direct store IO | defaults | DefaultMemoryControlPlane |
| Memory store plugins | `memory_store_plugin.py` | Protocol | no | EP | typed kind | fixture external package | resolver |
| Provider qualification | `provider_qualification.py` | types | no | harness | production gate | runner | MEM-ENT-13 |

**Public API audit:** `intergrax/memory/__init__.py` exposes **only** summary compression symbols — not canonical memory contracts. Contracts are imported from submodules (intentional cycle avoidance). **P3:** external docs may imply `from intergrax.memory import …` broader surface.

---

## 5. Runtime caller inventory (selected)

| Caller | Path | Surface used | Classification |
| ------ | ---- | ------------ | -------------- |
| CE LTM/episodic recall | `runtime/nexus/context/memory_context_invocation.py` | `plane.recall`, identity spine | canonical |
| Tool semantic search | `tools/providers/memory/service.py` | `plane.recall` + task view | canonical LTM / parallel task KV |
| Session consolidation | `runtime/user_profile/session_memory_consolidation_service.py` | `plane.remember` | canonical |
| Tier-3 host composition | `applications/_shared/memory_control_wiring.py` | builds DefaultMemoryControlPlane | composition |
| Entity projection wiring | `applications/_shared/entity_graph_wiring.py` | entity store/service | composition (specialized) |
| Session manager | `runtime/nexus/session/session_manager.py` | conversational + STI + profiles | mixed legacy + specialized |
| User profile instructions | `runtime/user_profile/user_profile_instructions_service.py` | UserProfileManager direct | legal internal (non-fact mutation path) |
| Agents ChatMessage | `agents/**`, `websearch/**` | conversational_memory types | legacy transcript model |
| Integration LTM wiring test | `tests/integration/applications/test_memory_vector_ltm_wiring.py` | direct manager.add_memory_entry | test composition |

---

## 6. Canonical MemoryControlPlane callers

| Operation | Canonical callers | Non-canonical / specialized | Potential bypass |
| --------- | ----------------- | --------------------------- | ---------------- |
| remember | DefaultMemoryControlPlane→capability; session_memory_consolidation_service; MEM-ENT-15 e2e | UserProfileManager in tests/instructions | integration test direct manager (F) |
| recall | memory_context_invocation; tools semantic path; plane tests | UserProfileManager.search_longterm_memory via capability only | Chat/session transcript stores (E) |
| forget | plane; governance tests | — | — |
| reconcile | plane; lifecycle coordinator | projection.reconcile direct (B internal) | — |
| apply_memory_supersession | plane; UserProfileManager lifecycle | — | — |

---

## 7. Potential bypasses

| Path | Caller | Target | RT reachable | Classification | Severity |
| ---- | ------ | ------ | ------------ | -------------- | -------- |
| Session transcript persistence | SessionManager / chat | ConversationalMemoryStore | yes | legacy parallel to plane | P2 |
| Entity graph legacy | entity_graph_memory | direct graph mutate | blocked by default | LEGACY guarded | P2 |
| Task KV tools | tools/providers/memory | TaskMemoryViewBinding | yes | legal parallel scope (TASK) | P2 evidence gap |
| Org profile memory | OrganizationProfileManager | org store not on plane | yes | separate domain | P1 cert gap |
| LTM vector projection | UserProfileLtmVectorProjection | vector ports | host-dependent | derived; must not write canonical | P2 if miswired |
| Direct manager in app test | test_memory_vector_ltm_wiring | add_memory_entry | test only | test-only | P3 |

---

## 8–16. Surface verdicts (summary)

| Surface | Verdict |
| ------- | ------- |
| **1 Semantic Control Plane** | ENTERPRISE-READY BUT NOT FULLY PROVEN — strong contracts + MEM-ENT-15 e2e; host wiring gaps for task/episodic |
| **2 UserProfile Memory** | ENTERPRISE-READY BUT NOT FULLY PROVEN — SQLite restart proof; external durable vendors not V5+ |
| **3 Entity/Temporal** | GAP — qual on in-memory; real durable entity store E2E missing |
| **4 Procedural** | GAP — service + qual; no production durable vendor E2E |
| **5 Long-Horizon** | GAP — same pattern as procedural |
| **6 SessionTurnIndex** | GAP — contract + vector adapter + qual; **P1** real vendor + Memory E2E |
| **7 Episodic** | GAP — placeholder `EpisodicMemoryCapability.recall_session_turns` |
| **8 Task Memory** | PARTIAL — runtime/task_memory + plane TASK scope; not all hosts enable |
| **9 Organization** | PARTIAL — runtime org profile store; taxonomy in memory package only |
| **10 Conversational** | LEGACY — parallel session history, not canonical LTM |
| **11 Strategies** | PARTIAL — replaceable; guarded not to own store mutation (tests enforce consolidation) |
| **12 Governance** | ENTERPRISE-READY BUT NOT FULLY PROVEN — MEM-ENT-10/15 evidence |
| **13 Identity** | PARTIAL — RequestIdentity spine for recall; MEM-ENT-15 R2 tests |
| **14 Observability** | PARTIAL — MEM-ENT-12 unit; full E2E trace qual gap |
| **15 Resolver/plugins** | PARTIAL — typed EP; production plugins external (fixture proves EP) |
| **16 Persistence/providers** | PARTIAL — SQLite user profile reference durable; others in-memory qual |

---

## 17. Enterprise Memory Record

Used in serialization (`user_profile_serialization.py`), governance, entity/procedural contracts, tests `test_enterprise_memory_record.py`. Runtime: all governed UserProfile entries and specialized records carrying provenance/trust/governance/lineage fields.

---

## 18–19. Temporal / supersession / reconciliation

- **Source of truth:** `UserProfileMemoryEntry` fields + `memory_temporal.is_memory_entry_active` + lifecycle coordinator.
- **Supersession:** `MemorySupersessionIntent` → plane → `UserProfileManager.apply_memory_supersession_with_lifecycle` — evidence: `test_mem_ent15_core_lifecycle.py`, governance tests.
- **Reconciliation:** `UserProfileMemoryProjection.reconcile` + plane.reconcile — evidence: LTM vector reconcile unit tests, MEM-ENT-15 recovery e2e.
- **Partial failure:** `MemoryControlPartialLifecycleError`, `UserProfileMemoryLifecyclePartialError`, `memory_projection_failure.classify_memory_projection_failure`.

---

## 20. Governance inventory

`MemorySecurityGovernanceService`, `canonical_memory_governance_source_authority`, `memory_specialized_mutation_governance`, `memory_specialized_disclosure_governance`, strategy defaults in `strategies/defaults/memory_security_governance.py`.

---

## 21. Identity

Canonical: `RequestIdentity` via `user_memory_scope`, `verified_request_identity_for_memory_recall`. Tests: `test_mem_ent15_*identity*`, `test_mem_ent15_r2_*`. Org/task scopes use separate IDs — cross-scope isolation tests in MEM-ENT-14.

---

## 22. Strategies

Protocols: extraction, dedup, promotion, ranking, conflict detect/resolve (`strategies/protocols.py`). Defaults: LLM extraction, sequence dedup, enterprise ranking, conservative conflict, accept-all promotion, security policies. **Hard rule evidence:** `test_mem_ent11_platform_boundaries`, consolidation tests assert no direct `add_memory_entry` from consolidation service source.

---

## 23. Resolver / plugin architecture

Flow: `discovery.discover_classified_memory_store_plugins` → `classifier.classify_memory_store_plugin` → `resolver.materialize_*` with `MemoryStoreMaterializationContext`. Kinds: USER_PROFILE, SESSION_STORAGE, SESSION_TURN_INDEX, ENTITY_TEMPORAL, PROCEDURAL, LONG_HORIZON. Entry-point group: **`intergrax.memory_stores`** (declared in test fixture `tests/fixtures/plugin_packages/memory_store_plugin/pyproject.toml`; not in root product pyproject).

---

## 24. Provider matrix

| Capability | Provider contract | Available providers | Default | Real backend | Qualification | Production qualified |
| ---------- | ----------------- | ------------------- | ------- | ------------ | ------------- | -------------------- |
| User profile | UserProfileStorePlugin | in_memory, sqlite, document_store, external fixture | host-chosen | SQLite ref | MEM-ENT-13 + durable harness | SQLite ref only |
| Entity temporal | EntityTemporalMemoryStorePlugin | in_memory plugin | host | none shipped | MEM-ENT-13 checks | no |
| Procedural | ProceduralMemoryStorePlugin | in_memory plugin | host | none shipped | MEM-ENT-13 | no |
| Long horizon | LongHorizonMemoryStorePlugin | in_memory plugin | host | none shipped | MEM-ENT-13 | no |
| Session turn index | SessionTurnIndexStorePlugin | in_memory, vector service, fixture plugin | host | vector ports | MEM-ENT-13 | **no E2E** |
| Session storage | SessionStoragePlugin | EP classifier | host | separate from STI | integration | audit gap |
| Conversational | ConversationalMemoryStore | in_memory, sqlite | session | SQLite unit | not unified qual | no |

---

## 25. Real-vendor maturity matrix

| Vendor / backend | Memory capability | Level | Evidence | Gap |
| ---------------- | ----------------- | ----- | -------- | --- |
| SQLite (relational_store) | UserProfile (+ org profile runtime) | V4–V5 | `test_sqlite_user_profile_store`, MEM-ENT-15 restart e2e | not all specialized stores |
| In-memory reference | all store contracts | V2 | unit + provider qual runner | not production |
| Document store adapter | UserProfile | V2 | unit | no E2E durable |
| Qdrant/pgvector/Chroma | SessionTurnIndex / LTM vector | V1–V2 | vector ports + app wiring tests | **not Memory E2E certified**; RAG paths separate |
| MongoDB | Memory store | V0 | integration exists elsewhere | **no Memory Mongo adapter** |
| PostgreSQL | Memory | V0 | RFC spike only | not implemented |
| External EP fixture | UserProfile | V3 | plugin e2e replaceability | qual only |

---

## 26. Tests evidence map

| Capability | Unit | Integration | E2E | Failure | Security | Vendor |
| ---------- | ---- | ----------- | --- | ------- | -------- | ------ |
| Control plane | MEM-ENT-3/3r/10 | — | MEM-ENT-15 suite | recovery e2e | governance tests | plugin e2e |
| UserProfile | extensive | document/sqlite | restart/core lifecycle | MEM-ENT-14/15 | isolation | durable qual |
| Entity | MEM-ENT-7 | — | partial advanced | resilience pkg | ENT-10c | qual |
| Procedural | MEM-ENT-8 | — | — | — | mutation gov | qual |
| Long horizon | MEM-ENT-9 | — | — | — | disclosure | qual |
| SessionTurnIndex | tenant scope, ENT-13b | app vector wiring | **missing dedicated** | — | qual | qual ref only |
| Strategies | MEM-ENT-4/6 | — | — | — | — | — |
| Observability | MEM-ENT-12/12a | — | — | — | — | — |
| Contract boundary | AST + import tests | — | — | — | — | — |

**Unit test files:** 55 under `tests/unit/memory/`. **Integration e2e:** 8 modules under `tests/integration/memory/e2e/`.

---

## 27. Evals

No dedicated Memory behavioral eval framework (precision/recall quality scenarios). **`MEMORY_BEHAVIORAL_EVAL_GAP`** — closest: `test_mem_ent13_behavioral_coverage.py` (qualification guards, not product eval).

Future categories: preference persistence, supersession, stale suppression, cross-tenant isolation, vendor degradation, session retrieval quality, compaction fidelity.

---

## 28. Concurrency (per store — documented/tested)

| Store | Thread | Async | Process | TX/CAS | Evidence |
| ----- | ------ | ----- | ------- | ------ | -------- |
| InMemory* | audit gap | partial | no | no | MEM-ENT-14 concurrency tests (selected) |
| SQLite UserProfile | sync sqlite | async wrapper | file lock OS | SQL TX | sqlite tests + restart |
| Vector STI | port-defined | async service | audit gap | audit gap | partial unit |

---

## 29. Retry / idempotency

Provider qual checks: idempotent delete/save (user profile), session turn index qual suite. Plane-level partial lifecycle documented; full idempotency matrix **audit gap** for specialized stores.

---

## 30. Observability

MemoryDiagnosticEmitter → MemoryObservabilitySink; terminal events for governance, procedural, long-horizon compaction, projections. Tests MEM-ENT-12. Diagnostics typed; correlation via identity/session in emitters — full vendor-neutral E2E **P2**.

---

## 31. Layer boundaries

- **memory → agents/applications:** none found — **PASS**
- **applications → memory:** allowed — composition roots
- **runtime → memory:** allowed
- **rag → memory:** no direct imports found — LTM via CE/plane

---

## 32. Vendor leakage (memory core)

Grep `intergrax/memory/**/*.py` for pymongo/qdrant/psycopg/chromadb/redis: **none** — **PASS** (adapters live in integrations/applications/vector ports).

---

## 33. Reflection / dynamic API

Public contracts use `@runtime_checkable` Protocols and dataclasses; resolver uses `isinstance` plugin checks. Serialization uses `Dict[str, Any]` in profile JSON — **classified acceptable at persistence boundary**; governance contracts strongly typed.

---

## 34. Documentation inventory

| Document | Class |
| -------- | ----- |
| `docs/project/architecture/MEMORY.md` | canonical hub |
| `docs/project/architecture/MEMORY_ARCHITECTURE.md` | canonical architecture |
| `docs/project/maintainers/plans/MEMORY.md` | plan |
| `MEMORY_STORE_PLUGIN_AUTHOR_GUIDE.md` | maintainer |
| `MEMORY_PROVIDER_EXTENSION_GUIDE.md` | canonical extension |
| `MEMORY_PROJECTION_EXTENSION_GUIDE.md` | canonical extension |
| `INTEGRAX_MEMORY_CONTRACT_BOUNDARY...md` | qualification/historical remediation |
| `assets/memory-platform-position-*.svg` | visual canonical |

---

## 35. Documentation truth gaps

| Claim area | Issue | Severity |
| ---------- | ----- | -------- |
| MEM-ENT “closeout” tone vs final enterprise program | architecture describes delivery; final audit not cert | P3 |
| Episodic “minimal” | accurately marked placeholder | OK |
| SessionTurnIndex | docs imply wiring; E2E vendor proof incomplete | P1 |
| Org memory | taxonomy without full plane integration | P2 |

---

## 36–37. Visual inventory & gaps

| Concern | Diagram | Accurate | Missing |
| ------- | ------- | -------- | ------- |
| System overview | mermaid in MEMORY_ARCHITECTURE | largely yes | recall/supersession sequence |
| Layer boundaries | yes | yes | deployment topology |
| Provider pipeline | partial text | partial | real vendor topology |
| SessionTurnIndex | minimal | gap | dedicated diagram |
| Security/trust | partial in arch | partial | standalone sequence |

---

## 38. Certification terminology audit

Questionable if read as **full enterprise cert today:** “enterprise architecture”, “MEM-ENT closeout”, “qualified” on sub-surfaces. Adequate when scoped to **MEM-ENT-15 e2e** or **provider qualification runner** — not whole Memory layer V7.

---

## 39. Enterprise certification matrix (per surface)

| Surface | Arch | Contracts | Plugin | Security | Recovery | E2E | Real vendor | Docs | Visuals | Verdict |
| ------- | ---- | --------- | ------ | -------- | -------- | --- | ----------- | ---- | ------- | ------- |
| Control plane | pass | pass | partial | pass | partial | pass ref | gap | pass | partial | ENTERPRISE-READY NOT FULLY PROVEN |
| UserProfile | pass | pass | pass | pass | pass ref | pass | SQLite only | pass | pass | ENTERPRISE-READY NOT FULLY PROVEN |
| Entity temporal | pass | pass | pass | partial | gap | partial | gap | pass | partial | GAP |
| Procedural | pass | pass | pass | partial | gap | partial | gap | pass | gap | GAP |
| Long horizon | pass | pass | pass | partial | gap | partial | gap | pass | gap | GAP |
| SessionTurnIndex | pass | pass | pass | partial | gap | **gap** | gap | partial | gap | GAP |
| Episodic | partial | placeholder | inject | n/a | n/a | gap | gap | ok | gap | GAP |
| Task memory | split | partial | partial | partial | partial | partial | sqlite partial | partial | gap | PARTIAL |
| Org memory | split | partial | partial | partial | gap | gap | sqlite org | gap | gap | PARTIAL |
| Conversational | legacy | partial | yes | n/a | sqlite | unit | sqlite | partial | no | LEGACY |
| Governance | pass | pass | strategies | pass | partial | yes | n/a | pass | partial | ENTERPRISE-READY NOT FULLY PROVEN |

---

## 40–43. Gaps

**P0:** NONE identified (no memory→applications import; no unguarded canonical bypass in production consolidation path).

**P1 (blocks final enterprise certification):**

1. SessionTurnIndex — no real-vendor Memory E2E + restart/failure qualification at V5+.
2. Specialized stores (entity, procedural, long-horizon) — no production durable vendor proof beyond in-memory qualification.
3. Organization memory — not integrated on MemoryControlPlane; enterprise scope incomplete.
4. **`MEMORY_BEHAVIORAL_EVAL_GAP`** — no eval framework.
5. Episodic capability — placeholder only for enterprise episodic recall.

**P2:** Conversational parallel session path; task memory host wiring inconsistency; LTM vector vs RAG evidence separation; concurrency semantics not documented per all providers; Mongo/Postgres Memory adapters absent.

**P3:** `memory/__init__.py` narrow exports; maintainer vs public doc drift; workspace_index_spike / postgres RFC clutter.

---

## 44–45. Confirmed vs unproven

**Confirmed enterprise-oriented surfaces (not fully certified):** MemoryControlPlane contract, UserProfile canonical stack, governance service, resolver/plugin typing, MEM-ENT-15 SQLite restart e2e, provider qualification framework.

**Unproven / gap:** SessionTurnIndex production E2E, specialized durable vendors, episodic completeness, behavioral evals, org-on-plane, full observability E2E.

---

## 46. Required next tasks (roadmap alignment)

| Step | Task |
| ---- | ---- |
| 2 | MEM-FINAL-AUDIT-2 — contract-first & layer audit |
| 3 | MEM-FINAL-AUDIT-3 — security, lifecycle, resilience |
| 4 | MEM-FINAL-AUDIT-4 — provider/vendor matrix hardening |
| 5 | MEM-FINAL-AUDIT-5 — real-vendor Memory E2E (STI + specialized) |
| 6 | MEM-FINAL-AUDIT-6 — behavioral evals |
| 7 | MEM-FINAL-AUDIT-7 — docs & diagrams certification |
| 8 | MEM-FINAL-AUDIT-8 — final certification gate |

---

## 47. Overall verdict

**PASS — MEMORY FINAL AUDIT INVENTORY COMPLETE**

(Inventory and classification complete at stated HEAD; **not** `MEMORY ENTERPRISE CERTIFIED`.)

---

## 48. Commit

Recorded when committed on `development` (see git log after commit).

---

# MEM-FINAL-AUDIT-2 — Enterprise Contract & Layer Audit

**Stage:** contract-first architecture · layer boundaries · runtime call-path proof (not security/lifecycle cert, not real-vendor E2E).  
**Baseline AUDIT-1 commit (ancestor required):** `b4ee4ef6e6081a0a4d006e8704b11575d378e31f`  
**AUDIT-2 execution HEAD (before commit):** `a177bb2e825ddd5b68bd434ef044e1b340bcaf93` · branch `development` · clean working tree.

---

## AUDIT-2 — Repo state (execution)

| Field | Value |
| ----- | ----- |
| HEAD (before) | `a177bb2e825ddd5b68bd434ef044e1b340bcaf93` |
| Branch | `development` |
| AUDIT-1 ancestor | YES (`merge-base --is-ancestor b4ee4ef… HEAD`) |
| Working tree | clean |
| Foreign WIP | none |
| Conflicts | none |

**Scope modules:** inventory M-001…M-070 unchanged except corrections below; deep read on `intergrax/memory/contracts/**`, `default_memory_control_plane.py`, `user_profile_manager.py`, `user_profile_memory_lifecycle.py`, `resolver/**`, `provider_qualification/**`, `applications/_shared/memory*_wiring.py`, `runtime/nexus/context/memory_context_invocation.py`, `runtime/user_profile/session_memory_consolidation_service.py`, `runtime/task_memory/**`, `runtime/organization/**`, `tools/providers/memory/**`, `tools/providers/ltm/**`, guard tests MEM-ENT-11/13 + contract boundary AST.

---

## AUDIT-2 — Enterprise binary gate matrix (runtime-reachable surfaces)

Legend: all gates must be YES except **Core vendor dispatch**, **Runtime bypass**, **Private cross-layer internals**, **Hard invariant → arbitrary plugin** (those must be NO).

| Surface | Contract | Consumer typed | Concrete only in composition | Replace w/o core edit | External via contract | Core vendor dispatch | Layer OK | Runtime bypass | Cross-layer `._` | Invariant/strategy confusion |
| ------- | -------- | -------------- | ------------------------------ | --------------------- | --------------------- | -------------------- | -------- | -------------- | ---------------- | ---------------------------- |
| MemoryControlPlane | YES | YES | YES | YES | YES | NO | YES | NO | NO | NO |
| DefaultMemoryControlPlane | YES (impl of Protocol) | composition returns Protocol | YES — ctor injects capabilities only | YES | YES | NO | YES | NO | NO | NO |
| UserProfile stack | YES | YES (`UserProfileStore`, capability) | YES stores in wiring/resolver | YES (EP + plugin tests) | YES (fixture EP) | NO | YES* | NO canonical mutation | NO | NO |
| Projections / lifecycle | YES | YES coordinator → `UserProfileMemoryProjection` | PARTIAL — default LTM vector projection built inside `UserProfileManager` when RAG deps set | PARTIAL | YES custom projection tests | NO | YES | NO | NO | NO |
| Entity temporal | YES | YES service/indexer/store Protocols | YES | YES (`_FakeEntityTemporalMemoryStorePlugin`) | YES | NO | YES | NO | NO | NO |
| Procedural / long-horizon | YES | YES | YES | YES (fake plugins ENT-8/9) | YES | NO | YES | NO | NO | NO |
| SessionTurnIndex | YES | YES (store Protocol) | YES vector adapter + EP | YES | YES qual + plugins | NO in memory core | YES | NO STI→qdrant in core | NO | NO |
| Task memory | YES (`TaskMemoryPersistence` + plane TASK scope) | YES coordinator/view | YES (`store.py` opener) | YES | host inject | NO in memory core | YES | NO — parallel **scope** not canonical user bypass | NO | NO |
| Organization memory | YES (`OrganizationProfileStore`) | YES manager→store | YES wiring | YES | partial | NO | YES | NO — not on plane by design | NO | NO |
| Conversational | YES (legacy store Protocol) | session/chat | YES | YES | YES | NO | YES | NO dual canonical LTM authority | NO | NO |
| CE recall | YES | `MemoryControlPlane` only | YES | YES | YES | NO | YES | NO | NO | NO |
| Tools LTM | YES | plane required fail-closed | YES | YES | YES | NO | YES | NO manager fallback | NO | NO |
| Tools task KV | YES | `TaskMemoryViewBinding` | YES | YES | YES | NO | YES | LEGAL parallel TASK domain | NO | NO |
| Session consolidation | YES | `plane.remember` | YES | YES | YES | NO | YES | NO | NO | NO |
| Resolver / EP | YES | typed classification | YES | YES plugin e2e | YES | NO | PARTIAL† | NO | NO | NO |
| Provider qualification | YES | checks on store Protocol | YES | YES | YES | NO in runner core | YES | NO | NO | NO |
| Recall strategies | YES | plane + pipeline | YES | YES MEM-ENT-6 | YES | NO | YES | NO | NO | NO |
| Governance | YES | plane before capability | YES | YES custom strategy tests | YES | NO | YES | NO | NO | NO |

\*Single **AUDIT-2 layer note:** `intergrax/memory/resolver/materialization.py` imports `ApplicationEnvironmentProfile` from `intergrax.applications.contracts` — memory package → applications tier (see layer matrix). Not a runtime bypass.  
†Resolver context type couples memory resolver to Tier-3 profile dataclass; replaceability of stores unaffected.

**AUDIT-2 CORRECTION (M-002):** DefaultMemoryControlPlane — binary gates **PASS** for contract/layer audit; “PARTIAL” in M-002 referred to certification evidence, not missing Protocol boundary.

---

## AUDIT-2 — Contract matrix (summary)

| Surface | Contract | Consumer typed to contract | Implementation injected | External replaceable | Verdict |
| ------- | -------- | -------------------------- | ----------------------- | -------------------- | ------- |
| Control plane | `MemoryControlPlane` | CE, tools LTM, consolidation | `build_default_memory_control_plane` | custom plane stub tests | PASS |
| UserProfile | `UserProfileStore`, `UserProfileMemoryCapability` | Manager, plane adapter | `memory_wiring`, resolver | SQLite + EP fixture | PASS |
| Entity / procedural / LH | store + capability Protocols | services, wiring | resolver + `_shared/*_wiring` | fake store plugins | PASS |
| SessionTurnIndex | `SessionTurnIndexStore` | vector service, EP | composition / RAG ports | plugin + in-memory | PASS (E2E vendor = AUDIT-5) |
| Task | `TaskMemoryPersistence`, `TaskMemoryCapability` | coordinator, plane TASK ops | `open_task_memory_store` | in-memory inject | PASS parallel domain |
| Org | `OrganizationProfileStore` | `OrganizationProfileManager` | `memory_wiring` | sqlite/in-mem | PASS parallel domain |
| Conversational | `ConversationalMemoryStore` | session paths | session wiring | sqlite/in-mem | LEGACY isolated |

---

## AUDIT-2 — Layer dependency matrix (real imports)

| From | To | Dependency | Legal | Evidence |
| ---- | -- | ---------- | ----- | -------- |
| `intergrax/memory/contracts` | stdlib, `intergrax.contracts` | types only | YES | AST guard `test_memory_contract_boundary` |
| `intergrax/memory/*` services | `intergrax/memory/stores/*` | **no** direct store imports in `*service.py` | YES | grep: 0 matches |
| `intergrax/memory` core | vendor SDKs | none | YES | grep pymongo/qdrant/psycopg/redis/chromadb/pinecone in `intergrax/memory/**`: **0**; `test_mem_ent13_guards`, ENT-8/9/12 vendor tests |
| `intergrax/memory/resolver/materialization.py` | `intergrax.applications.contracts.environment_profile` | materialization context | **NO (tier-up)** | single import — P1 layer debt |
| `intergrax/runtime/*` consumers | `intergrax.memory.contracts` | plane/capabilities | YES | CE, consolidation |
| `intergrax/tools/providers/ltm` | `MemoryControlPlane` | recall/remember | YES | no `UserProfileManager` |
| `applications/_shared/memory_wiring.py` | concrete stores + resolver | composition root | YES | legal boundary |
| `integrations/providers/*` | memory store Protocols | adapters | YES | sqlite task/user paths |

**Forbidden dependency count (runtime memory core):** memory→applications **1 module** (materialization context only); memory→vendor SDK **0**; contracts→implementation **0** (AST enforced).

---

## AUDIT-2 — Concrete implementation import matrix (selected)

| Consumer layer | Concrete import | Legal? |
| -------------- | --------------- | ------ |
| memory core services | none of SQLite/Qdrant/Mongo stores | YES |
| `default_memory_control_plane.py` | none — only capabilities | YES |
| `user_profile_manager.py` | `UserProfileLtmVectorProjection` default factory | **P1** — composition should own default projection wiring |
| `applications/_shared/memory_wiring.py` | InMemory/SQLite/org/session stores | YES |
| `runtime/task_memory/store.py` | `SQLiteTaskMemoryStore` | YES (runtime composition opener) |
| `runtime/nexus/session/session_manager.py` | `UserProfileManager` (typed service) | YES internal recall helper |
| tests / qual harness | in-memory + sqlite fixtures | TEST_ONLY / qual |

---

## AUDIT-2 — MemoryControlPlane boundary

**Runtime callers (production paths):** `memory_context_invocation` (`isinstance(plane, MemoryControlPlane)`), `session_memory_consolidation_service` (typed ctor), `tools/providers/ltm/service.py` (fail-closed if missing), `tools/providers/memory/service.py` semantic LTM branch (`plane.recall`).

**DefaultMemoryControlPlane** appears only in composition (`memory_control_wiring.build_default_memory_control_plane`) and tests — **PASS**.

**Constructor proof:** `DefaultMemoryControlPlane` fields are Protocol-typed capabilities + governance/emitter/strategies — **no store construction** (lines 378–388 `default_memory_control_plane.py`).

---

## AUDIT-2 — UserProfile / projection / entity / procedural / LH / STI / task / org (proof pointers)

| Area | Proof |
| ---- | ----- |
| UserProfileManager → store | `__init__(store: UserProfileStore)` — never SQLite type |
| Plane → capability | `UserProfileManagerMemoryCapability` adapter; plane calls capability methods |
| Lifecycle → projection | `UserProfileMemoryLifecycleCoordinator(projections: Sequence[UserProfileMemoryProjection])` |
| Entity indexer → store | `EntityTemporalMemoryService` + `DefaultEntityMemoryIndexer` use store Protocol |
| Procedural / LH | `ProceduralMemoryService` / `LongHorizonMemoryService` store-injected |
| STI | `SessionTurnIndexStore` Protocol; vector backend via ports (`session_turn_index_service.py`), not qdrant import in memory core |
| Task | host → `TaskMemoryCoordinator` → `TaskMemoryPersistence`; plane `task_memory: TaskMemoryCapability` for remember/delete TASK scope |
| Org | `OrganizationProfileManager(OrganizationProfileStore)` — **verdict: PARALLEL LEGITIMATE DOMAIN** (org semantic authority separate from user canonical plane; not MEMORY AUTHORITY FRAGMENTATION for user facts) |
| Conversational | session transcript store — **LEGACY_COMPAT**; not authoritative user LTM |

---

## AUDIT-2 — CE integration

`memory_context_invocation.py`: resolves plane from wiring extras as `MemoryControlPlane`; recall via `plane.recall` + identity spine — **no concrete memory implementation imports**.

---

## AUDIT-2 — Tools

- **LTM:** `_require_memory_control_context` — denies without plane; uses `plane.recall` / remember — **PASS** (`test_ltm_trusted_identity` asserts manager not called).
- **Task KV:** `memory_view` binding → `TaskMemoryCoordinator` path — parallel TASK scope, not user canonical mutation bypass.

---

## AUDIT-2 — Session consolidation

Source inspection + test: `SessionMemoryConsolidationService.consolidate_session` uses `self._memory_control_plane.remember`; AST guard `test_session_consolidation_service_does_not_call_profile_manager_add_memory_entry` — **PASS**.

---

## AUDIT-2 — Resolver / EP / fail-closed

Flow confirmed: `discover_classified_memory_store_plugins` → classifier → `materialize_*` with plugin ids from `MemoryProfile` — **no** `if provider == "x"` in `intergrax/memory/**` core (grep: 0). Missing/ambiguous plugin → explicit errors in wiring (`memory_plugin_bootstrap_errors`, resolver fail-closed tests in ENT-13). Entry-point group **`intergrax.memory_stores`** evidenced by fixture `tests/fixtures/plugin_packages/memory_store_plugin/pyproject.toml`.

**External plugin proof:** `test_mem_ent7/8/9_*_Fake*StorePlugin` — register class → resolve → materialize without core edit.

---

## AUDIT-2 — Provider qualification

Runner + checks operate on store **Protocol** instances; `test_qualification_core_has_no_vendor_imports` — no SQLite/provider_id dispatch in canonical runner — **PASS**.

---

## AUDIT-2 — Hard invariant vs replaceable strategy

| Invariant | Hard owner | Replaceable? | Enforcement |
| --------- | ---------- | ------------ | ----------- |
| Trusted identity / user scope | plane + `RequestIdentity` spine | NO | CE recall, LTM tools, MEM-ENT-15 |
| Governance before mutation | `DefaultMemoryControlPlane` + `MemorySecurityGovernanceService` | policy strategies only inside governance bundle | MEM-ENT-11 remember guard test |
| Canonical user facts | `UserProfileStore` via manager lifecycle | NO | plane remember path |
| Projection ≠ authority | lifecycle coordinator | projection impl replaceable | reconcile tests |
| Lineage / revision | models + store contracts | NO | qual checks |

**Strategy direct store mutations:** **0** in `intergrax/memory/strategies/**` (no store `save`/`put`).

**Projection as canonical authority:** **0** runtime paths treat vector/entity index as sole writer of user profile facts — writes go primary store then projections.

---

## AUDIT-2 — Replaceable mechanism map (minimum)

| Mechanism | Contract | Default | External path | Core edit for swap? |
| --------- | -------- | ------- | ------------- | ------------------- |
| UserProfileStore | `UserProfileStore` / plugin | host sqlite/in-mem | EP `intergrax.memory_stores` | NO |
| EntityTemporalMemoryStore | Protocol + plugin | in-mem ref | EP + fake plugin tests | NO |
| EntityMemoryIndexer | Protocol | `DefaultEntityMemoryIndexer` | inject indexer | NO |
| EntityTemporalMemoryCapability | Protocol | wired service | `_shared/entity_graph_wiring` | NO |
| Projection | `UserProfileMemoryProjection` | LTM vector if RAG deps | inject sequence | NO (but default concrete in manager = wiring smell) |
| Procedural / LH store | Protocol | in-mem | EP | NO |
| SessionTurnIndexStore | Protocol | vector/in-mem | EP + ports | NO |
| Recall strategies | `MemoryRecallStrategySet` | defaults bundle | inject set | NO |
| Observability sink | `MemoryObservabilitySink` | diagnostic emitter | inject | NO |
| Qualification check | store Protocol | harness | custom store instance | NO |

---

## AUDIT-2 — Runtime bypass inventory (re-verified)

| Path | Classification | Severity |
| ---- | -------------- | -------- |
| `SessionMemoryConsolidationService` → plane.remember | LEGAL_INTERNAL | — |
| Tools LTM without plane | fail-closed | — |
| `SessionManager.search_longterm_memory` → `UserProfileManager` | **CANONICAL_BYPASS** (recall skips plane governance/recall pipeline) | **P1** |
| Task tools → TaskMemoryView | PARALLEL_DOMAIN | P2 evidence |
| Org `add_memory_entry` | PARALLEL_DOMAIN | P2 cert gap |
| Conversational stores | LEGACY_COMPAT | P2 |
| Entity graph legacy | LEGACY guarded | P2 |
| Integration test direct manager | TEST_ONLY | P3 |
| `UserProfileManager.add_memory_entry` via plane capability | LEGAL_INTERNAL (owner) | — |

---

## AUDIT-2 — Raw public API / reflection / private access

| Location | Finding | Classification | Severity |
| -------- | ------- | -------------- | -------- |
| `contracts/memory_store_plugin.py` | `**kwargs: Any` factory hooks | opaque vendor payload | P2 |
| `contracts/memory_models.py` | `metadata: Dict[str, Any]` | serialization boundary | P2 |
| `memory_context_invocation.py` | `dict[str, Any]` engine payload | serialization boundary | P3 |
| Qualification runner | AST forbids getattr/setattr in qual core | guarded | OK |
| Resolver | no getattr in resolver package | OK | — |

Cross-owner `._store._conn` in runtime memory paths: **none flagged** in audited production modules (spot checks; full `_` scan deferred to AUDIT-3).

---

## AUDIT-2 — Composition proof flows (one each)

| Flow | config → instance |
| ---- | ----------------- |
| UserProfile | `MemoryProfile.user_profile_store_plugin_id` / sqlite baseline → `resolve_memory_platform_wiring` → `UserProfileStore` → `UserProfileManager` → plane |
| Entity | `enable_*` + resolver → `EntityTemporalMemoryStore` → service capability |
| Procedural / LH | plugin id → materialize → service |
| STI | host vector wiring + `SessionTurnIndexStore` |
| Task | `open_task_memory_store` / inject → coordinator → optional plane `TaskMemoryCapability` |
| Org | sqlite/in-mem org store in `memory_wiring` → `OrganizationProfileManager` |

---

## AUDIT-2 — Replaceability test evidence

- Fake store plugins: ENT-7/8/9 unit tests  
- Custom plane: `MemoryControlPlaneTestStub`, `RecordingMemoryControlPlane` (MEM-ENT-11, MEM-XINT-3)  
- Custom projection/indexer: `RecordingEntityMemoryIndexer` + protocol isinstance test  
- Plugin EP: integration memory plugin e2e (AUDIT-1 map)

**Bounded static checks run:** `uv run pytest tests/unit/memory/test_memory_contract_boundary.py tests/unit/memory/test_mem_ent13_guards.py tests/unit/memory/test_mem_ent11_platform_boundaries.py -q` → **26 passed**.

---

## AUDIT-2 — Architecture guard coverage

Existing: `test_memory_contract_boundary` (contracts AST), `test_mem_ent13_guards` (qual vendor/reflection), `test_mem_ent11_platform_boundaries` (consolidation/plane/governance), ENT-8/9/12 vendor import tests, `test_mem_ent15_r2_identity_guards` (consolidation source), MEM-XINT runtime recall plane test.

**New guards added in AUDIT-2:** none (code already guarded; gaps are architectural debt not fixable in audit-only task).

---

## AUDIT-2 — P0 / P1 / P2 / P3

**P0:** **NONE** (no ungoverned canonical mutation bypass; no core vendor SDK; no projection writes canonical truth; no strategy store mutation).

**P1:**

1. **Runtime LTM recall bypass:** `SessionManager.search_longterm_memory` calls `UserProfileManager.search_longterm_memory` without `MemoryControlPlane.recall` (governance/recall strategy envelope skipped on that path).
2. **Core default concrete projection:** `UserProfileManager._resolve_memory_projections` instantiates `UserProfileLtmVectorProjection` inside memory core when RAG deps present — should be composition-only for strict enterprise replaceability.
3. **Layer coupling:** `memory/resolver/materialization.py` → `applications.contracts.environment_profile` (memory must not depend on applications tier).

**P2:** Org/task parallel domains certification gaps; SessionTurnIndex real-vendor E2E; conversational legacy coexistence; public `Any` on plugin factories; incomplete composition proof for all hosts; fail-closed downgrade paths in dev profiles (documented in wiring comments).

**P3:** `memory/__init__.py` exports; maintainer doc drift; test-only manager shortcuts.

---

## AUDIT-2 — Changes to AUDIT-1 classifications

| ID | Change | Reason |
| -- | ------ | ------ |
| M-002 | AUDIT-2 CORRECTION: contract/layer **PASS**; keep “not fully proven” for cert | Binary gate matrix |
| M-071 (implicit) | SessionManager LTM search | New bypass row — recall not plane-routed |

---

## AUDIT-2 — Overall verdict

**PASS WITH CORRECTIONS — MEMORY CONTRACT/LAYER AUDIT NOT CLOSED**

Contract-first architecture and replaceability **hold** for canonical mutation, composition, plugins, and CE/tools LTM paths. Closure blocked by **P1** items: session-manager recall bypass, in-core default LTM projection materialization, and memory→applications import in resolver materialization context. No **MEM-FINAL-AUDIT-2 — ARCHITECTURE IMPLEMENTATION GAP** (fixes are wiring/routing refactors, not ownership model change).

---

## AUDIT-2 — Commit

| Field | Value |
| ----- | ----- |
| SHA | `56efcc5fc74492214d2b69e1a7910693edbd5afd` |
| Message | `docs(memory): audit enterprise contracts and layer boundaries` |
| Files | `docs/project/maintainers/qualification/MEMORY_FINAL_ENTERPRISE_AUDIT.md` |
| HEAD after | same commit on `development` |

---

> Wynik MEM-FINAL-AUDIT-1 musi zostać niezależnie zweryfikowany na podstawie kodu z GitHuba przed MEM-FINAL-AUDIT-2 (baseline `b4ee4ef6…`).  
> Wynik MEM-FINAL-AUDIT-2 musi zostać niezależnie zaudytowany na podstawie exact SHA z GitHuba przed rozpoczęciem MEM-FINAL-AUDIT-3.
