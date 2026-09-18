# Memory — Provider & Vendor Qualification Matrix (MEM-FINAL-AUDIT-4)

**Baseline:** MEM-FINAL-AUDIT-3 closed at `a8d750b0bd48c902e10d487cf201aa1e762b01de`  
**Audit execution HEAD:** `0ba1a514c60af1c319bd35fc714268de9640bcbd` · branch `development`  
**Ledger:** [`MEMORY_FINAL_ENTERPRISE_AUDIT.md`](MEMORY_FINAL_ENTERPRISE_AUDIT.md) (authoritative certification narrative)

---

## Qualification ladder (semantic map)

Official runtime gate: `MemoryProviderQualificationStatus` (`qualified` / `not_qualified` / `not_supported` / `blocked`) in `intergrax/memory/contracts/provider_qualification.py` + `MemoryProviderQualificationRunner`.

Enterprise ladder **V0–V8** (audit vocabulary — map to official status + evidence, not a second gate):

| Level | Meaning | Typical evidence |
| ----- | ------- | ---------------- |
| **V0** | Declared / RFC / planned only | Spike modules, docs without impl |
| **V1** | Implementation exists | Python class, wiring reachable |
| **V2** | Contract / unit qualified | `MemoryProviderQualificationRunner` → `QUALIFIED` on reference/in-proc store |
| **V3** | Plugin / resolver / materialization qualified | EP discovery → `materialize_*` → `isinstance` contract check |
| **V4** | Persistence / reference integration | File-backed SQLite, DocumentStore port binding |
| **V5** | Durable restart / recovery | `run_durable_user_profile_production_qualification` reopen/delete; MEM-ENT-15 SQLite composition restart |
| **V6** | Real-vendor E2E | Live service/container; **not inferred** from RAG vector adapters |
| **V7** | Failure / degradation / recovery vs vendor | Injected vendor faults + recovery proof |
| **V8** | Production-certified | Durable + vendor + production profile wiring + qual gate (none Memory-wide today) |

**Verdict labels (per capability):** `REFERENCE ONLY` · `CONTRACT QUALIFIED` · `PLUGIN QUALIFIED` · `DURABILITY QUALIFIED` · `RESTART QUALIFIED` · `REAL-VENDOR QUALIFIED` · `PRODUCTION QUALIFIED` · `NOT QUALIFIED` · `PLANNED ONLY` · `LEGACY` · `IMPLEMENTED BUT NOT RUNTIME-WIRED`

---

## Capability inventory (runtime-reachable surfaces)

| Capability | Contract | Primary composition |
| ---------- | -------- | ------------------- |
| UserProfile canonical store | `UserProfileStore` | `memory_wiring`, plane → `UserProfileManager` |
| UserProfile projection | `UserProfileMemoryProjection` | `UserProfileLtmVectorProjection` (LTM vector) |
| Entity temporal store | `EntityTemporalMemoryStore` | `entity_graph_wiring` + plugins |
| Entity indexer | `EntityMemoryIndexer` | `DefaultEntityMemoryIndexer` (derived) |
| Procedural store | `ProcedureMemoryStore` | `procedural_memory_wiring` + plugins |
| Long-horizon store | `LongHorizonMemoryStore` | `long_horizon_memory_wiring` + plugins |
| SessionTurnIndex | `SessionTurnIndexStore` | `build_session_turn_index_store` / vector adapter |
| Task memory persistence | `TaskMemoryPersistence` | `task_memory_wiring` (runtime, parallel scope) |
| Organization profile store | `OrganizationProfileStore` | SQLite bundle / in-memory (runtime/org) |
| Conversational legacy store | `ConversationalMemoryStore` | SessionManager transcript path |
| Observability sink | `MemoryObservabilitySink` | `memory_observability_wiring` |
| Memory plugin resolver | EP `intergrax.memory_stores` | `resolver/*`, overlay in `memory_wiring` |

---

## Required summary matrix

| Capability | Reference provider | Durable provider | Real vendor proof | Production qualified |
| ---------- | ------------------ | ---------------- | ----------------- | -------------------- |
| UserProfile store | `InMemoryUserProfileStore` | `SQLiteUserProfileStore` (sqlite integration) | **NO** (local file only) | **CONDITIONALLY** — sqlite lab/product harness; product preset gap (see P1) |
| UserProfile projection | `UserProfileLtmVectorProjection` | via RAG vector backend | **NO** Memory E2E | **NOT QUALIFIED** vendor |
| Entity temporal | `intergrax.in_memory_entity_temporal` | **NONE** in repo | **NO** | **NOT QUALIFIED** durable |
| Entity indexer | `DefaultEntityMemoryIndexer` | derived | n/a | **CONTRACT QUALIFIED** (service) |
| Procedural | `intergrax.in_memory_procedural` | **NONE** | **NO** | **NOT QUALIFIED** durable |
| Long-horizon | `intergrax.in_memory_long_horizon` | **NONE** | **NO** | **NOT QUALIFIED** durable |
| SessionTurnIndex | `InMemorySessionTurnIndexStore` | `VectorSessionTurnIndexStore` + Qdrant/pgvector backing | **YES (Qdrant + pgvector STI)** | **V6 Qdrant + pgvector reconnect qualified**; Chroma open |
| Task memory | `InMemoryTaskMemoryStore` | `SQLiteTaskMemoryStore` | **NO** | **CONDITIONALLY** — env/db path |
| Organization profile | `InMemoryOrganizationProfileStore` | `SQLiteOrganizationProfileStore` | **NO** | **CONDITIONALLY** — sqlite bundle only |
| Conversational | `InMemoryConversationalMemoryStore` | `SQLiteConversationalMemoryStore` | **NO** | **LEGACY** — not canonical plane |
| Observability | `NoOpMemoryObservabilitySink` | n/a | n/a | **REFERENCE** / test recording sink |
| Plugin resolver | built-in classifier | n/a | fixture EP only | **PLUGIN QUALIFIED** (contract) |

---

## Master matrix (compact)

| Capability | Provider | Type | Contract | Plugin | Durability | Restart | Failure | Real vendor | Production path | Level | Verdict |
| ---------- | -------- | ---- | -------- | ------ | ---------- | ------- | ------- | ----------- | --------------- | ----- | ------- |
| UserProfile | `InMemoryUserProfileStore` | in-proc | `UserProfileStore` | optional EP | NO | NO | partial unit | NO | lab/test default when no sqlite/mongo | V2 | REFERENCE ONLY / CONTRACT QUALIFIED |
| UserProfile | `SQLiteUserProfileStore` | sqlite file | `UserProfileStore` | integration bundle | YES | YES | unit | NO (local) | lab + sqlite `IntegrationProfile` | V5 | DURABILITY + RESTART QUALIFIED |
| UserProfile | `DocumentStoreUserProfileStore` | adapter | `UserProfileStore` | via mongo integration | backend-dependent | adapter-only test | unit | NO | mongo document_store path | V2–V3 | CONTRACT QUALIFIED; backend NOT vendor-qualified |
| UserProfile | `external.in_memory_user_profile` | fixture EP | `UserProfileStore` | YES | NO | NO | qual | NO | test/strict overlay only | V3 | PLUGIN QUALIFIED (replaceability) |
| Projection LTM | `UserProfileLtmVectorProjection` | derived | `UserProfileMemoryProjection` | NO | vector host | NO | reconcile tests | NO | when LTM+RAG enabled | V2 | NOT REAL-VENDOR QUALIFIED |
| Entity temporal | `InMemoryEntityTemporalMemoryStore` | in-proc | `EntityTemporalMemoryStore` | `intergrax.in_memory_entity_temporal` | NO | NO | qual runner | NO | default when entity graph on | V2 | REFERENCE ONLY |
| Entity temporal | `EntityGraphMemoryStore` | legacy | legacy graph | NO | optional | NO | guarded tests | NO | legacy apps | V1 | LEGACY |
| Entity indexer | `DefaultEntityMemoryIndexer` | derived | `EntityMemoryIndexer` | inject | NO | n/a | unit | NO | with entity store | V2 | CONTRACT QUALIFIED |
| Procedural | `InMemoryProceduralMemoryStore` | in-proc | `ProcedureMemoryStore` | `intergrax.in_memory_procedural` | NO | NO | qual runner | NO | profile flag | V2 | REFERENCE ONLY |
| Long-horizon | `InMemoryLongHorizonMemoryStore` | in-proc | `LongHorizonMemoryStore` | `intergrax.in_memory_long_horizon` | NO | NO | qual runner | NO | profile flag | V2 | REFERENCE ONLY |
| SessionTurnIndex | `InMemorySessionTurnIndexStore` | in-proc | `SessionTurnIndexStore` | classifiable | NO | NO | qual runner | NO | qual / tests | V2 | REFERENCE ONLY |
| SessionTurnIndex | `VectorSessionTurnIndexStore` | vector adapter | `SessionTurnIndexStore` | STI EP optional | Qdrant + pgvector reconnect proved | client reconnect | unit + 5D/5E E2E | **YES (Qdrant, pgvector)** | `enable_session_vector_index` + RAG | V6 | **REAL-VENDOR RECONNECT QUALIFIED (Qdrant, pgvector)** |
| Task memory | `InMemoryTaskMemoryStore` | in-proc | `TaskMemoryPersistence` | NO | NO | NO | unit | NO | tests | V2 | REFERENCE ONLY |
| Task memory | `SQLiteTaskMemoryStore` | sqlite file | `TaskMemoryPersistence` | sqlite opens | YES | partial integ | unit | NO | env `INTERGRAX_TASK_MEMORY_DB` / lab | V4–V5 | DURABILITY QUALIFIED (platform semantics) |
| Organization | `InMemoryOrganizationProfileStore` | in-proc | `OrganizationProfileStore` | NO | NO | NO | unit | NO | mongo path + org flag | V1 | NOT QUALIFIED durable |
| Organization | `SQLiteOrganizationProfileStore` | sqlite file | `OrganizationProfileStore` | sqlite bundle | YES | integ persist | unit | NO | sqlite integration | V4 | DURABILITY QUALIFIED (no Memory qual runner) |
| Conversational | `InMemoryConversationalMemoryStore` | in-proc | `ConversationalMemoryStore` | NO | NO | NO | unit | NO | session/chat | V2 | LEGACY |
| Conversational | `SQLiteConversationalMemoryStore` | sqlite | `ConversationalMemoryStore` | NO | YES | partial | unit | NO | session | V4 | LEGACY + local durable |
| Observability | `NoOpMemoryObservabilitySink` | noop | `MemoryObservabilitySink` | inject | n/a | n/a | n/a | NO | default | V1 | REFERENCE ONLY |
| Observability | `RecordingMemoryObservabilitySink` | test | `MemoryObservabilitySink` | inject | n/a | n/a | n/a | NO | tests MEM-ENT-12 | V2 | CONTRACT QUALIFIED |
| PostgreSQL Memory | `PostgresMemoryBackendConfig` spike | RFC | n/a | NO | n/a | n/a | n/a | NO | **not wired** | V0 | PLANNED ONLY |
| MongoDB Memory | *(none)* | generic `DocumentStore` only | via adapter | NO | if mongo backend | NO | fake factory tests | NOT_EXECUTED | document_store slug | V0–V2 | **NOT MEMORY-QUALIFIED** as vendor |
| Qdrant (STI) | `VectorSessionTurnIndexStore` + `qdrant` | integration backing | STI ports | NO Memory EP | reconnect proved (5D) | client reconnect | 5D suite | **YES** | vector flags + Qdrant | V6 | **MEMORY STI QUALIFIED (Qdrant)** |
| pgvector (STI) | `VectorSessionTurnIndexStore` + `pgvector` | integration backing | STI ports | NO Memory EP | reconnect proved (5E) | client reconnect | 5E suite | **YES** | vector flags + pgvector | V6 | **MEMORY STI QUALIFIED (pgvector)** |
| Chroma (STI) | same adapter pattern | integration layer | STI ports | NO Memory EP | **NO** | RAG tests only | **NO Memory STI E2E** | vector flags | V1 | **OPEN (5F)** |

---

## Provider IDs (discovery)

| Plugin / provider ID | Kind | Source |
| -------------------- | ---- | ------ |
| `intergrax.in_memory_entity_temporal` | entity_temporal | `in_memory_entity_temporal_memory_plugin.py` |
| `intergrax.in_memory_procedural` | procedural | `in_memory_procedural_memory_plugin.py` |
| `intergrax.in_memory_long_horizon` | long_horizon | `in_memory_long_horizon_memory_plugin.py` |
| `external.in_memory_user_profile` | user_profile_store | test fixture plugin |
| `external.in_memory_session_storage` | session_storage | test fixture plugin |
| EP names `external_user_profile` / `external_session_storage` | entry-point labels | `tests/fixtures/.../pyproject.toml` group `intergrax.memory_stores` |

**Uniqueness:** `test_duplicate_plugin_id_fails_closed` in `tests/unit/memory/test_memory_store_resolver.py` — duplicate IDs fail closed.  
**Unknown ID:** `MemoryStorePluginResolutionError` — `materialize_*` / catalog miss (`resolver.py`).  
**Wrong kind:** explicit kind mismatch error in `_select_classified_plugin`.

Qualification descriptor IDs (harness, not EP): `sqlite.user_profile`, `document_store.user_profile`.

---

## Evidence index (YES paths)

| Proof | Path |
| ----- | ---- |
| Unit contract qual (UserProfile in-mem) | `tests/unit/memory/test_mem_ent13_provider_qualification.py::test_reference_in_memory_user_profile_qualifies` |
| SQLite qual + runner | `test_mem_ent13_provider_qualification.py::test_sqlite_user_profile_qualifies` |
| SQLite durable reopen/delete | `tests/unit/memory/test_mem_ent13c_durable_provider_qualification.py::test_sqlite_durable_production_qualification_reopen_and_delete` |
| SQLite durable reference E2E (5B) | `tests/unit/memory/test_mem_final_audit_5b_sqlite_durable_reference.py` |
| SQLite PRODUCT admission + restart E2E (5B) | `tests/unit/applications/test_mem_final_audit_5b_sqlite_production_restart_e2e.py` |
| Trusted admission evidence bundle | `intergrax/memory/provider_qualification/user_profile_admission_evidence.py` |
| DocumentStore adapter not production durable | `test_mem_ent13c_durable_provider_qualification.py::test_document_store_in_memory_backend_is_not_production_durable` |
| External plugin materialization | `test_mem_ent13c_durable_provider_qualification.py::test_external_plugin_materialization_canonical_qualification` |
| STI reference qual | `tests/unit/memory/test_mem_ent13_behavioral_coverage.py::test_session_turn_index_reference_qualifies` |
| Plugin E2E replaceability | `tests/integration/memory/e2e/test_mem_ent15_plugin_replaceability.py` |
| SQLite composition restart | `tests/integration/memory/e2e/test_mem_ent15_restart.py::test_sqlite_composition_restart_recall_and_continue_mutation` |
| Resolver / duplicate ID | `tests/unit/memory/test_memory_store_resolver.py` |
| Wiring sqlite / in-mem / mongo | `tests/unit/applications/test_memory_wiring.py` |
| Mongo real qual | `test_mem_ent13c_durable_provider_qualification.py::test_real_mongodb_user_profile_qualification_not_certified_without_infra` → **NOT_EXECUTED** |
| SQLite CRUD unit | `tests/unit/memory/test_sqlite_user_profile_store.py` |
| Org sqlite integ | `tests/integration/runtime/organization/test_sqlite_organization_profile_store.py` |

**Fake / in-proc markers:** All `InMemory*` stores, `InMemoryDocumentStore` backend, fixture external plugins — never counted as V6.

---

## Default provider semantics

| Capability | Lab / test default | Production-oriented default | Fallback |
| ---------- | ------------------ | --------------------------- | -------- |
| UserProfile + session | SQLite when `relational_store.slug==sqlite` | **Gap:** `product_defaults` uses PostgreSQL relational preset — **does not** enable Memory sqlite path | InMemory when neither sqlite nor mongo document_store (`memory_wiring.py` priority 3) |
| Entity temporal | `intergrax.in_memory_entity_temporal` if flag on | same unless `entity_temporal_memory_store_plugin_id` set | none (explicit plugin materialize) |
| Procedural / LH | in-memory plugins when flags on | same | none |
| STI | none unless flag | `VectorSessionTurnIndexStore` if RAG stack present | `None` if vector backend missing (`assert_memory_vector_backend_available` fail-closed when flags require backend) |
| Task memory | null unless db env / flag | SQLite when path resolved | `InMemory` only via explicit test injection |

---

## Silent fallback audit

- **No** `except: return InMemory...` in `intergrax/memory/**`.
- **Explicit** non-durable fallback in `_resolve_baseline_memory_platform_wiring` when integration profile lacks sqlite **and** mongodb bindings (`memory_wiring.py` lines 215–223).
- **Vector memory:** fail-closed via `MemoryVectorBackendUnavailableError` when flags require backend but RAG stack incomplete (`memory_vector_wiring.py`).
- **STRICT + broken EP bootstrap:** `enforce_strict_memory_plugin_bootstrap` raises (`memory_wiring.py`).
- **Production silent non-durable for enabled LTM:** **risk** when `product_defaults` integration is PostgreSQL-only (not sqlite) and memory flags enabled — falls through to InMemory without error → **P1** (not hidden exception).

---

## Adapter vs backend (DocumentStore UserProfile)

| Layer | Qualification |
| ----- | ------------- |
| `DocumentStoreUserProfileStore` adapter | V2 canonical runner QUALIFIED |
| `InMemoryDocumentStore` backend | ADAPTER_RECREATION_ONLY — **not** production durable (`test_document_store_in_memory_backend_is_not_production_durable`) |
| Real MongoDB vendor | **NOT_EXECUTED** — `RealExternalProviderQualification.NOT_EXECUTED` |

---

## Vendor status (exact)

| Vendor | Memory-specific adapter | Evidence | Memory verdict |
| ------ | ------------------------ | -------- | -------------- |
| **SQLite** | Yes (UserProfile, Task, Org, Conversational, Session) | MEM-ENT-13C durable harness + MEM-FINAL-AUDIT-5B reference certification | **UserProfile: REFERENCE DURABLE / RESTART QUALIFIED (V5)** — behavioral + durability evidence feed; not external V6 |
| **MongoDB** | UserProfile via DocumentStore only | wiring unit test with factory | **NOT MEMORY-VENDOR QUALIFIED** |
| **PostgreSQL** | RFC spike only | `postgres_memory_backend_rfc.py` | **PLANNED ONLY (V0)** |
| **Qdrant** | `VectorSessionTurnIndexStore` backing via RAG ports | `test_mem_final_audit_5d_qdrant_session_turn_index_real_vendor.py` | **V6 STI reconnect qualified** |
| **pgvector** | same adapter pattern | pgvector STI E2E (5E) | **V6 RECONNECT QUALIFIED** |
| **Chroma** | same | none Memory STI | **< V6 (OPEN)** |

---

## Architecture guards (AUDIT-4 recheck)

| Check | Result |
| ----- | ------ |
| Memory core vendor SDK imports | **0** (`test_mem_ent13_guards.py`) |
| `intergrax/memory` → `applications` imports | **0** (grep) |
| Vendor-specific switch in memory core | **0** (`if provider == "qdrant"` etc.) |

---

## Gap ledger

| ID | Capability | Provider | Missing proof | Sev | Target |
| -- | ---------- | -------- | ------------- | --- | ------ |
| GAP-4-01 | UserProfile | product + PostgreSQL preset | Durable Memory backend wiring / fail-closed | **CLOSED (5A)** | `memory_provider_admission` + `test_mem_audit5a_production_provider_admission.py` |
| GAP-4-02 | SessionTurnIndex | `VectorSessionTurnIndexStore` + Qdrant | write → reconnect → recall | **CLOSED (5D, Qdrant only)** | `test_mem_final_audit_5d_qdrant_session_turn_index_real_vendor.py` |
| GAP-4-03 | Entity / Procedural / LH | in-memory only | durable vendor + restart | P2 | AUDIT-5 |
| GAP-4-04 | UserProfile | Mongo DocumentStore | real-vendor qual execution | **CLOSED (5C)** | `test_mem_final_audit_5c_mongo_user_profile_real_vendor.py` |
| GAP-4-05 | Organization | Mongo path | durable org store (uses InMemory org on mongo LTM path) | P2 | AUDIT-6 |
| GAP-4-06 | Task memory | SQLite | restart/failure vendor suite | P2 | AUDIT-5 |
| GAP-4-07 | Vector backends | Qdrant **CLOSED (5D)**; pgvector **CLOSED (5E)**; Chroma **OPEN** | Memory-scoped STI E2E | P2 | AUDIT-5 |
| GAP-4-08 | PostgreSQL | Memory bundle | implementation | P3 | post-RFC |

**P0:** NONE  
**P1:** NONE after MEM-FINAL-AUDIT-5A-R3 (GAP-5A-03 durability evidence closed)

## MEM-FINAL-AUDIT-5A-R3 — Trusted durability evidence admission

| Check | Result |
| ----- | ------ |
| Durability evidence contract | `intergrax/memory/contracts/provider_durability_evidence.py` |
| Declared durability | `memory_provider_durability` on stores — claim only; **not** admission authority |
| Admission authority | `MemoryProviderDurabilityEvidenceRegistry` + trusted identity binding |
| Behavioral qual alone | Insufficient for PRODUCT persistent (`DURABILITY_EVIDENCE_MISSING`) |
| MEM-ENT-13C reopen proof | `build_user_profile_admission_evidence_from_durable_qualification` + `durability_evidence_from_reopen_proof` (requires `delete_reopen_passed=True`) |
| Attack: RAM + claims DURABLE + behavioral QUALIFIED | Fail without platform durability evidence |

| Gap | Status |
| --- | ------ |
| GAP-5A-03 | CLOSED |

## MEM-FINAL-AUDIT-5B — SQLite durable reference E2E

| Check | Result |
| ----- | ------ |
| Provider ID | `sqlite.user_profile` / `USER_PROFILE_STORE` / `provider_version=None` |
| Cross-instance reopen + delete durability | `run_durable_user_profile_production_qualification` |
| `production_durable_qualified` | Requires `reopen_passed=True` **and** `delete_reopen_passed=True` |
| Trusted evidence feed | `build_user_profile_admission_evidence_from_durable_qualification` |
| PRODUCT admission | Behavioral + durability evidence, matching `qualification_run_id` |
| Application restart E2E | SessionManager + MemoryControlPlane recall/forget after SQLite reopen |
| Verdict | **REFERENCE DURABLE / RESTART QUALIFIED** (not REAL_VENDOR_RESTART) |

| Gap | Status |
| --- | ------ |
| GAP-5B-01 | **CLOSED** — SQLite UserProfile durable reference production feed |

## MEM-FINAL-AUDIT-5A-R2 — Trusted provider identity binding

| Check | Result |
| ----- | ------ |
| Identity contract | `intergrax/memory/contracts/provider_identity.py` (`MemoryProviderIdentity`) |
| Wiring pair | `MemoryPlatformWiring.user_profile_store` + `user_profile_store_identity` |
| Evidence lookup authority | `trusted_identity.provider_id` (+ capability, version) |
| External plugin identity | `plugin_id` from platform overlay → `plugin_user_profile_store_identity` |
| Spoof (`evil.plugin` → declares `sqlite.user_profile`) | `provider_identity_mismatch` (cannot use sqlite evidence) |
| Version binding | Exact match; unknown runtime version + versioned evidence → mismatch |
| Duplicate evidence | `AMBIGUOUS` → fail-closed |

| Gap | Status |
| --- | ------ |
| GAP-5A-02 | CLOSED |

## MEM-FINAL-AUDIT-5A — Production admission

| Check | Result |
| ----- | ------ |
| Admission owner | `applications/_shared/memory_provider_admission.py` (not MemoryControlPlane) |
| Policy inputs | reference flag + **trusted** behavioral + **trusted** durability evidence (declared qual/durability **not** trusted) |
| Overlay ordering | Admission after external plugin overlay on final `user_profile_store` |
| LAB reference InMemory | Allowed when `ApplicationProfile.LAB` |
| PRODUCT memory disabled | InMemory baseline allowed (store not admission-gated) |
| PRODUCT persistent + InMemory | `MemoryProviderAdmissionError` |
| SQLite PRODUCT | Requires trusted `USER_PROFILE_STORE` evidence (`QUALIFIED`); self-declared qual ignored |
| DocumentStore UserProfile (Mongo path) | Requires trusted behavioral + durability evidence for composite identity (`document_store.user_profile` + `mongodb`; proof `real_vendor_reconnect`) |

## MEM-FINAL-AUDIT-5D — Qdrant SessionTurnIndex real-vendor qualification

| Check | Result |
| ----- | ------ |
| Adapter | `VectorSessionTurnIndexStore` → RAG vector ports → Qdrant integration (`qdrant`) |
| Memory provider ID | `vector.session_turn_index` (`SESSION_TURN_INDEX_STORE`) |
| Backend vendor ID | `qdrant` (`QDRANT_VECTOR_STORE_PROVIDER_ID`) |
| Infrastructure | Local Qdrant (`INTERGRAX_QDRANT_*`, Docker `infra/docker/qdrant`) |
| Behavioral qual | `MemoryProviderQualificationRunner` on real Qdrant-backed store |
| Durability proof | Client/provider reconnect (`REAL_VENDOR_RECONNECT`); service restart **not executed** |
| Evidence source | `qdrant_session_turn_index_real_vendor_qualification` |
| STI production admission | **Enforced (5D-R)** — PRODUCT requires trusted qualification evidence for composite STI identity |
| Application E2E | `SessionManager` episodic recall after Qdrant client reconnect |
| Scope enforcement | Tenant-bound STI + Qdrant metadata filters (`tenant_id`, `session_id`, …) |
| Verdict | **V6 REAL-VENDOR DURABILITY/RECONNECT QUALIFIED** (Qdrant backing only) |

| Gap | Status |
| --- | ------ |
| GAP-4-02 (Qdrant) | **CLOSED** |
| GAP-4-07 (Qdrant) | **CLOSED** |
| GAP-4-07 (pgvector) | OPEN |
| GAP-4-07 (Chroma) | OPEN |
| GAP-5D-01 | **CLOSED (5D-R)** |

## MEM-FINAL-AUDIT-5E — pgvector SessionTurnIndex real-vendor qualification

| Check | Result |
| ----- | ------ |
| Adapter | `VectorSessionTurnIndexStore` → RAG vector ports → pgvector integration (`pgvector`) |
| Memory provider ID | `vector.session_turn_index` (`SESSION_TURN_INDEX_STORE`) |
| Backend vendor ID | `pgvector` (`PGVECTOR_VECTOR_STORE_PROVIDER_ID`) |
| Infrastructure | PostgreSQL 16 + pgvector extension (`infra/docker/postgresql` service `pgvector`; `INTERGRAX_PGVECTOR_*`) |
| Behavioral qual | `MemoryProviderQualificationRunner` on real pgvector-backed store |
| Durability proof | Client/provider reconnect (`REAL_VENDOR_RECONNECT`); PostgreSQL service restart **not executed** |
| Evidence source | `pgvector_session_turn_index_real_vendor_qualification` |
| Scope enforcement | SQL `tenant_id` / namespace / workspace + JSONB metadata filters; cosine distance |
| Cross-vendor admission | Qdrant evidence cannot admit pgvector runtime; adapter-only evidence fails |
| Application E2E | PRODUCT `SessionManager` episodic recall after pgvector client reconnect |
| Verdict | **V6 REAL-VENDOR DURABILITY/RECONNECT QUALIFIED** (pgvector backing) |

| Gap | Status |
| --- | ------ |
| GAP-5E-01 (pgvector STI lacked Memory-scoped real-vendor qualification) | **CLOSED** |
| GAP-4-07 (pgvector) | **CLOSED** |
| GAP-4-07 (Qdrant) | **CLOSED** (unchanged) |
| GAP-4-07 (Chroma) | OPEN |

## MEM-FINAL-AUDIT-5E-R — Qdrant + pgvector Real-Vendor Regression Verification

| Check | Result |
| ----- | ------ |
| Verified SHA | `a640c98f6fee7cb313f371efc102faf8fc0aa8e6` |
| Qdrant suite @ SHA | `test_mem_final_audit_5d_qdrant_session_turn_index_real_vendor.py` — **PASS (13)** |
| pgvector suite @ SHA | `test_mem_final_audit_5e_pgvector_session_turn_index_real_vendor.py` — **PASS (19)** |
| Same-SHA proof | **YES** — both suites executed on identical HEAD |
| Qdrant infra | `intergrax-qdrant` Docker, `localhost:6333` (`INTERGRAX_QDRANT_HOST` default) |
| pgvector infra | PostgreSQL **16.10**, pgvector extension **0.8.0**, psycopg **3.3.4**, DSN target `localhost:5433/intergrax_pgvector` (password omitted) |
| Cross-vendor isolation | `provider_id` + `capability` + `backing_provider_id` + `backing_provider_version`; reverse mismatch covered (5D-R + 5E E2E) |
| Production changes | **NONE** |

| Gap | Status |
| --- | ------ |
| GAP-5E-01 | **FULLY CLOSED** |
| GAP-4-07 (Qdrant) | **CLOSED** |
| GAP-4-07 (pgvector) | **CLOSED** |
| GAP-4-07 (Chroma) | OPEN |

## MEM-FINAL-AUDIT-5E-R2 — Current-HEAD Real-Vendor Verification

| Check | Result |
| ----- | ------ |
| Previous same-SHA verification (5E-R) | `a640c98f6fee7cb313f371efc102faf8fc0aa8e6` |
| **Current-head re-verification** | **`9acff5d0925e962e09bca6c525343ff227980bc8`** |
| Qdrant suite @ SHA | `test_mem_final_audit_5d_qdrant_session_turn_index_real_vendor.py` — **PASS (13)** |
| pgvector suite @ SHA | `test_mem_final_audit_5e_pgvector_session_turn_index_real_vendor.py` — **PASS (19)**, DSN required |
| Same-SHA proof | **YES** |
| Qdrant infra | `intergrax-qdrant` Docker, `localhost:6333` |
| pgvector infra | PostgreSQL **16.10**, pgvector **0.8.0**, psycopg **3.3.4**, `127.0.0.1:5433/intergrax_pgvector` |
| Production changes | **NONE** |

| Gap | Status |
| --- | ------ |
| GAP-5E-01 | **FULLY CLOSED** |
| GAP-4-07 (Qdrant) | **CLOSED** |
| GAP-4-07 (pgvector) | **CLOSED** |
| GAP-4-07 (Chroma) | OPEN |

## MEM-FINAL-AUDIT-5D-R — SessionTurnIndex trusted production admission

| Check | Result |
| ----- | ------ |
| Generic admission | `evaluate_production_session_turn_index_store_admission` |
| Backing resolver | `vector_store_backing_provider_id(IntegrationProfile)` |
| Enforcement | `validate_session_turn_index_store_admission` in Tier-3 vector wiring |
| Tests | `test_mem_final_audit_5d_r_session_turn_index_production_admission.py` |
| Qdrant PRODUCT E2E | Trusted evidence required; mismatch/missing fail closed |

## MEM-FINAL-AUDIT-5D-R2 — SessionTurnIndex plugin identity hardening

| Check | Result |
| ----- | ------ |
| Plugin path | Classified `SessionTurnIndexStorePlugin` only; identity from classifier `plugin_id` |
| Builtin path | `vector.session_turn_index` + integration-profile vector backing when no plugin |
| Forbidden | Builtin Qdrant evidence admitting external plugin; reflection-based `plugin_id` |
| Selection | Multiple STI plugins without explicit policy → fail closed |
| Tests | `test_mem_final_audit_5d_r2_session_turn_index_plugin_identity_hardening.py` |

| Gap | Status |
| --- | ------ |
| GAP-5D-02 | **CLOSED** |
| GAP-5D-01 | **CLOSED (5D-R)** |

## MEM-FINAL-AUDIT-5D-R3 — SessionTurnIndex provider identity override elimination

| Check | Result |
| ----- | ------ |
| API | `build_session_turn_index_store` — removed caller `provider_identity`; platform-owned derivation only |
| Attack surface | External plugin cannot be admitted under builtin Qdrant identity via caller override |
| Direct injection | Separate `session_turn_index_store` + `session_turn_index_store_identity` on `build_session_manager_from_environment` |
| Tests | `test_mem_final_audit_5d_r3_session_turn_index_identity_override_elimination.py` |

| Gap | Status |
| --- | ------ |
| GAP-5D-03 | **CLOSED** |
| GAP-5D-02 | **CLOSED** |
| GAP-5D-01 | **CLOSED (5D-R)** |

## MEM-FINAL-AUDIT-5C — Mongo UserProfile real-vendor qualification

| Check | Result |
| ----- | ------ |
| Adapter | `DocumentStoreUserProfileStore` → `DocumentStore` → Mongo `mongodb` provider |
| Memory provider ID | `document_store.user_profile` (`USER_PROFILE_STORE`) |
| Backend vendor ID | `mongodb` (integration manifest slug) |
| Infrastructure | HARDEN-4F Docker Mongo (replica set `rs0`, local container) |
| Behavioral qual | `MemoryProviderQualificationRunner` on real Mongo-backed store |
| Durability proof | Client/provider reconnect (`REAL_VENDOR_RECONNECT`); not service restart |
| Evidence source | `mongo_real_vendor_qualification` |
| PRODUCT admission | Behavioral + durability evidence, matching `qualification_run_id` |
| Application E2E | MemoryControlPlane remember/recall/forget after Mongo client reconnect |
| Write semantics | `replace_one` + `upsert=True` on `(partition_key, row_key)` unique index |
| Verdict (execution) | Real Mongo behavioral + reconnect durability proved |
| Identity binding (pre-5C-R) | Evidence keyed on adapter ID only — **GAP-5C-01** |

## MEM-FINAL-AUDIT-5C-R — Composite adapter/backend identity binding

| Check | Result |
| ----- | ------ |
| Composite trusted identity | `provider_id` + `backing_provider_id` on `MemoryProviderIdentity` |
| Evidence / registry | Qualification + durability records and `resolve()` discriminate backing |
| Mongo qualification descriptor | `backing_provider_id=mongodb` |
| Substitution attack | Mongo evidence + `InMemoryDocumentStore` adapter wiring → **FAIL CLOSED** |
| Verdict | **REAL-VENDOR DURABILITY/RECONNECT QUALIFIED** (`document_store.user_profile` + `mongodb`) |

| Gap | Status |
| --- | ------ |
| GAP-5C-01 | **CLOSED** (Mongo evidence cannot qualify non-Mongo DocumentStore backend) |
| GAP-4-04 | **FULLY CLOSED** |

## MEM-FINAL-AUDIT-5A-R — Trusted qualification evidence

| Check | Result |
| ----- | ------ |
| Evidence contract | `intergrax/memory/contracts/provider_qualification_evidence.py` |
| Runner → evidence | `qualification_evidence_from_result` |
| Registry | `MemoryProviderQualificationEvidenceRegistry` (+ `InMemoryMemoryProviderQualificationEvidenceRegistry`) |
| Composition injection | `qualification_evidence_registry` on `resolve_memory_platform_wiring` |
| Spoof test | Self-declared `QUALIFIED` without registry → `qualification_evidence_missing` |

---

## AUDIT-5 candidate matrix

| Capability | Provider | AUDIT-5 scenario | Infrastructure | Restart | Failure |
| ---------- | -------- | ---------------- | -------------- | ------- | ------- |
| UserProfile | SQLite | MEM-ENT-15 extension: multi-tenant file isolation under load | local sqlite | YES (extend) | transaction/lock injection |
| UserProfile | DocumentStore → Mongo | qual runner + durable harness on real container | MongoDB testcontainer | YES | connection drop |
| SessionTurnIndex | Vector + Qdrant | upsert turns → restart Qdrant → search recall | Qdrant + embeddings | YES | collection missing |
| SessionTurnIndex | Vector + pgvector | same | Postgres+pgvector | YES | DB restart |
| SessionTurnIndex | Vector + Chroma | same | Chroma service | YES | API error |
| Task memory | SQLite | persist keys → reopen db → read | sqlite file | YES | corrupt db handling |
| Organization | SQLite | profile persist → reopen | sqlite | YES | optional |
| Entity temporal | TBD durable plugin | qual runner + reopen | vendor TBD | YES | P1 strategic |

**Priority:** P0 test coverage = STI real vendor (GAP-4-02); P1 = product wiring fail-closed + Mongo durable UserProfile; P2 = specialized durable stores.

---

## Qualification test command (AUDIT-4 bounded)

```text
uv run pytest tests/unit/memory/test_mem_ent13_provider_qualification.py \
  tests/unit/memory/test_mem_ent13c_durable_provider_qualification.py \
  tests/unit/memory/test_mem_ent13_behavioral_coverage.py \
  tests/unit/memory/test_memory_store_resolver.py \
  tests/unit/applications/test_memory_wiring.py \
  tests/integration/memory/e2e/test_mem_ent15_restart.py \
  tests/integration/memory/e2e/test_mem_ent15_plugin_replaceability.py -q
```

**Result:** 86 passed (log: `.tmp/session/mem-final-audit-4/pytest-qual.log`).
