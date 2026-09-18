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
| SessionTurnIndex | `InMemorySessionTurnIndexStore` | `VectorSessionTurnIndexStore` (ephemeral vector ports) | **NO** | **NOT QUALIFIED** |
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
| SessionTurnIndex | `VectorSessionTurnIndexStore` | vector adapter | `SessionTurnIndexStore` | STI EP optional | via RAG | **NO proof** | unit scope | **NO** | `enable_session_vector_index` + RAG | V1–V2 | **NOT REAL-VENDOR QUALIFIED** |
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
| Qdrant / pgvector / Chroma | RAG `vectorstore_manager` | integration layer | STI ports | NO Memory EP | unknown | **NO** | RAG tests only | **NO Memory STI E2E** | vector flags | V1 | **UNPROVEN CAPABILITY MAPPING** for Memory |

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
| **SQLite** | Yes (UserProfile, Task, Org, Conversational, Session) | file-backed tests + MEM-ENT-15 restart | **DURABILITY/RESTART** per store; not external V6 |
| **MongoDB** | UserProfile via DocumentStore only | wiring unit test with factory | **NOT MEMORY-VENDOR QUALIFIED** |
| **PostgreSQL** | RFC spike only | `postgres_memory_backend_rfc.py` | **PLANNED ONLY (V0)** |
| **Qdrant** | None for Memory; RAG vector integration may use Qdrant elsewhere | **zero** Memory STI tests naming qdrant | **< V6** |
| **pgvector** | same as Qdrant pattern | none Memory | **< V6** |
| **Chroma** | same | none Memory | **< V6** |

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
| GAP-4-02 | SessionTurnIndex | `VectorSessionTurnIndexStore` + any vector backend | write → restart/reconnect → recall | P2 | AUDIT-5 P0 candidate |
| GAP-4-03 | Entity / Procedural / LH | in-memory only | durable vendor + restart | P2 | AUDIT-5 |
| GAP-4-04 | UserProfile | Mongo DocumentStore | real-vendor qual execution | P2 | AUDIT-5 |
| GAP-4-05 | Organization | Mongo path | durable org store (uses InMemory org on mongo LTM path) | P2 | AUDIT-6 |
| GAP-4-06 | Task memory | SQLite | restart/failure vendor suite | P2 | AUDIT-5 |
| GAP-4-07 | All vector backends | Qdrant/pgvector/Chroma | Memory-scoped E2E | P2 | AUDIT-5 |
| GAP-4-08 | PostgreSQL | Memory bundle | implementation | P3 | post-RFC |

**P0:** NONE  
**P1:** NONE (GAP-4-01 closed in MEM-FINAL-AUDIT-5A — production persistent USER/LTM fails closed on reference/unknown/non-qualified providers)

## MEM-FINAL-AUDIT-5A — Production admission

| Check | Result |
| ----- | ------ |
| Admission owner | `applications/_shared/memory_provider_admission.py` (not MemoryControlPlane) |
| Policy inputs | `MemoryStoreProviderMetadata` + `MemoryProviderQualificationStatus` (no class-name / vendor switches) |
| Overlay ordering | Admission after external plugin overlay on final `user_profile_store` |
| LAB reference InMemory | Allowed when `ApplicationProfile.LAB` |
| PRODUCT memory disabled | InMemory baseline allowed (store not admission-gated) |
| PRODUCT persistent + InMemory | `MemoryProviderAdmissionError` |
| SQLite qualified | Admitted when metadata `QUALIFIED` + `DURABLE` |
| DocumentStore UserProfile (Mongo path) | Durable metadata but `NOT_QUALIFIED` until AUDIT-5C — **fail-closed** when persistent flags on |

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
