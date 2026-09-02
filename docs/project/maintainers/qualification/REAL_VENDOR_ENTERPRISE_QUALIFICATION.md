# PROVIDER-QUAL-9 — Real-Vendor-Wide Enterprise Qualification

**Status:** `READY_FOR_REVIEW`

**Baseline ancestor:** `4c2e6e962f5dcff78b6fabe16afe68af50220cb0`

**Audit HEAD:** `cf34a48bca6fac1ce343a864caebec624e3bbf1c` (branch `development`, ancestor verified)

**Audit date:** 2026-09-02

---

## 1. Executive summary

Enterprise-wide qualification audit of **existing** real vendor integrations in Intergrax. This phase is **audit and evidence**, not vendor feature expansion.

**Verdict:** Platform provider abstractions, plugin registration, typed integration contracts, runtime composition, qualification lifecycle, persistence, observability, and diagnostics **work against real backends** without vendor-specific logic in qualification core. Mandatory PostgreSQL proofs re-executed successfully.

| Gate | Result |
|------|--------|
| Provider inventory reflects code truth | YES |
| No qualification-core vendor switch | YES (architecture gate) |
| No second provider registry | YES |
| No second qualification framework | YES |
| Capability qualification domain-owned | YES (CW persistence suite) |
| Real providers use real services (mandatory PG) | YES |
| Synthetic vs real clearly separated | YES |
| Provider config provider-owned | YES |
| Pluginability preserved | YES |
| Typed public contracts used | YES |
| Unsupported capability fails explicitly | YES (MySQL/Oracle CW) |
| No vendor silent fallback | YES |
| No secret leakage in evidence model | YES (design + tests) |
| Version provenance truthful | PARTIAL — execution-supplied (documented gap) |
| Qualification evidence durable where claimed | YES (Mongo persistence proof) |
| Observability/diagnostics reused | YES |
| No circular qualification claim | YES (Mongo is persistence backend, not DocumentStore subject) |
| Integration test ≠ qualification claim | YES (matrix below) |
| Mandatory environments pass | YES (PostgreSQL) |
| Unresolved P0/P1 from this audit | NONE |

**Production code changes in this phase:** 0

---

## 2. Provider inventory

**Authority:** `intergrax/integrations/providers/` directory layout + `register.py` / `bundle.py` / `integration.py` + `intergrax/runtime/integrations/registry_v2.py` + tests.

**Scale:** 37 provider categories, ~200 registered slugs. Standard registration pattern: `register.py` → factory registration, `bundle.py` → runtime factories, `integration.py` → `PlatformIntegrationContract` implementation, `manifest.py` → preset metadata.

### 2.1 Focus categories (audit depth)

#### Relational store (15 providers)

| provider_id | adapter | registration | typed contract | real backend env | integration tests | qualification suite | classification |
|-------------|---------|--------------|----------------|------------------|-------------------|---------------------|----------------|
| postgresql | `relational_store/postgresql/` | YES | `RelationalStoreIntegrationContract` | `infra/docker/postgresql` (5434), `infra/integration` (5432) | unit + integration + E2E | `cw.postgresql.repository.v1` | **A** |
| sqlite | `relational_store/sqlite/` | YES | j.w. | local file | unit + multi-provider qual | `cw.sqlite.repository.v1` | **A** |
| mysql | `relational_store/mysql/` | YES | j.w. | `infra/docker/mysql` (3306) | unit only | — | **B** |
| oracle | `relational_store/oracle/` | YES | j.w. | no docker in repo | P2 conformance bundle only | — | **B** |
| azure_sql, bigquery, cloud_sql, databricks, duckdb, motherduck, mssql, neon, snowflake, supabase, timescaledb | respective dirs | YES | j.w. | vendor-specific / none local | partial P-batch | — | **B/C** |

#### Document store (3 providers)

| provider_id | adapter | registration | typed contract | real backend env | integration tests | qualification | classification |
|-------------|---------|--------------|----------------|------------------|-------------------|---------------|----------------|
| mongodb | `document_store/mongodb/` | YES | `DocumentStore` + `DocumentStoreVendorIntegrationContract` | `infra/docker/mongodb` (27017) | unit + HARDEN + qual persistence | persistence/discovery/validity proof only | **B** |
| cassandra | `document_store/cassandra/` | YES | j.w. | `infra/docker/cassandra` | unit | — | **B** |
| dynamodb | `document_store/dynamodb/` | YES | j.w. | AWS / localstack | P2 conformance | — | **B** |

#### Vector store (10 providers)

| provider_id | adapter | registration | typed contract | real backend env | integration tests | qualification | classification |
|-------------|---------|--------------|----------------|------------------|-------------------|---------------|----------------|
| qdrant | `vector_store/qdrant/` | YES | `VectorStoreIntegrationContract` | `infra/docker/qdrant` (6333) | unit + platform proof + RAG harness | live RAG harness only | **B** |
| chroma | `vector_store/chroma/` | YES | j.w. | `infra/docker/chromadb` | live qualification doc | RAG live qual | **B** |
| pgvector | `vector_store/pgvector/` | YES | j.w. | pgvector compose (5433) | live qualification | RAG live qual | **B** |
| pinecone, weaviate, milvus, vespa, inmemory, lancedb, typesense | respective dirs | YES | j.w. | partial docker | partial unit | — | **B/C** |

#### Key-value cache (4 providers)

| provider_id | adapter | registration | typed contract | real backend env | integration tests | qualification | classification |
|-------------|---------|--------------|----------------|------------------|-------------------|---------------|----------------|
| redis | `key_value_cache/redis/` | YES | `KeyValueCacheIntegrationContract` | `infra/integration` redis:7.2.4 (6379) | unit + distributed integration (5 files) | — | **B** |
| memcached, elasticache, upstash_redis | respective dirs | YES | j.w. | memcached docker | P2 conformance | — | **B** |

#### Message bus (12), object storage (9), search (12), observability (25)

All follow the same registration pattern. Real Docker services exist for kafka, rabbitmq, nats, minio, elasticsearch, etc. under `infra/integration/docker-compose.yml` profiles. **No `ProviderQualificationSuite`** exists for these categories. Classification: **B** (real adapter + connectivity tests) or **C** (scaffold/incomplete).

### 2.2 Classification legend

| Code | Meaning |
|------|---------|
| **A** | QUALIFICATION_READY_EXISTING_CONTRACT |
| **B** | REAL_PROVIDER_TESTABLE_BUT_NO_QUALIFICATION_CONTRACT |
| **C** | PROVIDER_IMPLEMENTATION_INCOMPLETE |
| **D** | LEGACY / NON_CANONICAL_PATH |
| **E** | DOCUMENTED_ONLY / NO_REAL_IMPLEMENTATION |
| **F** | ENVIRONMENT_UNAVAILABLE |

### 2.3 Catalog manifest coverage

`intergrax/integrations/registry/catalog_manifests.py` includes stable presets for: `POSTGRESQL`, `SQLITE`, `REDIS`, `QDRANT`, `INMEMORY`, and others. **Not in catalog manifests:** `MYSQL`, `ORACLE`, `MONGODB` (registered via provider packages, not Tier-3 preset shortcuts).

---

## 3. Category / capability map

Qualification is **capability-scoped**, never vendor-global.

| Capability ID | Domain | Suite ID | Providers with binding | Providers without binding |
|---------------|--------|----------|------------------------|---------------------------|
| `collaborative_work.persistence.v1` | `collaborative_work` | `cw.postgresql.repository.v1` | postgresql | mysql, oracle, all non-relational |
| `collaborative_work.persistence.v1` | `collaborative_work` | `cw.sqlite.repository.v1` | sqlite | — |
| DocumentStore durability (qual infrastructure) | `core.qualification` | — (persistence layer) | mongodb (as evidence store) | — |
| RAG vector operations | `rag` | live harness (not ProviderQualification) | qdrant, chroma, pgvector | pinecone, weaviate, … |
| Key-value cache semantics | `distributed` | — | redis (integration only) | memcached, elasticache |

**Fail-closed example:** MySQL relational provider registered and testable for `RelationalStoreIntegrationContract`, but **no** `bind_collaborative_work_materialization` → CW qualification correctly unsupported, not PostgreSQL fallback.

---

## 4. Platform reuse

| Mechanism | Reused | Notes |
|-----------|--------|-------|
| `ProviderQualificationSubject` / `ProviderQualificationRun` | YES | `intergrax/core/qualification/provider.py` |
| `execute_provider_qualification` shared runner | YES | `intergrax/core/qualification/execution.py` |
| `ProviderQualificationSuite` + domain binding Protocol | YES | `intergrax/core/qualification/suite.py` |
| Requalification identity | YES | `requalification.py` |
| Persistence | YES | `DocumentStoreProviderQualificationPersistence` |
| Discovery / validity lifecycle | YES | `discovery.py`, `validity*.py` |
| Observability | YES | `observability.py` (QUAL-7 closed) |
| Diagnostics | YES | `PlatformProblemSignal` downstream by host |
| Integration registry v2 | YES | single contract-aware registry |
| Plugin registration | YES | `register_integration_plugin` |
| Second qualification engine | NO | not created |
| Vendor dispatch in qual core | NO | verified by gate |
| Functional qualification (Q1–Q5) | YES (separate subsystem) | cross-domain diagnostics, not provider qual |

---

## 5. Pluginability

**Onboarding path:**

1. Implement provider under `intergrax/integrations/providers/<category>/<slug>/`
2. Register via `register.py` → integration catalog
3. Optionally register registry v2 contract binding
4. Domain qualification requires `ProviderQualificationDomainBinding` materializer (not automatic)

**Architecture proofs:**

| Proof | Location | Claim |
|-------|----------|-------|
| External KV plugin | `tests/unit/integrations/test_external_plugin.py` | synthetic provider registers + resolves without core change |
| Synthetic qual binding | `test_synthetic_third_provider_requires_no_core_vendor_dispatch_changes` | qualification runner accepts third binding without core edit |
| ACME VK reference plugin | `tests/integration/vendor_knowledge/test_acme_reference_external_provider_proof.py` | entry-point external provider (requires editable package install) |

**Blockers for new vendor onboarding:** none at registration layer. Capability qualification requires domain owner to add `ProviderQualificationDomainBinding` — by design.

---

## 6. Provider abstraction audit

| Check | Result |
|-------|--------|
| Vendor-specific core changes (this phase) | 0 |
| Second provider registry | 0 |
| Qualification-core vendor `if provider_id ==` branches | 0 (gate) |
| Reflection-based authoritative dispatch in qual core | 0 |
| Vendor SDK imports in qual core / CW suite | 0 (gate) |
| Vendor SDK imports in provider adapters | Expected (boundary modules only) |

**Static vendor leakage gate:** `tests/unit/core/qualification/test_provider_qualification_vendor_abstraction_gate.py` — scans `execution.py`, `suite.py`, `repository_qualification_suite.py` for forbidden patterns (`psycopg`, vendor ID branches, etc.).

**CW vendor neutrality:** `tests/unit/collaborative_work/test_vendor_neutrality.py` — AST scan of domain modules.

---

## 7. Real backend matrix

### 7.1 Environment evidence (this audit run)

| Service | Container | Image | Port | Health | Used in proof |
|---------|-----------|-------|------|--------|---------------|
| PostgreSQL | `intergrax-postgresql` | `postgres:16.6` | 5434→5432 | healthy | YES — mandatory |
| MongoDB | `intergrax-mongodb` | `mongo:7.0` | 27017 | healthy | YES — qual persistence |
| Qdrant | `intergrax-qdrant` | `qdrant/qdrant:v1.16.2` | 6333-6334 | up | available, not re-run this phase |
| pgvector | `intergrax-pgvector` | `pgvector/pgvector:0.8.0-pg16` | 5433 | healthy | available |
| MySQL | — | `mysql:8.4` compose exists | 3306 | **not running** | unit tests only |
| Redis | — | `redis:7.2.4` in integration compose | 6379 | **not running** | skipped (optional dep) |

DSN used (sanitized): `postgresql://***@localhost:5434/intergrax`, `mongodb://localhost:27017/intergrax_qual`

### 7.2 Per-provider proof results

| Provider | Capability | Test | Result |
|----------|------------|------|--------|
| PostgreSQL | CW persistence | `test_provider_qualification_execution_postgresql.py` | PASS |
| PostgreSQL | CW repository | `test_postgresql_repository.py` (14 tests) | PASS |
| PostgreSQL | CW E2E | `tests/e2e/collaborative_work/ -m e2e` (35 tests) | PASS |
| SQLite | CW persistence + requalification | `test_provider_qualification_multi_provider_proof.py` | PASS |
| MongoDB | Qual persistence/discovery/validity | 3 integration files (6 tests) | PASS |
| MySQL | Relational unit | `test_mysql.py` | PASS (unit, no real backend in run) |
| Redis | Unit | `test_redis.py` | SKIP/FAIL without `redis` extra |
| Qdrant | Unit metadata filter | `test_qdrant_metadata_filter.py` | FAIL without `vector-qdrant` extra |

---

## 8. Qualification matrix

| Provider | Category | Capability | CONNECTIVITY_TESTED | CONTRACT_TESTED | QUALIFIED | PRODUCTION_QUALIFIED |
|----------|----------|------------|---------------------|-----------------|-----------|----------------------|
| postgresql | relational_store | `collaborative_work.persistence.v1` | YES | YES | YES | YES |
| sqlite | relational_store | `collaborative_work.persistence.v1` | YES | YES | YES | NO (lab) |
| mysql | relational_store | relational (no CW) | unit only | YES (unit) | NO | NO |
| oracle | relational_store | relational (no CW) | partial | partial (P2) | NO | NO |
| mongodb | document_store | DocumentStore port | YES | YES | NO | NO |
| mongodb | — | qual evidence persistence | YES | YES | N/A (infra) | N/A |
| redis | key_value_cache | KV cache | integration exists | YES | NO | NO |
| qdrant | vector_store | RAG vector ops | YES | YES | NO | NO |

**Important:** MongoDB qualification persistence proof does **not** qualify MongoDB DocumentStore semantics. Evidence backend ≠ qualification subject.

---

## 9. Version provenance

| Provider | Version source | Classification | Notes |
|----------|----------------|----------------|-------|
| postgresql | `DEPLOYMENT_CONFIG` + execution-supplied `"16.6"` in test harness | D for qual evidence | Docker image `postgres:16.6`; not backend-introspected in qual core |
| sqlite | `USER_SUPPLIED` / execution config | D | `"execution-config-lab"` in multi-provider proof |
| mysql | `UNKNOWN` | E | no version in qual path |
| oracle | `UNKNOWN` | E | no docker, no qual |
| mongodb | `CLIENT/DRIVER_VERSION` (pymongo) available; not in qual subject | C/D | gap: no canonical Integrations introspection API |
| redis | `DEPLOYMENT_CONFIG` (compose pin) | C | not propagated to qual |
| qdrant | `DEPLOYMENT_CONFIG` (compose pin v1.16.2) | C | not propagated to qual |

**Gap (P2):** `ProviderQualificationSubject.provider_version` is execution-supplied. Platform does not introspect backend version at qualification-core layer. Do not treat D/E as authoritative backend version in production qualification claims.

---

## 10. Security / secret safety

| Check | Status |
|-------|--------|
| DSN/password in `ProviderQualificationRun.evidence` | Prevented by design (subject carries IDs, not secrets) |
| Provider-owned config | YES — `IntegrationProfile.options` per provider slug |
| Test harness env vars | DSN via `INTERGRAX_*` env, not committed |
| Evidence persistence documents | No raw connection strings in qual persistence schema |
| Exception stringification | Provider adapters normalize; qual core does not stringify raw vendor errors into evidence |

---

## 11. Failure semantics

Provider/category contracts use `IntegrationConfigurationError`, `IntegrationError` from `intergrax/integrations/contracts/base.py`. Qualification layer adds:

- `ProviderQualificationResolutionError` — profile resolution failure
- `ProviderQualificationMaterializationError` — binding materialization failure
- `ProviderQualificationSubjectMismatchError` — fail-closed identity mismatch
- `ProviderQualificationSuiteInfrastructureError` — infra vs semantic rejection separation

Unsupported capability: explicit error at materialization (no silent fallback).

---

## 12. Lifecycle / cleanup

| Provider | open/use/close | Proof |
|----------|----------------|-------|
| PostgreSQL CW | `CollaborativeWorkRepositories.close()` | integration + E2E |
| SQLite CW | same pattern | multi-provider proof |
| Mongo qual persistence | DocumentStore session lifecycle | durable reopen test |
| Redis | client factory in provider bundle | integration distributed tests |

No client leak patterns observed in qualification execution path; materialization handle `close()` invoked in suite teardown.

---

## 13. Transactions / concurrency

| Claim | Proved | Scope |
|-------|--------|-------|
| PostgreSQL CW transactions (commit/rollback) | YES | repository integration + E2E |
| SQLite CW transactions | YES | multi-provider semantic suite |
| Cross-connection concurrency | YES | HARDEN proofs (mongo workers), E2E |
| MySQL/Oracle transaction semantics via qual | NO | no qual suite |

Claims limited to what CW contract promises; PostgreSQL-specific semantics not imposed on other relational vendors.

---

## 14. Observability / diagnostics

Unchanged from PROVIDER-QUAL-7/8:

- Qualification lifecycle signals via `ProviderQualificationExecutionObservabilityPort`
- Infrastructure phases: resolution → materialization → suite → persistence
- Diagnostics: `PlatformProblemSignal` owned by core qualification; host/composition downstream
- No vendor-specific qualification logging stack

---

## 15. Legacy bypass findings

| Finding | Severity | Location pattern |
|---------|----------|------------------|
| LLM adapters outside Integrations | P2 (intentional) | `intergrax/llm_adapters/` — separate registry |
| Embedding providers outside Integrations | P2 (intentional) | `intergrax/rag/embedding/` |
| P-shared factories with lazy vendor import | P3 | `integrations/_shared/p3/factories.py` |
| Conversation channel cutover test drift | P2 | `google_chat`, `mattermost`, `rocket_chat` legacy factory detection |
| `google_workspace` integration imports `threading` | P3 | flagged by runtime cutover gate |

No P0/P1 architecture violations in qualification path. Direct vendor construction in `intergrax/core/` outside provider adapters: **not found** for audited providers.

---

## 16. Gaps by severity

### P0
None.

### P1
None from this audit.

### P2
1. **Provider version not backend-introspected** — qual evidence uses execution-supplied version strings.
2. **No shared semantic qualification suite for DocumentStore, VectorStore, KeyValueCache** — would require domain owner contract (ARCHITECTURAL_DECISION_REQUIRED for new suites).
3. **MySQL/Oracle lack CW materialization** — narrow provider extension, not qual-core change.
4. **Runtime cutover test failures** for some conversation_channel providers (pre-existing, unrelated to qual).

### P3
1. Catalog manifest gaps for mysql/oracle/mongodb presets.
2. Optional dependency extras not in default `uv sync` (`integrations-postgresql`, `vector-qdrant`, redis in dev group).
3. Oracle has no local Docker qualification environment.

---

## 17. Exact commands / results

```text
# Repository safety
git fetch --no-tags origin development
git branch --show-current          → development
git rev-parse HEAD                 → cf34a48bca6fac1ce343a864caebec624e3bbf1c
git merge-base --is-ancestor 4c2e6e962f5dcff78b6fabe16afe68af50220cb0 HEAD → OK

# Dependency sync (required for PostgreSQL real proofs)
uv sync --extra integrations-postgresql --extra integrations-mongodb

# Mandatory regression
uv run pytest tests/unit/core/qualification/ -q
→ 224 passed

# PostgreSQL real proofs
$env:INTERGRAX_COLLABORATIVE_WORK_POSTGRESQL_DSN="postgresql://intergrax:intergrax@localhost:5434/intergrax"
uv run pytest tests/integration/core/qualification/test_provider_qualification_execution_postgresql.py \
  tests/integration/collaborative_work/test_postgresql_repository.py \
  tests/integration/core/qualification/test_provider_qualification_multi_provider_proof.py -q
→ 23 passed

uv run pytest tests/e2e/collaborative_work/ -m e2e -q
→ 35 passed

# MongoDB qual infrastructure proofs
$env:INTERGRAX_MONGODB_DSN="mongodb://localhost:27017/intergrax_qual"
uv run pytest tests/integration/core/qualification/test_provider_qualification_persistence_durable_reopen.py \
  tests/integration/core/qualification/test_provider_qualification_discovery_mongo.py \
  tests/integration/core/qualification/test_provider_qualification_validity_mongo.py -q
→ 6 passed

# Architecture / plugin gates
uv run pytest tests/unit/core/qualification/test_provider_qualification_vendor_abstraction_gate.py \
  tests/unit/collaborative_work/test_vendor_neutrality.py \
  tests/unit/integrations/test_external_plugin.py \
  tests/unit/core/qualification/test_provider_qualification_execution_runner.py::test_synthetic_third_provider_requires_no_core_vendor_dispatch_changes -q
→ 9 passed
```

Session logs: `.tmp/session/provider-qual-9/`

---

## 18. Provider onboarding rules

1. Add provider under `integrations/providers/<category>/<slug>/` with register/bundle/integration.
2. Register typed contract via existing catalog + registry v2.
3. Keep vendor SDK imports inside provider boundary modules.
4. Provider config via `IntegrationProfile.options[slug]` — never in qual core.
5. Capability qualification requires domain `ProviderQualificationDomainBinding` — not automatic on registration.
6. Label synthetic/architecture-only providers explicitly; never `PRODUCTION_QUALIFIED`.
7. Version in qual evidence must reflect provenance class (prefer backend introspection when available).

---

## 19. Next phase

1. **Domain-owned semantic suites** for DocumentStore, VectorStore, KeyValueCache (requires architectural decision per category).
2. **MySQL/Oracle CW materialization** — narrow `bind_collaborative_work_materialization` adapters.
3. **Backend version introspection** via existing provider contracts (not qual-core generic subsystem).
4. **INDEPENDENT PROVIDER-QUAL-9 SHA AUDIT** at commit SHA after merge.

Do **not** start final platform-wide enterprise audit in this phase.

---

## 20. Files

| Kind | Path |
|------|------|
| Docs | `docs/project/maintainers/qualification/REAL_VENDOR_ENTERPRISE_QUALIFICATION.md` |
| Production | none (audit-only) |
| Tests | existing suites re-executed (no new tests) |

---

## Appendix A — Docker service reference

Central stack: `infra/integration/docker-compose.yml` (profiles: core, queue, rag, data, secrets, observability, cloud, heavy, vllm, llama-cpp, all).

Per-service compose under `infra/docker/<service>/docker-compose.yml`. No testcontainers usage in test code.
