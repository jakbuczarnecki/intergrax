# MP-6G — E2E / Isolation / Idempotency Qualification

## Canonical status

**MP-6G — QUALIFIED / CERTIFIED** (live PostgreSQL provider-neutral E2E contract suite; subject to independent GitHub audit)

| Milestone | Status |
| --- | --- |
| **MP-6G-C1-Q1** | **CLOSED** |
| **MP-6G-C1** | **CLOSED / RECERTIFIED** |
| **MP-6G** | **CLOSED / CERTIFIED** |
| **MP-6H** | **NEXT** |
| **MP-6** | **IN PROGRESS** (final enterprise certification awaits MP-6H) |

## Qualification identity

| Field | Value |
| --- | --- |
| **baseline_sha** | `31ef13456aff1753313a4dcd5c0933f79be0561a` — MP-6F-C1 enterprise baseline |
| **initial_mp6g_harness_sha** | `fefd4bf9b2317ebe0f451c37fce8b1db1972203f` — initial MP-6G harness |
| **mp6g_c1_harness_sha** | `30900bbd566d956fb5df7f16b00e15057a5a7a91` — MP-6G-C1 hardened harness |
| **qualification_sha** | `5ff90667569bf23c88a4db98d489966e309f4ab7` — exact committed tree for authoritative live PostgreSQL qualification |
| **qualification_sha ancestry** | `mp6g_c1_harness_sha` is an ancestor of `qualification_sha`; delta is test-support only (`tests/integration/collaborative_work/test_mp6g_postgresql_e2e_qualification.py` per-scenario isolated PostgreSQL schemas). **No MP-6G production semantic change.** No MP-6A–F production semantic change. |
| **evidence_update_commit_sha** | _recorded in final evidence commit after this document update (not equal to `qualification_sha`)_ |

> **Evidence vs qualification:** Auditors must bind execution results to `qualification_sha`. Evidence markdown may land in a later commit.

## Hermeticity and import isolation

| Field | Value |
| --- | --- |
| **source_tree_mode** | `clean exact committed checkout` |
| **qualification_sha == HEAD at execution** | **yes** (`5ff90667569bf23c88a4db98d489966e309f4ab7`) |
| **main working tree** | clean at authoritative PostgreSQL invocation (no unrelated source overrides) |
| **import root verification** | `D:\Projekty\intergrax\intergrax\__init__.py` (pattern: `<repo>/intergrax/__init__.py` under qualification tree) |
| **editable-install contamination** | **none** — `intergrax` resolved under repository root matching `qualification_sha` |
| **PYTHONPATH worktree injection** | **not used** |

## PostgreSQL environment

| Field | Value |
| --- | --- |
| **provider** | PostgreSQL |
| **version (runtime)** | 16.6 (Debian 16.6-1.pgdg120+1 on x86_64-pc-linux-gnu) |
| **compose path** | `infra/docker/postgresql/docker-compose.yml` (service `postgresql`, image `postgres:16.6`) |
| **host / port** | `localhost:5434` |
| **database** | `intergrax` |
| **DSN** | `INTERGRAX_COLLABORATIVE_WORK_POSTGRESQL_DSN` set at execution (**credentials REDACTED** in this artifact; pattern `postgresql://***@localhost:5434/intergrax`) |
| **schema strategy** | per-scenario isolated schema `collaborative_work_test_<uuid>` for provider-neutral contract scenarios; shared schema only for concurrent duplicate/distinct pair factories (independent connections, same schema) |

### Environment probe (non-authoritative)

```text
environment_probe = PASS
authoritative_pytest_invocation = not started (at probe time)
```

Probe verified connection, driver import, and PostgreSQL 16.6 version string. Probe pytest invocations **not** counted toward qualification integrity.

## Commands

### SQLite MP-6G qualification (pre-PostgreSQL gate)

```text
uv run pytest tests/qualification/mp6/ -q
```

### Authoritative live PostgreSQL qualification (exactly once at `qualification_sha`)

```powershell
$env:INTERGRAX_COLLABORATIVE_WORK_POSTGRESQL_DSN = "postgresql://***@localhost:5434/intergrax"
uv run pytest tests/integration/collaborative_work/test_mp6g_postgresql_e2e_qualification.py -m "integration and network" -v --tb=short
```

Invokes `run_mp6g_e2e_contract_suite(...)` via `test_mp6g_postgresql_e2e_contract_suite`.

### MP-6 / source-domain regression bundle (at `qualification_sha`)

```text
uv run pytest tests/unit/collaborative_work/test_mp6d_collaborative_activity_append_store.py tests/unit/collaborative_work/test_mp6d_q1_postgresql_qualification_evidence_gates.py tests/unit/collaborative_work/test_mp6e_collaborative_activity_read.py tests/unit/collaborative_work/test_mp6e_c1_composition_and_documentation_gates.py tests/unit/collaborative_work/test_mp6f_collaborative_activity_source_adapters.py tests/unit/collaborative_work/test_mp6f_source_integration_architecture_gates.py tests/qualification/mp6/ tests/unit/collaborative_work/test_shared_work_service.py tests/unit/collaborative_work/test_artifact_service.py tests/unit/collaborative_work/test_decision_binding_service.py tests/unit/collaborative_work/test_context_view_composition.py -q
```

## Execution integrity

| Field | SQLite (`tests/qualification/mp6/`) | PostgreSQL (authoritative) |
| --- | --- | --- |
| **pytest invocations** | 1 | 1 |
| **retries** | 0 | 0 |
| **xdist** | no | no |
| **sharding** | no | no |
| **passed** | 8 | 1 |
| **skipped** | 0 | 0 |
| **xfailed** | 0 | 0 |
| **failed** | 0 | 0 |

PostgreSQL authoritative session log: `.tmp/session/MP-6G-C1-Q1/pytest-postgresql-authoritative.log` (session-local; not committed).

### Authoritative run history

| qualification_sha | outcome | notes |
| --- | --- | --- |
| `30900bbd566d956fb5df7f16b00e15057a5a7a91` | **failed** | replay scenario: shared fixture schema across scenarios leaked prior Activity rows (`len(page2.activities) != 1`). Classified **test-support defect**; no production change. |
| `5ff90667569bf23c88a4db98d489966e309f4ab7` | **passed** | per-scenario isolated schemas; authoritative single invocation. |

## Production E2E path (architecture proof)

```text
CollaborativeWorkService / Artifact / Decision / ContextView (MP-6F wiring)
  → CollaborativeActivityPublicationPort (PublicationPort — sole source ingress)
  → MP-6C publisher authority + ingestion
  → MP-6D PostgreSQL append store
  → MP-6E authorized read service / read port
```

**Boundary proof (qualification harness):**

- **no** source → AppendStore bypass
- **no** source → provider bypass
- **no** direct `INSERT INTO collaborative_activity` semantic shortcut
- **PublicationPort only** for durable Activity creation from sources
- semantic **actor** (`SOURCE_ACTOR`) **≠** MP-6 **publisher** principal
- read **consumer** is a distinct principal from publisher

Provider-specific qualification setup only: `cast(PostgreSQLCollaborativeWorkStore, bundle.store)` in PostgreSQL integration test (not a production abstraction change).

## Ownership

| Concern | Owner |
| --- | --- |
| **effective durability** | MP-6C ingestion (qualification helpers do not override final durability) |
| **recorded_at** | store |
| **append_position** | store |
| producers must not supply store-owned fields | preserved |

## Source coverage matrix (PostgreSQL E2E via `run_mp6g_e2e_contract_suite`)

| Source flow | PostgreSQL E2E | Status |
| --- | --- | --- |
| **WorkItem** create | yes | **PASS** |
| **WorkItem** transition | yes | **PASS** |
| **WorkItem** update | — | **WORK_ITEM_UPDATED — NOT APPLICABLE / no canonical MP-6F seam** |
| **Assignment** create | yes | **PASS** |
| **Assignment** transition | yes | **PASS** |
| **WorkArtifact** create | yes | **PASS** |
| **WorkArtifact** version publish | yes | **PASS** |
| **Decision** binding create | yes | **PASS** |
| **ContextView** compose | yes | **PASS** |

## Isolation matrix

| Scenario | PostgreSQL | Status |
| --- | --- | --- |
| tenant isolation | yes | **PASS** |
| workspace isolation | yes | **PASS** |
| publisher workspace restriction | yes | **PASS** |
| read authority (tenant/workspace) | yes | **PASS** |
| cursor scope mismatch fail-closed | yes | **PASS** |
| cursor not authorization token | yes | **PASS** |
| unauthorized read (zero provider query) | yes | **PASS** |

## Idempotency matrix

| Scenario | PostgreSQL | Status |
| --- | --- | --- |
| source replay (stable activity_id / append_position / recorded_at / durability) | yes | **PASS** |
| same stable id cross-tenant → distinct Activity | yes | **PASS** |
| same stable id cross-workspace → distinct Activity | yes | **PASS** |
| concurrent duplicate ingestion (2 callers, 1 logical activity) | yes | **PASS** |

## Concurrency matrix

| Scenario | PostgreSQL | Status |
| --- | --- | --- |
| concurrent duplicate (MP-6C/MP-6D ingestion path; independent DB connections) | yes | **PASS** |
| concurrent distinct (2 activities, distinct append positions, no loss) | yes | **PASS** |

## Pagination matrix

| Scenario | PostgreSQL | Status |
| --- | --- | --- |
| no duplicate pages | yes | **PASS** |
| no skip pages | yes | **PASS** |
| late `occurred_at` (append order authoritative) | yes | **PASS** |
| append between pages | yes | **PASS** |
| cursor scope mismatch denied | yes | **PASS** |

## Failure / recovery matrix

| Scenario | PostgreSQL | Status |
| --- | --- | --- |
| unknown publisher denied (zero durable Activity) | yes | **PASS** |
| wrong-workspace publisher denied | yes | **PASS** |
| unauthorized read denied | yes | **PASS** |
| mapping failure (source committed, no Activity, error propagated) | yes | **PASS** |
| publication failure → retry idempotency → one Activity | yes | **PASS** |
| custom ingestion policy pluginability | yes | **PASS** |

## Production code changes at qualification

```text
NONE
```

Test-support change at `qualification_sha` relative to `mp6g_c1_harness_sha`: PostgreSQL qualification harness per-scenario schema isolation only.

## Findings

```text
BLOCKING FINDINGS: NONE
```

## Historical qualification states (archival)

Earlier drafts marked live PostgreSQL proof as pending when the integration backend was unavailable in agent-only sessions. That pending state is superseded by the **Canonical status** section above after MP-6G-C1-Q1 closure.

## Regression summary (at `qualification_sha`)

| Suite | Result |
| --- | --- |
| `tests/qualification/mp6/` | 8 passed, skipped: 0, xfailed: 0, failed: 0 |
| MP-6A–F + source-domain regression bundle | 184 passed |
