# MP-6D-Q1 — PostgreSQL Provider Qualification (Collaborative Activity Append Store)

## 1. Task and qualification status

| Field | Value |
| --- | --- |
| **task** | MP-6D-Q1-E1 — PostgreSQL Qualification Evidence Record |
| **qualification status** | **PASS** — live PostgreSQL provider qualification executed and reproduced at `qualification_sha` (below) |
| **MP-6D-Q1** | **CLOSED / RECERTIFIED** (subject to independent GitHub audit of this artifact) |
| **MP-6D** | **CLOSED / RECERTIFIED** (PostgreSQL relational append-store transactional/concurrency qualification only) |
| **bounded scope** | PostgreSQL 16.6 single-node integration environment; independent DB sessions; not multi-region, sharding, HA, or production load certification |

## 2. Evidence identity

| Field | Value |
| --- | --- |
| **implementation_sha** | `131b5e8fac9e6c05bcc0415677dee91288691500` — MP-6D append store production code (`feat(collaborative-work): add atomic mp6 activity persistence`) |
| **qualification_docs_sha** | `1d17af1af8aba05da2984dc2260a4ea30aaa326b` — MP-6D-Q1 docs/recertification commit (`test(collaborative-work): qualify mp6 postgres append store`) |
| **qualification_sha** | `7d46bee3a007e32510488f82d34476f221527a52` — repository `HEAD` at successful live PostgreSQL qualification execution (2026-09-19) |
| **evidence_introduction_sha** | `5890538c4164f5546bd573a63a725fd007b43d6b` — commit that first introduced this qualification evidence artifact |
| **branch** | `development` |
| **HEAD SHA at execution** | `7d46bee3a007e32510488f82d34476f221527a52` |
| **origin/development SHA at execution** | `7d46bee3a007e32510488f82d34476f221527a52` |
| **HEAD == origin/development** | **true** |
| **clean worktree at qualification** | **true** (no tracked/untracked changes before pytest; session logs under `.tmp/` only) |

## 3. Evidence mechanism

Platform runtime `ProofReceipt` / `ProviderQualificationRun` persistence applies to PROVIDER-QUAL catalog flows. **This MP-6D-Q1 evidence uses the repository markdown qualification record pattern** (same family as `MP-6B_CORE_DTO_AND_CONTRACT_HARDENING.md`, `MP-5H_FINAL_ENTERPRISE_CERTIFICATION.md`). No parallel CW-specific proof framework was added.

## 4. Provider environment

| Field | Value |
| --- | --- |
| **provider** | PostgreSQL |
| **version** | 16.6 (Debian 16.6-1.pgdg120+1) |
| **host** | `localhost:5434` (host port mapped from repo Docker compose) |
| **database** | `intergrax` |
| **container/runtime** | Docker — `infra/docker/postgresql/docker-compose.yml`, service `postgresql`, container `intergrax-postgresql`, image `postgres:16.6` |
| **DSN credentials** | **REDACTED** — operator supplies `INTERGRAX_COLLABORATIVE_WORK_POSTGRESQL_DSN`; canonical local pattern `postgresql://***@localhost:5434/intergrax` |
| **qualification schema** | Per-test isolated schema `collaborative_work_test_<uuid>` (fixture teardown drops schema) |

## 5. Qualification command (exact)

```powershell
$env:INTERGRAX_COLLABORATIVE_WORK_POSTGRESQL_DSN = "postgresql://***@localhost:5434/intergrax"
uv run pytest tests/integration/collaborative_work/test_postgresql_collaborative_activity_append_store.py -m "integration and network" -v --tb=short
```

Infrastructure (if not already up):

```powershell
docker compose -f infra/docker/postgresql/docker-compose.yml up -d
```

## 6. Execution integrity

| Field | Value |
| --- | --- |
| **pytest invocations (qualification)** | 1 |
| **pytest invocations (MP-6A/B/C/D regression bundle)** | 1 |
| **automatic test retries on failure** | 0 |
| **pytest-xdist** | no |
| **test sharding** | no |
| **raw console log** | Captured locally at qualification time under `.tmp/session/MP-6D-Q1-E1/pg-qualification.log` (gitignored; summary below is authoritative for auditors who rerun) |

## 7. PostgreSQL qualification result (exact)

```text
passed: 3
skipped: 0
xfailed: 0
failed: 0
```

Duration (qualification run): **1.16s** (pytest summary at `qualification_sha`).

### 7.1 Executed tests

| Test function | Role |
| --- | --- |
| `test_postgresql_collaborative_activity_append_store_contract` | Shared append-store contract suite (idempotency, replay immutability, workspace/tenant isolation, lookup) |
| `test_postgresql_collaborative_activity_concurrent_duplicate` | Two concurrent writers, same idempotency key |
| `test_postgresql_collaborative_activity_concurrent_distinct` | Eight concurrent writers, distinct idempotency keys |

Implementation file: `tests/integration/collaborative_work/test_postgresql_collaborative_activity_append_store.py`. Shared contracts: `tests/unit/collaborative_work/collaborative_activity_append_store_contract.py`.

## 8. Independent bundle concurrency evidence

Concurrency qualification opens **bundle A** (primary fixture) and **bundle B** via a second call to `open_postgresql_collaborative_work_repositories(config=bundle_a.store.config, schema_name=bundle_a.store.schema_name)` — **independent store instances / DB sessions**, not a shared in-process singleton.

Stores are materialized with `collaborative_activity_append_store_from_postgresql_bundle(...)`.

**RLock is not cross-store correctness proof:** each `PostgreSQLCollaborativeActivityAppendStore` holds a per-instance `threading.RLock()`. Per-instance RLock was not shared between the competing stores. Concurrent duplicate/distinct tests deliberately use two bundles so competing writers do **not** share one RLock; PostgreSQL correctness was proven across independent store instances / DB sessions.

## 9. Concurrent duplicate evidence

Verified semantics (shared contract `run_concurrent_duplicate_contract`):

- 2 concurrent writers
- same idempotency key
- one logical activity
- same `activity_id`, `append_position`, `recorded_at`, `durability_class`
- both `append_idempotent` calls succeed; returned activities equal

## 10. Concurrent distinct evidence

Verified semantics (shared contract `run_concurrent_distinct_contract`, `parallel=8`):

- 8 concurrent writers
- same tenant/workspace
- distinct idempotency keys (`stable_id` distinct per writer)
- `append_position` values **1..8** with no collisions
- all eight activities materialized

## 11. Atomicity and rollback evidence

**Single provider transaction (`PostgreSQLCollaborativeActivityAppendStore._append_once`):**

1. idempotency **lookup** (`_load_by_idempotency_key`)
2. workspace **`append_position` allocation** — UPSERT on `collaborative_activity_workspace_sequence` with `ON CONFLICT DO UPDATE` + `RETURNING`
3. **`recorded_at`** assignment via injected `utc_now`
4. durable **insert** into `collaborative_activities`

**Rollback / replay on unique violation:**

- unique violation on insert → **rollback current transaction**
- **fresh replay transaction** → existing activity lookup by idempotency key
- if row exists, return stored activity (concurrent duplicate path)

SQLite unit suite additionally proves transaction rollback leaves no partial row (`test_mp6d_transaction_rollback_leaves_no_partial_row`).

## 12. Constraint evidence (PostgreSQL DDL)

From `intergrax/collaborative_work/postgresql_repository.py` schema bootstrap:

- `collaborative_activities.activity_id` — **PRIMARY KEY**
- idempotency — **UNIQUE** `(tenant_id, workspace_id, source_qualified_id, source_stable_id, activity_type_qualified_id)`
- ordering — **UNIQUE** `(tenant_id, workspace_id, append_position)`
- `collaborative_activity_workspace_sequence` — **PRIMARY KEY** `(tenant_id, workspace_id)`

## 13. Replay immutability evidence

Contract suite (`run_append_store_contract_suite`) verifies:

- same idempotency key
- changed requested durability, effective durability, and `occurred_at` on republication
- **original stored activity returned** (no in-place mutation)

## 14. SQLite / MP-6A–D regression evidence

At `qualification_sha`, single pytest process:

```text
tests/unit/collaborative_work/test_mp6a_*.py
tests/unit/collaborative_work/test_mp6b_*.py
tests/unit/collaborative_work/test_mp6c_*.py
tests/unit/collaborative_work/test_mp6d_collaborative_activity_append_store.py
→ 135 passed, 0 skipped, 0 xfailed, 0 failed
```

Combined with live PostgreSQL integration (3 passed): **138 passed** total MP-6A/B/C/D + PostgreSQL integration qualification bundle.

PostgreSQL integration is included in MP-6D regression scope; SQLite append-store concurrency remains qualified in `test_mp6d_collaborative_activity_append_store.py`.

## 15. Documentation gate evidence

At `qualification_sha`:

```text
uv run pytest tests/unit/collaborative_work/test_mp6a_documentation_regression_gates.py -q
→ 6 passed, 0 skipped, 0 xfailed, 0 failed
```

Post-evidence artifact gate: `tests/unit/collaborative_work/test_mp6d_q1_postgresql_qualification_evidence_gates.py`.

## 16. git diff check

```text
git diff --check → PASS (no conflict markers or whitespace errors in tracked diff at qualification time)
```

## 17. Encoding / ADR evidence

Commit `1d17af1af8aba05da2984dc2260a4ea30aaa326b` repaired **ADR-MP-007** UTF-8 encoding (mojibake removal) in bounded canonical MP-6 status/docs paths. Qualification at `qualification_sha` assumes repaired ADR on disk.

## 18. Provenance links

| Artifact | Path |
| --- | --- |
| Append store implementation | `intergrax/collaborative_work/collaborative_activity_append_store.py` |
| PostgreSQL integration qualification | `tests/integration/collaborative_work/test_postgresql_collaborative_activity_append_store.py` |
| Shared contract suite | `tests/unit/collaborative_work/collaborative_activity_append_store_contract.py` |
| MP-6D SQLite/unit qualification | `tests/unit/collaborative_work/test_mp6d_collaborative_activity_append_store.py` |
| Composition / bundle open | `intergrax/collaborative_work/persistence.py` (`open_postgresql_collaborative_work_repositories`) |
| Ownership ADR | `docs/project/technical/adr/entries/2026-09-18/ADR-MP-007.md` |
| Implementation commit | `131b5e8fac9e6c05bcc0415677dee91288691500` |
| Qualification/docs commit | `1d17af1af8aba05da2984dc2260a4ea30aaa326b` |

## 19. Reproducibility (operator prerequisites)

1. Docker available; start canonical PostgreSQL: `docker compose -f infra/docker/postgresql/docker-compose.yml up -d`
2. Configure `INTERGRAX_COLLABORATIVE_WORK_POSTGRESQL_DSN` (credentials **not** committed)
3. Checkout `qualification_sha` or later `development` containing this artifact
4. Run qualification command (section 5)
5. Expect **3 passed, 0 skipped, 0 xfailed, 0 failed**
6. Run MP-6 regression bundle (section 14) and documentation gates (section 15)

## 20. Production code changes for this evidence task

**NONE** for store/runtime — evidence and documentation gates only.

## 21. Findings

**BLOCKING FINDINGS: NONE** (at evidence authoring against live rerun at `qualification_sha`).

## 22. Status transition (after valid artifact commit)

```text
MP-6D-Q1-E1 — CLOSED
MP-6D-Q1 — CLOSED / RECERTIFIED
MP-6D — CLOSED / RECERTIFIED
MP-6E — CLOSED / RECERTIFIED
MP-6 — ENTERPRISE CERTIFIED / CLOSED (MP-6H — CLOSED / CERTIFIED)
```
