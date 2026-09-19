# MP-6E-C1 — PostgreSQL Read Provider Qualification (Collaborative Activity Read Port)

## 1. Task and qualification status

| Field | Value |
| --- | --- |
| **task** | MP-6E-C1 — contract-driven read wiring, PostgreSQL read qualification, documentation repair |
| **requalification task** | **MP-6E-C1-Q1** — Exact-SHA PostgreSQL read requalification |
| **qualification status** | **PASS** — live PostgreSQL read provider qualification executed once on exact committed `qualification_sha` (session log under `.tmp/session/MP-6E-C1-Q1/`) |
| **MP-6E-C1-Q1** | **CLOSED** (subject to independent GitHub audit of this artifact) |
| **MP-6E-C1** | **CLOSED / RECERTIFIED** (subject to independent GitHub audit of this artifact) |
| **MP-6E** | **CLOSED / RECERTIFIED** (scoped read; SQLite + PostgreSQL read providers qualified on live integration DB) |
| **MP-6F** | **NEXT** |
| **bounded scope** | PostgreSQL 16.6 single-node integration environment; per-test isolated schema; not multi-region, HA, or production load certification |

## 2. Evidence identity

| Field | Value |
| --- | --- |
| **implementation_sha** | `eb586ccefea80ff05e0dd4f4f1e784ed1c398350` — MP-6E scoped read stack (`feat(collaborative-work): add mp6e scoped activity read`) |
| **mp6e_c1_correction_sha** | `3c7b45d63e3c2d9dfd1afd22101506983bf25ce0` — MP-6E-C1 composition / documentation correction |
| **qualification_sha** | `fe79e01769853eb3655bfd90e9f26e1be6e1dcdd` — exact committed Git tree used for live PostgreSQL read qualification execution (ancestor of `mp6e_c1_correction_sha`; no MP-6E read-path delta between the two SHAs) |
| **HEAD SHA at PostgreSQL execution** | `fe79e01769853eb3655bfd90e9f26e1be6e1dcdd` |
| **origin/development SHA at PostgreSQL execution** | `fe79e01769853eb3655bfd90e9f26e1be6e1dcdd` |
| **HEAD == origin/development at execution** | **yes** |
| **worktree clean at PostgreSQL execution** | **no** — unrelated local modifications outside collaborative-work / qualification scope; **all MP-6E read qualification paths matched `qualification_sha` with no local diff** |
| **branch** | `development` |

> **Evidence vs qualification:** This file may be updated in a later commit (`evidence_update_commit_sha`). That commit is **not** the `qualification_sha`. Auditors must bind results to `qualification_sha` above.

## 3. Provider environment

| Field | Value |
| --- | --- |
| **provider** | PostgreSQL |
| **version** | 16.6 (Debian 16.6-1.pgdg120+1) |
| **host** | `localhost:5434` |
| **database** | `intergrax` |
| **container/runtime** | Docker — `infra/docker/postgresql/docker-compose.yml`, service `postgresql`, image `postgres:16.6` |
| **DSN credentials** | **REDACTED** — `INTERGRAX_COLLABORATIVE_WORK_POSTGRESQL_DSN`; canonical local pattern `postgresql://***@localhost:5434/intergrax` |
| **qualification schema** | Per-test isolated schema `collaborative_work_test_<uuid>` |

## 4. Qualification command (exact)

```powershell
$env:INTERGRAX_COLLABORATIVE_WORK_POSTGRESQL_DSN = "postgresql://***@localhost:5434/intergrax"
uv run pytest tests/integration/collaborative_work/test_postgresql_collaborative_activity_read_store.py -m "integration and network" -v --tb=short
```

## 5. Execution integrity

| Field | Value |
| --- | --- |
| **pytest invocations (PostgreSQL qualification)** | 1 |
| **retries** | 0 |
| **xdist** | no |
| **sharding** | no |

## 6. PostgreSQL qualification result (exact)

```text
passed: 3
skipped: 0
xfailed: 0
failed: 0
```

### 6.1 Executed tests (exact function names at `qualification_sha`)

| Test function | Role |
| --- | --- |
| `test_postgresql_collaborative_activity_read_port_contract` | Shared read-port conformance (first/next page, boundary, empty, no duplicate/skip, concurrent append between pages, limit semantics) |
| `test_postgresql_collaborative_activity_read_isolation` | Workspace + tenant isolation |
| `test_postgresql_collaborative_activity_read_late_occurred_at_ordering` | `append_position` ordering independent of late `occurred_at` |

## 7. Semantics confirmed

| Topic | Evidence |
| --- | --- |
| **Authorization before provider** | authorization is upstream of provider; provider executes authorized scoped query only. MP-6E unit gates (`test_mp6e_deny_zero_provider_calls`, `test_mp6e_policy_failure_zero_provider_calls`, `test_mp6e_cross_tenant_request_denied_before_store`, `test_mp6e_invalid_cursor_zero_provider_calls`, composition custom-policy deny) |
| **Keyset pagination** | `append_position > cursor`, `ORDER BY append_position ASC`, `LIMIT n+1` |
| **No OFFSET** | `compile_collaborative_activity_read_query` + `test_mp6e_c1_read_store_sql_uses_keyset_without_offset` |
| **Scope-bound cursor** | Shared invalid/scope mismatch contracts + MP-6E service gates |
| **Provider parity** | Same `collaborative_activity_read_port_contract` helpers for SQLite (unit) and PostgreSQL (integration) |
| **Forward live feed** | New appends after page N may appear on later pages; documented in `COLLABORATIVE_WORK.md` MP-6E read semantics |
| **Read transaction isolation** | READ COMMITTED / statement-level reads; no fixed cross-page snapshot |

## 8. Composition correction (MP-6E-C1)

`build_collaborative_activity_read_service` injects `CollaborativeActivityReadAuthorizationPolicy | None` (contract Protocol); default policy is construction fallback only (`build_default_collaborative_activity_read_authorization_policy`).

## 9. Production code delta for qualification

**NONE** for MP-6E-C1-Q1 requalification. Prior MP-6E-C1 composition typing correction remains at `mp6e_c1_correction_sha`.

## 10. Out of scope (not certified)

- Read stack redesign, cursor redesign, authorization redesign, provider SQL redesign
- PostgreSQL provider-side authorization (authorization remains upstream)
- Multi-region / HA / production load

## 11. Status transition (on PASS)

```text
MP-6E-C1-Q1 — CLOSED
MP-6E-C1 — CLOSED / RECERTIFIED
MP-6E — CLOSED / RECERTIFIED
MP-6F — NEXT
MP-6 — IN PROGRESS
```

**BLOCKING FINDINGS: NONE**
