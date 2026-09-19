# MP-6E-C1 — PostgreSQL Read Provider Qualification (Collaborative Activity Read Port)

## 1. Task and qualification status

| Field | Value |
| --- | --- |
| **task** | MP-6E-C1 — contract-driven read wiring, PostgreSQL read qualification, documentation repair |
| **qualification status** | **PASS** — live PostgreSQL read provider qualification executed (session log under `.tmp/session/MP-6E-C1/`) |
| **MP-6E-C1** | **CLOSED** (subject to independent GitHub audit of this artifact) |
| **MP-6E** | **CLOSED / RECERTIFIED** (scoped read; SQLite + PostgreSQL read providers qualified on live integration DB) |
| **bounded scope** | PostgreSQL 16.6 single-node integration environment; per-test isolated schema; not multi-region, HA, or production load certification |

## 2. Evidence identity

| Field | Value |
| --- | --- |
| **implementation_sha** | `eb586ccefea80ff05e0dd4f4f1e784ed1c398350` — MP-6E scoped read stack (`feat(collaborative-work): add mp6e scoped activity read`) |
| **qualification_execution_base_sha** | `509e8b5394b1adb8c1dec7aed217dacbdc82231f` — repository `HEAD` at live PostgreSQL read qualification execution (pre–MP-6E-C1 commit) |
| **branch** | `development` |

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

## 5. PostgreSQL qualification result (exact)

```text
passed: 3
skipped: 0
xfailed: 0
failed: 0
```

### 5.1 Executed tests

| Test function | Role |
| --- | --- |
| `test_postgresql_collaborative_activity_read_port_contract` | Shared read-port conformance (first/next page, boundary, empty, no duplicate/skip, concurrent append between pages, limit semantics) |
| `test_postgresql_collaborative_activity_read_isolation` | Workspace + tenant isolation |
| `test_postgresql_collaborative_activity_read_late_occurred_at_ordering` | `append_position` ordering independent of late `occurred_at` |

## 6. Semantics confirmed

| Topic | Evidence |
| --- | --- |
| **Authorization before provider** | MP-6E unit gates (`test_mp6e_deny_zero_provider_calls`, composition custom-policy deny) |
| **Keyset pagination** | Shared contract suite — `append_position > cursor`, `ORDER BY append_position ASC`, `LIMIT` |
| **No OFFSET** | `compile_collaborative_activity_read_query` + `test_mp6e_c1_read_store_sql_uses_keyset_without_offset` |
| **Scope-bound cursor** | Shared invalid/scope mismatch contracts + MP-6E service gates |
| **Provider parity** | Same `collaborative_activity_read_port_contract` helpers for SQLite (unit) and PostgreSQL (integration) |
| **Forward live feed** | Documented in `COLLABORATIVE_WORK.md` MP-6E read semantics — no cross-page snapshot guarantee |
| **Read transaction isolation** | Default PostgreSQL read-committed per statement; pages are not a single snapshot unless a future snapshot token is introduced |

## 7. Composition correction (MP-6E-C1)

`build_collaborative_activity_read_service` injects `CollaborativeActivityReadAuthorizationPolicy | None` (contract Protocol); default policy is construction fallback only (`build_default_collaborative_activity_read_authorization_policy`).

## 8. Production code delta for qualification

| Path | Change |
| --- | --- |
| `intergrax/collaborative_work/collaborative_activity_composition.py` | Contract typing for read authorization policy parameter only |

No changes to read service, evaluator, cursor codec, or provider SQL semantics were required for PostgreSQL PASS.

## 9. Status transition (on PASS)

```text
MP-6E-C1 — CLOSED
MP-6E — CLOSED / RECERTIFIED
MP-6F — NEXT
MP-6 — IN PROGRESS
```
