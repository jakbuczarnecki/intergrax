# MP-6G — E2E / Isolation / Idempotency Qualification

**Status:** MP-6G — QUALIFICATION IN PROGRESS / **POSTGRESQL QUALIFICATION PENDING** (live PG skipped: backend unavailable in agent session)

## Identity

| Field | Value |
|-------|-------|
| baseline_sha | `31ef13456aff1753313a4dcd5c0933f79be0561a` |
| implementation_test_sha | `fefd4bf9b2317ebe0f451c37fce8b1db1972203f` |
| qualification_sha | _pending clean HEAD + live PG pass_ |
| evidence_update_sha | _this document commit_ |

## Commands

```text
uv run pytest tests/qualification/mp6/test_mp6g_e2e_qualification.py tests/qualification/mp6/test_mp6g_architecture_gates.py -q
uv run pytest tests/integration/collaborative_work/test_mp6g_postgresql_e2e_qualification.py -m "integration and network" -v --tb=short
```

## Execution integrity

| Field | SQLite | PostgreSQL |
|-------|--------|------------|
| pytest invocations | _pending_ | _pending_ |
| retries | 0 | 0 |
| xdist | no | no |
| passed / skipped / xfailed / failed | 4 / 0 / 0 / 0 | 0 / 1 / 0 / 0 (backend unavailable) |

## Source E2E matrix (initial harness)

| Source flow | SQLite E2E | PostgreSQL E2E | Replay | Isolation | Readback | Status |
|-------------|------------|----------------|--------|-----------|----------|--------|
| WorkItem create | yes | pending | yes | yes | yes | IN PROGRESS |
| WorkItem transition | yes | pending | — | — | yes | IN PROGRESS |
| Assignment create | contract TBD | pending | — | — | — | PLANNED |
| WorkArtifact create/version | contract TBD | pending | — | — | — | PLANNED |
| Decision binding | contract TBD | pending | — | — | — | PLANNED |
| ContextView compose | contract TBD | pending | — | — | — | PLANNED |

_Update this artifact after green SQLite + live PostgreSQL runs with exact counts and SHAs._
