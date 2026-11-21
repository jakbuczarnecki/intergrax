# EQO-PKG-A4-027 — implementation evidence

## Base SHA

`8e146e21cd8c63a47d621edc05d5f0036ba0c9ee` (`origin/arch-freeze/rw-train`)

## Root cause

`EQOA-A4-027` / `EDER-F-0136`: missing `tenant_id` on research serving was filled with `research-tenant`.

## Write set

- `applications/research_application/serving/schemas.py` — required `tenant_id` and `user_id` (no defaults).
- `applications/research_application/serving/fastapi_router.py` — unauthenticated path does not imply a tenant.
- `tests/unit/applications/test_research_run_request_missing_tenant_id_eqo_a4_027.py`
- `governance/tickets/T-IGX-052-A.md`

## Ticket

`governance/tickets/T-IGX-052-A.md`

## Pytest

Command:

```text
C:\Users\XPS\.local\bin\uv.exe run pytest tests/unit/applications/test_research_run_request_missing_tenant_id_eqo_a4_027.py --basetemp D:/Projekty/_wt/intergrax/_scratch/eqo-w2-a4027/pytest
```

Result: **TEST_NOT_RUN** — `D:/Projekty/_wt/_locks/pytest.lock` already held; pytest was not started (global lock policy).

## Scoped limitation

Other tenant or user defaults elsewhere in the repository were not surveyed; only the research serving write set was changed.