# EQO-PKG-A4-027 — implementation evidence

## Base SHA

`a0598ba3e4f4f903eb4da3db8ea012d5cbf71225` (`development`). The same behavior was first committed on `eqo/w2-a4-027` at `06604af5347883dd26c98d54e72e49f2f6f58013` against train `8e146e21`. Cherry-pick of those commits conflicted on ticket files. This branch replays only the behavior onto `development`.

## Root cause

`EQOA-A4-027` / `EDER-F-0136`: missing `tenant_id` on research serving was filled with `research-tenant`. Follow-up: whitespace-only `tenant_id` passed `Field(min_length=1)` and `if not body.tenant_id` in the unauthenticated router path.

## Write set

- `applications/research_application/serving/schemas.py` — required `tenant_id` and `user_id` (no defaults); `tenant_id` rejected when blank after strip.
- `applications/research_application/serving/fastapi_router.py` — unauthenticated path strips `tenant_id` and returns `tenant_id_required` when empty (including `model_construct` bypass).
- `tests/unit/applications/test_research_run_request_missing_tenant_id_eqo_a4_027.py`
- `governance/tickets/T-IGX-055-B.md`

## Ticket

`governance/tickets/T-IGX-055-B.md`

## Pytest

Command:

```text
C:\Users\XPS\.local\bin\uv.exe run --extra dev-ci pytest tests/unit/applications/test_research_run_request_missing_tenant_id_eqo_a4_027.py --basetemp D:/Projekty/_wt/intergrax/_scratch/eqo-dev-a4027/pytest
```

Result: **1 passed**, exit 0, on branch `eqo/dev-a4-027` cut from `a0598ba3`.

## Scoped limitation

Other tenant or user defaults elsewhere in the repository were not surveyed; only the research serving write set was changed. `T-IGX-052-A.md` on `development` is a different ticket and is not modified.
