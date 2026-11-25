# T-IGX-055-B — EQOA-A4-027: research serving rejects missing tenant_id (no research-tenant default)

## Scope

Remove silent `research-tenant` / `research-user` defaults from research HTTP serving. `ResearchRunRequestV1` requires explicit `tenant_id` and `user_id`. The research router must not substitute a tenant when there is no authenticated principal. Maps to EQO-PKG-A4-027 / `EDER-F-0136`.

Write set authorized under the EQO-PKG-A4-027 grant from the 2026-11-21 orchestration session. Required behavior is fixed by DP-23 and C3 (missing `tenant_id` is rejected; no silent research-tenant default).

Write set only:

- `applications/research_application/serving/schemas.py`
- `applications/research_application/serving/fastapi_router.py`
- `tests/unit/applications/test_research_run_request_missing_tenant_id_eqo_a4_027.py`
- `docs/project/maintainers/qualification/enterprise_quality_oversight/eqo_closure/orchestration/packets/EQO-PKG-A4-027/IMPLEMENTATION_EVIDENCE.md`
- `governance/tickets/T-IGX-055-B.md` (this ticket)

## Plan digest

Plan digest: n/a (EQO WAVE-02 package EQO-PKG-A4-027; behavior fixed by DP-23 and C3).

## Test contract

`uv run pytest tests/unit/applications/test_research_run_request_missing_tenant_id_eqo_a4_027.py --basetemp D:/Projekty/_wt/intergrax/_scratch/eqo-w2-a4027/pytest`

## Out of scope

Other applications with `default` tenant literals; `harness_host_runtime.py`; FRZ-TEN-10 promotion; `development` merge; host factory paths; matrix or register lifecycle updates.

## Lifecycle state

in_progress