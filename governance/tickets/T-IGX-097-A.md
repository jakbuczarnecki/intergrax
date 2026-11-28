# T-IGX-097-A — EC-ACCELERATION-02 Package B (EQO-CL-01 / EC-F16 FastAPI lifecycle)

## Scope

PRE_FCS_CONSOLIDATED_DEFECT_SETTLEMENT EC-ACCELERATION-02 Package B: replace legacy FastAPI `add_event_handler("shutdown", …)` registration in `intergrax/fastapi_core/app_factory.py` with native lifespan shutdown semantics; focused lifecycle regression; MCP lifespan composition preserved via existing Tier-3 `fastapi_mcp` helpers.

## Plan digest

Plan digest: n/a (EC-ACCELERATION-02 operator batch INTEGRAX | ENTERPRISE CAMPAIGN EQO | PRE-FCS | CONSOLIDATED DEFECT SETTLEMENT)

## Approved

Approved: operator-delegate @ 2026-11-28T16:05:00+01:00

Delegation: EC-ACCELERATION-02 operator execution authority (ChatGPT consolidated contract; Maximum Safe Batch 2.0 Package B)

## Write set

```text
intergrax/fastapi_core/app_factory.py
intergrax/applications/_shared/fastapi_mcp.py
tests/unit/api/fastapi_core/test_app_factory_lifecycle.py
tests/unit/applications/test_fastapi_mcp.py
governance/tickets/T-IGX-097-A.md
```

## Test contract

Focused lifecycle regression (HTTP-only, explicit/implicit adapter, explicit run_service, MCP-mounted); P2C 7/7; MCP P4 lifecycle; adjacent D4 serving lifecycle where affected; `uv run ruff check` on write set; `git diff --check`; `uv run python -m scripts.governance.igx_gov ticket`.

## Out of scope

Dependency version changes, monkeypatch compatibility shims, Tier-3 application wiring imports in fastapi_core, COMPAT-X/REPLAY-X owner decisions, formal G2/EQO_CLOSED.

## Lifecycle state

in_progress