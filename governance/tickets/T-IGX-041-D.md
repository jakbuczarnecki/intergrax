# T-IGX-041-D — PDC-X batch B-P7-D4: repair of the IA-D4 audit findings for domain D4 (HOST-RUNTIME), loop 1

## Scope

Repair ticket for the independent audit `PDC_P7_D4_AUDIT_REPORT.md` (IA-D4, audited SHA `c08efe46a74a20595468d8d8bf06eab2dca5f573`, verdict PARTIAL), under Operator option B of `DELEGATION-PDC-P7-REPAIR.md`. In scope: F-D4-01 (cancellation during `start()`, `run_until_stopped()` and `stop()` rolls back lease, components and runtime), F-D4-04 (the `runtime.ready()` probe is bounded at refresh and at startup), F-D4-07 (the public-API fingerprint covers model config, field constraints and validator names; manifest version 1.0.1), F-D4-09 (TL-FIX-C status and the isolation bound reconciled in the architecture, the plan and three satellite rows), F-D4-05 gate part (a gate that pins the composition root and declares the lab MCP tool wiring as the one exception), and by-products F-D4-12 (bounded event and metric publishes in the engine dispatcher and the supervisor), F-D4-13 (all six metrics through the export bridge, the documented opt-in), F-D4-19 (b) (three symlink tests skip only when symlink creation is not permitted, with a stubbed complement), a platform-independent test of the POSIX lock branch, and the crash-test docstring of F-D4-03. Excluded with reason and reopen condition (plan section "Excluded findings"): F-D4-02 (no production caller of the drain/flush seam, F-04 family), F-D4-03 product claim (no process watcher in the product), the re-routing of the lab MCP tools (F-D4-05), F-D4-06 (the tenant arguments of the hosted sandbox session are unread labels; per-tenant sessions are a new capability), F-D4-08 (Operator decision on the ownership of `business_agents_deploy_enabled`), and the LOW/INFO findings F-D4-10, -11, -14, -15, -16, -17, -18, -19 (a)/(c), -20. Product paths: yes (`intergrax/hosting/engine/engine.py`, `intergrax/hosting/eventing.py`, `intergrax/hosting/supervisor/supervisor.py`, `intergrax/hosting/public_api.py`); no hotspot file (`intergrax/applications/_shared/`, `intergrax/contracts/`) is in the write set. This ticket is not a certification verdict and not a closure of D4; the product-path changes keep the IIAC activation stale (ACT-N11) until the P8-04 rebind and the EBH-7 identity gate red until the final re-pin. Plan: `PLAN_T-IGX-041-D.md`.

## Plan digest

Plan digest: 011a4656ac298920738db2f0435100642a47cfa6fb5f0ccca4131023dafb97a0

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-041-D.md`](plans/PLAN_T-IGX-041-D.md).

## Test contract

```text
uv run --extra mcp pytest tests/qualification/pdc_x/d4_repair tests/qualification/pdc_x/d4 tests/qualification/host_01 tests/unit/hosting -q
uv run ruff check intergrax/hosting/engine/engine.py intergrax/hosting/eventing.py intergrax/hosting/supervisor/supervisor.py intergrax/hosting/public_api.py tests/qualification/pdc_x/d4_repair tests/qualification/pdc_x/d4/test_d4_crash_restart_subprocess.py tests/unit/hosting/instance
uv run pyright intergrax/hosting/engine/engine.py intergrax/hosting/eventing.py intergrax/hosting/supervisor/supervisor.py intergrax/hosting/public_api.py tests/qualification/pdc_x/d4_repair
uv run python scripts/qualification/pdc_x_wbs.py --check
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

- All excluded findings listed in the plan; the product active-work adapter and runner seam, a product process watcher, the re-routing of the lab MCP tools, per-tenant sandbox sessions, the ownership decision for `business_agents_deploy_enabled`, CI and impact-map wiring, the requirement matrix.
- `scripts/governance/`, `.github/`, `intergrax/contracts/`, every `intergrax/applications/_shared/` file, `intergrax/hosting/instance/`, `intergrax/hosting/runner.py`, `intergrax/hosting/shutdown.py`, the TENANT-X carrier re-pin (T-IGX-041-I), the EBH-7 re-pin, the IIAC activation record and rebind, FRZ promotion, OD signatures, certification.

## Lifecycle state

plan_pending
