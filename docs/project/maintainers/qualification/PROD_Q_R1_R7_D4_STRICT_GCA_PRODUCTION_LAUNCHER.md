# PROD-Q-R1-R7-D4 — Strict GCA production launcher / ASGI admission

| Field | Value |
|-------|-------|
| **Stage** | PROD-Q-R1-R7-D4 |
| **Parent** | PROD-Q-R1-R7 |
| **Successor** | PROD-Q-R1-R7-Q |
| **Cursor disposition** | **READY FOR AUDIT** (not CLOSED) |
| **Evidence HEAD** | `0724d658402d2d6a86e38277115b9186ac10435e` |

## Sanctioned entry

1. `open_strict_production_serving_session` — `intergrax/applications/_shared/strict_production_serving_launcher.py`
2. `create_governed_contractor_strict_production_asgi_app` / `build_governed_contractor_strict_production_uvicorn_factory` — `applications/governed_contractor_application/host/strict_production_serving.py`
3. GCA `main.run()` remains fail-closed; operator uses external ASGI + factory.

## Startup order (Decision C)

D2 `open_production_process_lifecycle` → D3 reconciliation → optional D2 `deploy_and_activate` → `bind_colocated_serving_runtime_after_activation` → GCA `create_governed_contractor_process_app` → external ASGI.

Explicit activation intent scope is validated **before** `open_production_process_lifecycle` (`strict_production_serving_launcher.py`).

## Regression evidence (D4-R1-EVIDENCE)

| ID | Evidence |
|----|----------|
| **D4-03** | `test_d4_03_legacy_main_run_fail_closed` — `main.run()` → `SystemExit(1)`, stderr cites sanctioned D4 launcher/factory; `open_production_process_lifecycle` not invoked; no active lifecycle slot. |
| **D4-21** | `test_d4_21_activation_intent_scope_mismatch_before_lifecycle` — intent revision scope ≠ `StrictProductionServingEnvironment` → `StrictProductionServingLauncherError` before lifecycle open (patched `open_production_process_lifecycle` uncalled). |

## Tenant isolation (bounded D4 review)

**Verdict: PASS** (local contract evidence; not R7-Q certification).

| Invariant | Owner / evidence |
|-----------|------------------|
| Canonical tenant resolution | `ProductionControlPlaneGovernance.environment_tenant_resolver` (`production_control_plane_governance_wiring.py`); injected at D2/D4 wiring, not invented by launcher. |
| Launcher does not substitute tenant identity | `StrictProductionServingEnvironment` carries only `application_id` + `application_environment_id`; no tenant field on launcher session types. |
| Scope consistency | Activation intent revision must match serving environment before lifecycle (`strict_production_serving_launcher.py`); reconciliation targets single `(app, env)` pair. |
| Governance retains mutation authority | `deploy_and_activate(..., principal=governance.principal)`; deny path: `test_d4_governance_deny_blocks_activation`. |
| No alternate GCA tenant authorization path | GCA strict path consumes validated handoff only; legacy `main.run()` fail-closed (D4-03). |

**R7-Q still required:** multi-tenant E2E, adversarial cross-tenant certification (`FRZ-TEN-*` global closure).

## Tests (unit, `-p no:xdist`)

`tests/unit/applications/governed_contractor/test_prod_q_r1_r7_d4_strict_production_gca_launcher.py` (14 cases including D4-03, D4-21).

## FRZ mapping (local evidence only — global OPEN)

| Criterion | Local D4 contribution |
|-----------|----------------------|
| FRZ-PRD-03 | Sanctioned strict production entry documented; legacy `main.run()` blocked (D4-03). |
| FRZ-PRD-04/05/07/08/09/10/12 | Launcher lifecycle, fail-closed, strict/reference separation, readiness struct, shutdown via ASGI lifespan + session close. |
| FRZ-SEC-01/04 | Fail-closed legacy entry; governance deny on activation (`test_d4_governance_deny_blocks_activation`). |
| FRZ-TEN-01/02/03/10/12 | Bounded PASS above; full tenant certification deferred to R7-Q. |
| FRZ-REG-02/03/09 | D4-03, D4-21 regression tests in D4 unit module. |

No autonomous global FRZ PASS.

## Recommended stage status

**READY FOR AUDIT** — independent D4-R1 evidence slice complete; D4 / R7 / global freeze remain open pending external audit and R7-Q.
