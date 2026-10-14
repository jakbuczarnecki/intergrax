# PROD-Q-R1-R7-D4 — Strict GCA production launcher / ASGI admission

| Field | Value |
|-------|-------|
| **Stage** | PROD-Q-R1-R7-D4 |
| **Parent** | PROD-Q-R1-R7 |
| **Successor** | PROD-Q-R1-R7-Q |
| **Cursor disposition** | **READY FOR INDEPENDENT CLOSURE** (not CLOSED) |
| **Implementation baseline** | `9b6efa8dc3993a3d00460cf1dc45d6bdabe65c72` |
| **Evidence HEAD** | `0dd51fe1c7fdd9f4f93d4abdf26f89bea879709e` |
| **D4-R1 corrective** | **Independently accepted** @ `339ffc12aa13d9c6f72b031e9c3b64cd52971787` — D4-03 proves `main.run()` fail-closed via public `SystemExit(1)` + stderr only; **rejects** prior `b3d9aced0f255d249812622647a356c0b91d76f8` D4-03 private `_active_production_process_lifecycle` inspection (test-boundary violation). **Remains valid** on evidence HEAD chain. **Does not** autonomously close **PROD-Q-R1-R7-D4** or **PROD-Q-R1-R7**. |
| **D4-R2 corrective** | **Independently accepted** (audit disposition **PASS**) — implementation `b4336eb03a9f2d6eabed0edbbe48526c33a07957`; evidence HEAD `0dd51fe1c7fdd9f4f93d4abdf26f89bea879709e` (test-boundary: slot release via public `open_production_process_lifecycle` re-open, not private `production_process_lifecycle` module state). **Does not** close **PROD-Q-R1-R7**, **PROD-Q**, or global **FRZ** rows. |

## Sanctioned entry

1. `open_strict_production_serving_session` — `intergrax/applications/_shared/strict_production_serving_launcher.py`
2. `create_governed_contractor_strict_production_asgi_app` / `build_governed_contractor_strict_production_uvicorn_factory` — `applications/governed_contractor_application/host/strict_production_serving.py`
3. GCA `main.run()` remains fail-closed; operator uses external ASGI + factory.

## Startup order (Decision C)

D2 `open_production_process_lifecycle` → D3 reconciliation → optional D2 `deploy_and_activate` → `bind_colocated_serving_runtime_after_activation` → GCA `create_governed_contractor_process_app` → external ASGI.

Explicit activation intent scope is validated **before** `open_production_process_lifecycle` (`strict_production_serving_launcher.py`).

## D4-R2 root cause (accepted corrective)

| Item | Detail |
|------|--------|
| **Finding** | After lifecycle open, strict serving session startup failure path called `close_process_resources()` but **swallowed** cleanup exceptions, masking operator-visible dual failures (**FRZ-PRD-10** local). Initial D4-R2 regression for slot release used **private** `production_process_lifecycle` module state (same test-boundary class as superseded D4-03 @ `b3d9aced…`). |
| **Production fix** | `open_strict_production_serving_session` @ `b4336eb03a9f2d6eabed0edbbe48526c33a07957` — aggregate cleanup failures with primary startup error via `cleanup_errors_from_close_failure` / `raise_primary_with_cleanup_errors`; successful cleanup still releases slot for subsequent public lifecycle open. |
| **Test fix** | @ `0dd51fe1c7fdd9f4f93d4abdf26f89bea879709e` — `test_d4_r2_startup_failure_cleanup_success_preserves_primary_and_releases_slot` proves slot availability via **public** `open_production_process_lifecycle` context manager only. |

## Regression evidence (D4-R1 + D4-R2)

| ID | Evidence |
|----|----------|
| **D4-03** | `test_d4_03_legacy_main_run_fail_closed` @ `339ffc12…` (unchanged on HEAD chain) — `main.run()` → `SystemExit(1)`; stderr cites sanctioned D4 launcher/factory; patched `open_production_process_lifecycle` not called (**public** contract only). |
| **D4-21** | `test_d4_21_activation_intent_scope_mismatch_before_lifecycle` — intent revision scope ≠ `StrictProductionServingEnvironment` → `StrictProductionServingLauncherError` before lifecycle open (patched `open_production_process_lifecycle` uncalled). |
| **D4-R2-a** | `test_d4_r2_startup_failure_cleanup_success_preserves_primary_and_releases_slot` @ `0dd51fe…` — primary `StrictProductionServingLauncherError`; exactly one `close_process_resources`; subsequent public lifecycle open succeeds. |
| **D4-R2-b** | `test_d4_r2_startup_failure_cleanup_failure_preserves_both_errors` @ `b4336eb…` — `ExceptionGroup` with primary launcher error + cleanup `RuntimeError`; cleanup invoked once. |

**Regression batch (exact-SHA, pre-recorded):** `tests/unit/applications/governed_contractor/test_prod_q_r1_r7_d4_strict_production_gca_launcher.py` — **16** cases @ evidence HEAD `0dd51fe1c7fdd9f4f93d4abdf26f89bea879709e` (includes D4-03, D4-21, D4-R2-a/b).

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

`tests/unit/applications/governed_contractor/test_prod_q_r1_r7_d4_strict_production_gca_launcher.py` (16 cases including D4-03, D4-21, D4-R2-a/b).

## FRZ mapping (local evidence only — global OPEN)

| Criterion | Local D4 contribution |
|-----------|----------------------|
| FRZ-PRD-03 | Sanctioned strict production entry documented; legacy `main.run()` blocked (D4-03). |
| FRZ-PRD-04/05/07/08/09/10/12 | Launcher lifecycle, fail-closed, strict/reference separation, readiness struct, shutdown via ASGI lifespan + session close; **D4-R2** startup-failure cleanup aggregation + public slot-release proof @ `0dd51fe…`. |
| FRZ-SEC-01/04 | Fail-closed legacy entry; governance deny on activation (`test_d4_governance_deny_blocks_activation`). |
| FRZ-TEN-01/02/03/10/12 | Bounded PASS above; full tenant certification deferred to R7-Q. |
| FRZ-REG-02/03/09 | D4-03, D4-21, D4-R2-a/b regression tests in D4 unit module @ evidence HEAD `0dd51fe…`. |

No autonomous global FRZ PASS.

## D4 acceptance matrix (reconciled)

| Slice | SHA | Independent disposition | Local D4 blocker |
|-------|-----|-------------------------|------------------|
| Implementation | `9b6efa8dc3993a3d00460cf1dc45d6bdabe65c72` | prior audit scope | — |
| D4-R1 | `339ffc12aa13d9c6f72b031e9c3b64cd52971787` | **ACCEPTED** (test-boundary) | cleared |
| D4-R2 impl | `b4336eb03a9f2d6eabed0edbbe48526c33a07957` | **ACCEPTED** (production + regressions) | cleared |
| D4-R2 evidence HEAD | `0dd51fe1c7fdd9f4f93d4abdf26f89bea879709e` | **ACCEPTED** (test-boundary slot proof) | **none** (mandatory **R7-Q** remains outside D4) |

## Recommended stage status

**READY FOR INDEPENDENT CLOSURE** — D4-R1 @ `339ffc12…` and D4-R2 corrective chain @ `0dd51fe…` **independently accepted** per supplied audit outcomes; **no** formal operator **CLOSED** recorded. **PROD-Q-R1-R7-Q** remains mandatory (cross-tenant E2E, structural replaceability, recovery, physical provider evidence, strict-path fake reachability, production admission). **PROD-Q-R1-R7** / **PROD-Q** remain **NOT CLOSED**.
