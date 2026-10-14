# PROD-Q-R1-R7-D4 — Strict GCA production launcher / ASGI admission

| Field | Value |
|-------|-------|
| **Stage** | PROD-Q-R1-R7-D4 |
| **Parent** | PROD-Q-R1-R7 |
| **Successor** | PROD-Q-R1-R7-Q |
| **Cursor disposition** | **READY FOR AUDIT** (not CLOSED) |
| **Evidence HEAD** | `9b6efa8dc3993a3d00460cf1dc45d6bdabe65c72` |

## Sanctioned entry

1. `open_strict_production_serving_session` — `intergrax/applications/_shared/strict_production_serving_launcher.py`
2. `create_governed_contractor_strict_production_asgi_app` / `build_governed_contractor_strict_production_uvicorn_factory` — `applications/governed_contractor_application/host/strict_production_serving.py`
3. GCA `main.run()` remains fail-closed; operator uses external ASGI + factory.

## Startup order (Decision C)

D2 `open_production_process_lifecycle` → D3 reconciliation → optional D2 `deploy_and_activate` → `bind_colocated_serving_runtime_after_activation` → GCA `create_governed_contractor_process_app` → external ASGI.

## Tests (unit, `-p no:xdist`)

`tests/unit/applications/governed_contractor/test_prod_q_r1_r7_d4_strict_production_gca_launcher.py` (12 cases).

Regression slice @ same HEAD: D4 + R5 + R6 + D2 + D3 + D1b + architecture gate — **51 passed**.

## FRZ mapping (evidence only — global OPEN)

- FRZ-PRD-04/05/07/08/09/10/12 — launcher lifecycle, fail-closed, strict/reference separation, readiness struct, shutdown via ASGI lifespan + session close.
