# PROD-Q-R1-R7-Q-P3-A — Cross-tenant strict production qualification evidence

| Field | Value |
|-------|-------|
| **Status** | **READY FOR AUDIT** (evidence package); qualification outcome **PASS** on P3-A1 corrective scope |
| **Child** | **PROD-Q-R1-R7-Q-P3-A** |
| **Parent** | **PROD-Q-R1-R7-Q-P3** — **NOT CLOSED** |
| **Branch** | `development` |
| **START_HEAD** | `16fe72228313d80cb9aaee1042338b63ecdeb7c3` |
| **Execution SHA** | `d9717e374` |

## Preflight

| Check | Result |
|-------|--------|
| Branch | `development` |
| START_HEAD match | **yes** @ `5d6d688fb006807beca151599210fc14b5510956` |
| Working tree @ run | clean (pre-edit) |
| **PROD-Q-R1-R7-Q-P3-P0** | **READY FOR AUDIT** (inventory) |
| **PROD-Q-R1-R7-Q-P2** | **CLOSED** (operator) |

## Code and contract inventory

| Symbol | Path |
|--------|------|
| `ApplicationEnvironmentTenantResolver` | `intergrax/agent_distribution/control_plane_governance.py` |
| `StaticApplicationEnvironmentTenantResolver` | same |
| `TenantScopedControlPlaneMutationEvaluator` | same (`tenant_authority_mismatch`) |
| `ControlPlaneMutationAuthorizationBoundary` | `intergrax/runtime/governance/control_plane_mutation_authorization.py` |
| `ProductionControlPlaneMutationAuthorizationBoundary` | `intergrax/applications/_shared/production_control_plane_governance_wiring.py` |
| `authorize_scoped_control_plane_mutation` | `control_plane_governance.py` |
| `ProductionLifecycleLauncher.deploy_and_activate` | `intergrax/applications/_shared/production_lifecycle.py` |
| `ActivationService` | agent_distribution activation pipeline (via `run_deploy_and_activate`) |
| `open_strict_production_serving_session` | `intergrax/applications/_shared/strict_production_serving_launcher.py` |
| `ServingRuntimeHandoff` / bind | `serving_runtime_handoff.py`, `colocated_serving_runtime_composition.py` |
| `StoreBackedColocatedServingUnitExecutionAdmission` | `colocated_serving_unit_execution_admission.py` |
| `create_governed_contractor_strict_production_asgi_app` | `applications/governed_contractor_application/host/strict_production_serving.py` |
| HTTP `/run` | `governed_contractor_application/serving/fastapi_router.py` |
| Execution governance admission (HTTP) | `admit_strict_production_serving_root_governance_identity` — `strict_production_serving_execution_tenant_authority.py`; gate `StrictProductionServingTenantAuthorityTaskExecutionPort` **before** colocated `admit_execution` |
| Host composition tenant for topology | `factory.py` → `manifest.app_id` via `build_harness_host_runtime` |

**Authoritative tenant (activation):** principal `RequestIdentity.tenant_id` vs `ApplicationEnvironmentTenantResolver.resolve_tenant_id` at `authorize_scoped_control_plane_mutation` / `TenantScopedControlPlaneMutationEvaluator`.

**HTTP request tenant:** body field `tenant_id` — **untrusted**; must equal `ServingRuntimeHandoff.identity.application_environment_id` (P2-C continuity); admitted identity uses `ProductionControlPlaneGovernance.principal`, not forged body `user_id`.

## Scenarios

### P3-T-01 — Cross-tenant activation / serving

- **Tenant A:** `p3a-tenant-a` (principal)
- **Tenant B:** `p3a-tenant-b` (environment resolver)
- **Scope:** `app_a` / `env-prod`, revision `rev-p3a-t01-cross`
- **Rejection boundary:** `TenantScopedControlPlaneMutationEvaluator` → `tenant_authority_mismatch` before `deploy_and_activate` commit
- **Physical:** no new traffic revision; adapter unit count unchanged when activation never authorized

### P3-T-02 — Wrong-tenant HTTP

- Legitimate strict session under governance B (`p3a-tenant-b`)
- HTTP POST `/run` with `tenant_id=p3a-tenant-a` vs positive control `tenant_id=env-prod` (`application_environment_id`)
- **Post P3-A1:** wrong-tenant rejected at `StrictProductionServingTenantAuthorityTaskExecutionPort` before colocated admission; inner delegate not called
- **Rejection boundary:** `assert_task_tenant_matches_strict_production_serving_authority` + strict admitted identity mapper

## Test command and results

```text
uv run pytest -p no:xdist -q tests/qualification/prod_q/test_prod_q_r1_r7_q_p3_a_cross_tenant_strict.py
```

| Test | Result |
|------|--------|
| `test_p3a_g00_qualification_baseline_ancestor` | PASS |
| `test_p3t01_cross_tenant_activation_blocked_before_handoff` | PASS |
| `test_p3t01_legitimate_tenant_b_state_unchanged_after_cross_tenant_attempt` | PASS |
| `test_p3t02_wrong_tenant_http_does_not_invoke_inner_execution` | **PASS** |

**P3-A1 regression (same module):** 4/4 passed (~22s with unit slice).

**P2-C HTTP regression:** `test_prod_q_r1_r7_q_p2_c_http_execution_admission.py` 6/6 passed (same run).

## Real / stub boundary

| Real | Stub |
|------|------|
| Strict session, VENV_BUNDLE adapter, governance resolver, `deploy_and_activate` authz, handoff, store-backed admission, strict ASGI factory | Terminal LLM; `TaskBoundAgenticDelegate.execute` mock for execution proof |

No monkeypatch of tenant validator or Governance under test.

## Enterprise architecture audit (local contribution)

| Criterion | P3-T-01 | P3-T-02 |
|-----------|---------|---------|
| Tenant isolation | **PASS** | **PASS** (P3-A1) |
| Layer boundaries (Gov vs execution) | **PASS** at control-plane | **PASS** — trusted handoff + governance principal |
| Semantic ownership | Governance owns activation tenant | Apps strict serving tenant authority owns HTTP gate |
| Fail-closed | **PASS** | **PASS** |
| No shadow authorization | **PASS** | **PASS** |
| Physical admission without authz | N/A | **PASS** — tenant gate before `admit_execution` |
| Regression protection | Tests added | P2-C replay green |

## FRZ mapping (contribution only — globals unchanged)

| ID | P3-T-01 | P3-T-02 |
|----|---------|---------|
| **FRZ-SEC-01** | PASS | PASS (local P3-A1) |
| **FRZ-SEC-04** | PASS | PASS (local P3-A1) |
| **FRZ-PRD-08** | PASS | PASS (local P3-A1) |
| **FRZ-PRD-09** | PASS (no rogue READY) | PASS (session stable) |
| **FRZ-REG-02/03/09** | Evidence + tests | Evidence records FAIL |

## Open blockers and next child

**P3-A1 corrective (bounded):** `strict_production_serving_execution_tenant_authority.py` + host wiring + tenant gate port; governance injected from `StrictProductionServingSession.lifecycle.governance`.

**P3-A overall:** **READY FOR AUDIT** — P3-T-01/T-02 green @ P3-A1 commit; extended T-03..T-08 matrix partially covered (unit + P2-C); full adversarial matrix remains parent **P3** scope.
