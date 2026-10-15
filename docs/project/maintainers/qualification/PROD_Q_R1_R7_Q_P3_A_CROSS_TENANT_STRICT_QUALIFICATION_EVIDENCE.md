# PROD-Q-R1-R7-Q-P3-A — Cross-tenant strict production qualification evidence

| Field | Value |
|-------|-------|
| **Status** | **READY FOR AUDIT** (evidence package); qualification outcome **FAIL** on HTTP tenant isolation |
| **Child** | **PROD-Q-R1-R7-Q-P3-A** |
| **Parent** | **PROD-Q-R1-R7-Q-P3** — **NOT CLOSED** |
| **Branch** | `development` |
| **START_HEAD** | `5d6d688fb006807beca151599210fc14b5510956` |
| **Execution SHA** | _(record at commit push)_ |

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
| Execution governance admission (HTTP) | `admit_harness_root_governance_identity` — `intergrax/applications/_shared/harness_admitted_root_governance_identity.py` (maps **request** `task.tenant_id`) |
| Host composition tenant for topology | `factory.py` → `manifest.app_id` via `build_harness_host_runtime` |

**Authoritative tenant (activation):** principal `RequestIdentity.tenant_id` vs `ApplicationEnvironmentTenantResolver.resolve_tenant_id` at `authorize_scoped_control_plane_mutation` / `TenantScopedControlPlaneMutationEvaluator`.

**HTTP request tenant:** body field `tenant_id` on `GovernedContractorRunRequestV1` — **not** authenticated; admitted via harness identity mapper unless a separate gate rejects mismatch.

## Scenarios

### P3-T-01 — Cross-tenant activation / serving

- **Tenant A:** `p3a-tenant-a` (principal)
- **Tenant B:** `p3a-tenant-b` (environment resolver)
- **Scope:** `app_a` / `env-prod`, revision `rev-p3a-t01-cross`
- **Rejection boundary:** `TenantScopedControlPlaneMutationEvaluator` → `tenant_authority_mismatch` before `deploy_and_activate` commit
- **Physical:** no new traffic revision; adapter unit count unchanged when activation never authorized

### P3-T-02 — Wrong-tenant HTTP

- Legitimate strict session under governance B (`p3a-tenant-b`)
- HTTP POST `/run` with `tenant_id=p3a-tenant-a` vs positive control `tenant_id=app_a` (manifest host tenant)
- **Observed:** wrong-tenant request **invoked** `TaskBoundAgenticDelegate.execute` (colocated admission acquired; no canonical tenant deny before inner execution)
- **Rejection boundary expected:** fail-closed compare of request tenant to serving-session authority — **not present** on strict GCA HTTP path @ HEAD

## Test command and results

```text
uv run pytest -p no:xdist -q tests/qualification/prod_q/test_prod_q_r1_r7_q_p3_a_cross_tenant_strict.py
```

| Test | Result |
|------|--------|
| `test_p3a_g00_qualification_baseline_ancestor` | PASS |
| `test_p3t01_cross_tenant_activation_blocked_before_handoff` | PASS |
| `test_p3t01_legitimate_tenant_b_state_unchanged_after_cross_tenant_attempt` | PASS |
| `test_p3t02_wrong_tenant_http_does_not_invoke_inner_execution` | **FAIL** — inner delegate called on wrong tenant |

**Summary:** 3 passed, 1 failed (~15s).

## Real / stub boundary

| Real | Stub |
|------|------|
| Strict session, VENV_BUNDLE adapter, governance resolver, `deploy_and_activate` authz, handoff, store-backed admission, strict ASGI factory | Terminal LLM; `TaskBoundAgenticDelegate.execute` mock for execution proof |

No monkeypatch of tenant validator or Governance under test.

## Enterprise architecture audit (local contribution)

| Criterion | P3-T-01 | P3-T-02 |
|-----------|---------|---------|
| Tenant isolation | **PASS** | **FAIL** |
| Layer boundaries (Gov vs execution) | **PASS** at control-plane | **FAIL** — HTTP metadata drives admitted identity |
| Semantic ownership | Governance owns activation tenant | Execution harness owns HTTP admit @ HEAD |
| Fail-closed | **PASS** | **NOT PROVEN** |
| No shadow authorization | **PASS** | N/A |
| Physical admission without authz | N/A | **FAIL** — admission + inner invoke |
| Regression protection | Tests added | Tests fail until corrective child |

## FRZ mapping (contribution only — globals unchanged)

| ID | P3-T-01 | P3-T-02 |
|----|---------|---------|
| **FRZ-SEC-01** | PASS | FAIL |
| **FRZ-SEC-04** | PASS | FAIL |
| **FRZ-PRD-08** | PASS | FAIL |
| **FRZ-PRD-09** | PASS (no rogue READY) | PASS (session stable) |
| **FRZ-REG-02/03/09** | Evidence + tests | Evidence records FAIL |

## Open blockers and next child

**ARCHITECTURE DECISION REQUIRED** — strict GCA HTTP execution must bind admitted root governance identity (and admission) to authoritative serving-session tenant (control-plane / manifest host tenant), not unauthenticated `GovernedContractorRunRequestV1.tenant_id` via `admit_harness_root_governance_identity`.

**Mandatory corrective child (bounded):** e.g. **PROD-Q-R1-R7-Q-P3-A1** — production admitted-identity port for strict serving HTTP (no ad-hoc route string compare); re-run P3-T-02 module.

**P3-A overall:** **FAIL** for end-to-end tenant isolation; **READY FOR AUDIT** of this evidence commit.
