# PROD-Q-R1-R7-Q-P3-INTEGRATION-GATE — mandatory IG-E01 / IG-F01 qualification evidence

| Field | Value |
|-------|-------|
| **Status** | **READY FOR AUDIT — TESTS NOT EXECUTED** |
| **Parent** | **PROD-Q-R1-R7-Q-P3** — **NOT CLOSED** (parent status unchanged) |
| **Branch** | `development` |
| **START_HEAD** | `16f48d0fc545bb114e0579c8e4591a506991ee87` |
| **Source SHA (pre-commit)** | `16f48d0fc545bb114e0579c8e4591a506991ee87` |
| **Pytest executed** | **0** |
| **Production code delta** | **0** |

**Wprowadzone zmiany muszą zostać niezależnie zaudytowane na podstawie kodu z commitu znajdującego się na GitHubie. Raport Cursor AI nie jest podstawą finalnego zamknięcia zadania.**

## 1. Preflight (@ START_HEAD)

| Check | Result |
|-------|--------|
| Branch | `development` |
| HEAD vs **EXPECTED_START_HEAD** | **match** @ `16f48d0fc545bb114e0579c8e4591a506991ee87` |
| Working tree @ task start | **clean** |
| Canonical docs | Roadmap / freeze checklist / R7-Q gate plan / P3-P0 / P3-FINAL matrix / P3-E & P3-F evidence / DISPATCH-Q1 / ADR-03 / P3-A2 ADR — read for scope alignment (no FRZ promotion) |
| **PROD-Q-R1-ADR-03** | **RATIFIED** |
| **PROD-Q-R1-DISPATCH-Q1** | **ACCEPTED (scoped)** — prerequisite for IG-F01 chain |
| **P3-A2 ADR** | **PROPOSED** — not ratified here |

## 2. Implemented nodes (NOT EXECUTED)

| Node ID | Module |
|---------|--------|
| **IG-E01** | `tests/qualification/prod_q/test_prod_q_r1_r7_q_p3_integration_gate.py::test_ig_e01_strict_activation_intent_fail_prepare_no_handoff` |
| **IG-F01** | `tests/qualification/prod_q/test_prod_q_r1_r7_q_p3_f_strict_repeat_dispatch_negative.py::test_strict_session_external_work_no_second_provider_mutation_same_invocation_id` |

Both nodes: **IMPLEMENTED — NOT EXECUTED — AWAITING OPERATOR AUTHORIZATION**.

## 3. IG-E01 — boundary and expectations

**Boundary:** `open_strict_production_serving_session` → `activation_intent` → `ProductionLifecycleLauncher.deploy_and_activate` → `ActivationService.prepare_candidate` → real `VenvBundleColocatedRuntimeDeploymentAdapter.prepare` (tampered `materialization_artifact_digest` vs on-disk bundle).

**Real:** strict opener, D2/D4 lifecycle storage, SQLite AD stores, A1 adapter, governance wiring, canonical bundle seeding (`test_prod_q_r1_r7_d4_strict_production_gca_launcher` helpers).

**Not mocked:** `prepare_candidate`, `deploy_and_activate`, `commit_activation`, `open_strict_production_serving_session`.

**Expected negative outcomes:** exception before session return (`VenvBundleColocatedDeploymentError` and/or `StrictProductionServingLauncherError`); no returned `StrictProductionServingSession` for failed intent; traffic pointer unchanged; prior good revision still opens traffic-ready session with valid `ServingRuntimeHandoff` and colocated admission.

**Gap:** does not assert every partial durable row / FAILED instance field (bounded to traffic pointer + handoff continuity).

## 4. IG-F01 — identity and boundary

**Boundary:** `open_production_process_lifecycle` (strict production wiring: Venv A1 adapter, lifecycle-owned `provider_invocation_db`) → `composition.provider_invocation_store` (`SQLiteProviderInvocationStore`) → `build_governed_external_work_production_runtime` → `ExternalWorkAdapter.create_and_map` → `_tracked_execute` → `GovernedProviderInvocationDispatchGate`.

**Invocation identity:** fixed `ProviderInvocation.invocation_id` (`inv-ig-f01-fixed-identity`) passed explicitly on both attempts (sanctioned adapter API — **not** HTTP orchestrator re-entry).

**Expected:** first dispatch inserts intent + exactly one `DeterministicExternalWorkFake` mutation; intent remains without outcome; second equivalent dispatch fails with `dispatch_not_admitted`; provider mutation count stays **1**.

**Coverage class:** **strict composition-level** — **HTTP / ASGI re-entry with stable invocation_id: NOT PROVEN**.

## 5. Static review (pre-execution)

| Check | IG-E01 | IG-F01 |
|-------|--------|--------|
| Unauthorized mocks of production gates / opener / prepare | **none** | **none** |
| Bypass of governance or dispatch gate | **none** | **none** |
| Test-only dispatch path | **none** | uses production factory only |
| Correct invocation identity | N/A | fixed ID, same object semantics |
| Provider counter | N/A | `DeterministicExternalWorkFake.create_calls` |
| External network / real provider | **none** | **none** |
| Fixture cost | D4 bundle materialization ×2 + one failed strict open + one successful strict open | lifecycle open + external-work composition (no full strict activate) |

## 6. Dependencies and expected qualification cost

| Node | Heavy dependencies | Historical class |
|------|-------------------|------------------|
| **IG-E01** | D4 `_qualified_canonical_bundle`, dual lifecycle open, strict opener (activate + reconcile) | D4 family — matrix **≤5 min** (not measured @ this HEAD) |
| **IG-F01** | SQLite PIS + GR-6/GR-7 production external-work stack | PIS-01 / GR-7-A8 class — matrix **≤5 min** (not measured @ this HEAD) |

## 7. Git report (post-push)

| Field | Value |
|-------|--------|
| **FINAL_COMMIT** | *(see session report after push)* |
| **Files added** | Two qualification tests + this evidence record |

## 8. Operator authorization

Allowlisted pytest (when authorized):

```text
tests/qualification/prod_q/test_prod_q_r1_r7_q_p3_integration_gate.py::test_ig_e01_strict_activation_intent_fail_prepare_no_handoff
tests/qualification/prod_q/test_prod_q_r1_r7_q_p3_f_strict_repeat_dispatch_negative.py::test_strict_session_external_work_no_second_provider_mutation_same_invocation_id
```
