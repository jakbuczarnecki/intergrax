# PROD-Q-R1-R7-Q-P3-INTEGRATION-GATE — mandatory IG-E01 / IG-F01 qualification evidence

| Field | Value |
|-------|-------|
| **Status** | **READY FOR AUDIT — TESTS NOT EXECUTED** |
| **Parent** | **PROD-Q-R1-R7-Q-P3** — **NOT CLOSED** (parent status unchanged) |
| **Branch** | `development` |
| **START_HEAD** | `7a5f3524dc2648a8161db397d48d86a2d59b2410` |
| **Source SHA (pre-commit)** | `7a5f3524dc2648a8161db397d48d86a2d59b2410` |
| **Pytest executed** | **0** |
| **Production code delta** | **0** |

**Wprowadzone zmiany muszą zostać niezależnie zaudytowane na podstawie kodu z commitu znajdującego się na GitHubie. Raport Cursor AI nie jest podstawą finalnego zamknięcia zadania.**

## 1. Preflight (@ START_HEAD)

| Check | Result |
|-------|--------|
| Branch | `development` |
| HEAD vs **EXPECTED_START_HEAD** | **match** @ `7a5f3524dc2648a8161db397d48d86a2d59b2410` |
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

**Prior false-positive risk:** mutating only `runtime_revision.materialization_artifact_digest` while leaving projection input and `ActivateRuntimeRevisionRequest` unchanged makes `run_deploy_and_activate` fail at admission digest consistency (`ReferenceProductionLifecycleError`) **before** `ActivationService.prepare_candidate` — a pytest pass would not prove integrated P3-T-07.

**Corrected physical failure mechanism:** after D4 `_qualified_canonical_bundle` seeds a fully consistent `RegistryProjectionInputBundle`, `RuntimeRevision`, locator, and `expected_artifact_digest`, the test corrupts on-disk venv bundle bytes (`bundle_marker.py`) **without** changing any admission fields. `deploy_and_activate` admission-input validation succeeds; the **first** expected failure is `VenvBundleColocatedRuntimeDeploymentAdapter.prepare` → `verify_venv_bundle_artifact_bytes` digest mismatch (`VenvBundleColocatedDeploymentError`, D1 `test_wrong_digest_rejected` class).

**Boundary:** `open_strict_production_serving_session` → `activation_intent` → `deploy_and_activate` / `run_deploy_and_activate` (admission satisfied) → `ActivationService.prepare_candidate` → real A1 `prepare`.

**Proof adapter reached:** non-invasive `prepare` call-count spy wraps the real `VenvBundleColocatedRuntimeDeploymentAdapter.prepare` method (delegates to class implementation); assertion `prepare_calls >= 1` after the negative path. `prepare_candidate` and `deploy_and_activate` are **not** mocked or replaced.

**Real:** strict opener, D2/D4 lifecycle storage, SQLite AD stores, A1 adapter, governance wiring, canonical bundle seeding (`test_prod_q_r1_r7_d4_strict_production_gca_launcher` helpers).

**Instrumented only:** `prepare` call counter (spy delegate).

**Expected negative outcomes:** `VenvBundleColocatedDeploymentError` matching digest failure before session return; no returned `StrictProductionServingSession` for failed intent; traffic pointer unchanged; prior good revision still opens traffic-ready session with valid `ServingRuntimeHandoff` and colocated admission; lifecycle cleanup on failed strict open (opener `except` closes process resources).

**Limitations:** does not assert every partial durable row / FAILED deployment instance field; does not spy `prepare_candidate` separately (prepare spy is the physical boundary witness); post-failure traffic checks use the same durable store handle opened during good-revision activation (SQLite paths unchanged).

**Node ID:** `tests/qualification/prod_q/test_prod_q_r1_r7_q_p3_integration_gate.py::test_ig_e01_strict_activation_intent_fail_prepare_no_handoff`

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
| **FINAL_COMMIT** | `6952f1c4c3acaba6fa5952ba3e8606d07557c7c7` |
| **Files touched @ correction** | `test_prod_q_r1_r7_q_p3_integration_gate.py`, this evidence record |

## 8. Operator authorization

Allowlisted pytest (when authorized):

```text
tests/qualification/prod_q/test_prod_q_r1_r7_q_p3_integration_gate.py::test_ig_e01_strict_activation_intent_fail_prepare_no_handoff
tests/qualification/prod_q/test_prod_q_r1_r7_q_p3_f_strict_repeat_dispatch_negative.py::test_strict_session_external_work_no_second_provider_mutation_same_invocation_id
```
