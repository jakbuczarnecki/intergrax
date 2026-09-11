# DG-001B4 — Current-head revalidation

**Verdict:** REVALIDATED (PASS)

**Task:** `DG-001B4-CURRENT-HEAD-REVALIDATION`

**Historical qualification:** [`DG_001B4_WORKER_PRE_B5_INTEGRATION_QUALIFICATION.md`](DG_001B4_WORKER_PRE_B5_INTEGRATION_QUALIFICATION.md)

**Historical qualification SHA / baseline:** `1af51065070dae2f5b777abc2e0bc9fc1821bb67`

**Cross-system correction ancestor:** `d5c5b1cf02e1578d4738d6bbfbddef223faaabdb`

**START_HEAD:** `555ea245176eb26995074fb50254ce4807de710a`

**QUALIFICATION_BASELINE_HEAD:** `555ea245176eb26995074fb50254ce4807de710a`

**Revalidation date:** 2026-09-09

**Branch:** `development`

**PRODUCTION_CHANGES:** NO

---

## 1. Ancestry gate

| Ancestor | SHA | Result |
|----------|-----|--------|
| DG-001B3 | `e0e99e907e5f7e1b02e452bab0e1eefc5822a07b` | PASS |
| Cross-system audit correction | `d5c5b1cf02e1578d4738d6bbfbddef223faaabdb` | PASS |

Both are ancestors of `QUALIFICATION_BASELINE_HEAD`.

---

## 2. Revalidation scope

Revalidation reuses the historical B4 proof harness and contracts without redesign:

| Artifact | Status |
|----------|--------|
| `scripts/proof/dg001b4_pre_b5_qualification_contracts.py` | unchanged, canonical |
| `scripts/proof/dg001b4_pre_b5_qualification_support.py` | unchanged, canonical |
| `scripts/proof/dg001b4_pre_b5_worker_child.py` | unchanged, canonical |
| `tests/unit/runtime/architecture/test_dg001b4_pre_b5_integration_qualification.py` | unchanged, canonical |
| `tests/unit/scripts/proof/test_dg001b4_pre_b5_qualification.py` | unchanged, canonical |

Real worker bootstrap path remains:

```text
local_workspace_application.host.background_worker_main.main()
  → mint_bootstrap_attempt_id()
  → activate_local_workspace_reference_production_authority()
  → BootstrapIdentitySnapshot (attempt_id + application_id + process_role)
  → run_guarded_hosted_bootstrap_segment(
        producer=_BOOTSTRAP_FAILURE_PRODUCER,
        readiness=B3_TENANT_BINDING,
        stage=DEPENDENCY_RESOLUTION,
        segment=build_local_workspace_worker_bootstrap_diagnostics,
     )
  → [on failure] HostedBootstrapFailureRecord → reporter(s)
  → primary exception re-raised
```

---

## 3. Scenario A — Pre-B5 failure

**Test:** `test_scenario_a_pre_b5_failure_emits_record_and_preserves_primary_exception`

| Check | Historical B4 | Current baseline | Result |
|-------|---------------|------------------|--------|
| Single `HostedBootstrapFailureRecord` | PASS | observed | PASS |
| Schema `intergrax.hosting.bootstrap_failure_record` | PASS | observed | PASS |
| Stage/phase/reason correct | PASS | observed | PASS |
| Primary `RuntimeError` preserved | PASS | observed | PASS |
| No `APPLICATION_FAILED` pre-B5 | PASS | observed | PASS |
| No `DiagnosticProblem` pre-B5 | PASS | observed | PASS |

Production entrypoint conformance: `test_scenario_a_worker_main_entrypoint_uses_production_producer_and_guarded_segment` — PASS.

---

## 4. Scenario B — Identity boundary

**Test:** `test_scenario_b_pre_b5_identity_boundary_has_attempt_id_without_fabricated_identity`

| Field | Historical B4 | Current baseline | Result |
|-------|---------------|------------------|--------|
| `bootstrap_attempt_id` present | PASS | observed | PASS |
| `application_id` = `local_workspace` | PASS | observed | PASS |
| `process_role` = `background_worker` | PASS | observed | PASS |
| `instance_id` not fabricated | PASS | observed | PASS |
| `diagnostic_tenant_id` not fabricated | PASS | observed | PASS |
| No `ProblemId` / `TaskId` / `RunId` / `AttemptId` / `ExecutionId` | PASS | observed | PASS |

---

## 5. Scenario C — Reporter failure isolation

**Test:** `test_scenario_c_reporter_failure_does_not_mask_primary_failure`

| Check | Historical B4 | Current baseline | Result |
|-------|---------------|------------------|--------|
| Secondary reporter may fail | PASS | observed | PASS |
| Reporter error isolated | PASS | observed | PASS |
| Primary failure still propagates | PASS | observed | PASS |
| First reporter still receives record | PASS | observed | PASS |
| No retry loop from reporting | PASS | observed | PASS |

---

## 6. Scenario D — Post-B5 HOST-DIAG-3 boundary

**Test:** `test_scenario_d_post_b5_boundary_preserves_host_diag3_application_failed_flow`

| Check | Historical B4 | Current baseline | Result |
|-------|---------------|------------------|--------|
| `APPLICATION_FAILED` → HOST-DIAG-3 → `DiagnosticProblem` | PASS | observed | PASS |
| Pre-B5 producer does not capture post-B5 flow | PASS | observed | PASS |
| No duplicate Problem | PASS | observed | PASS |
| No duplicate `APPLICATION_FAILED` | PASS | observed | PASS |
| No HOST-DIAG-3 bypass | PASS | observed | PASS |

Pre-B5 success path: `test_pre_b5_success_path_reaches_post_b5_guard_without_bootstrap_failure_record` — PASS.

---

## 7. Contract summary

| Contract | Historical B4 | Current baseline | Result |
|----------|---------------|------------------|--------|
| Pre-B5 record creation | PASS | observed | PASS |
| Identity non-fabrication | PASS | observed | PASS |
| Primary failure preservation | PASS | observed | PASS |
| Reporter isolation | PASS | observed | PASS |
| No Problem pre-B5 | PASS | observed | PASS |
| HOST-DIAG-3 post-B5 | PASS | observed | PASS |
| No Execution identity dependency pre-B5 | PASS | observed | PASS |

---

## 8. Identity proof

Pre-B5 bootstrap identity (`bootstrap_attempt_id`, `application_id`, `process_role`) remains separate from Execution Engine identity (`ExecutionId`, `RunId`, `AttemptId`). Confirmed by:

- `test_worker_main_has_no_execution_identity_imports` — PASS
- `test_process_bootstrap_module_has_no_execution_identity_imports` — PASS
- `test_identity_rules_no_instance_id_before_b1` — PASS

---

## 9. Problem absence proof

After controlled pre-B5 failure, `InMemoryProblemPersistence` remains empty. No `HostedApplicationEvent` / `APPLICATION_FAILED` emitted from pre-B5 guarded segment. Verified in Scenario A.

---

## 10. Post-B5 non-regression

Post-B5 worker construction failure still flows through canonical HOST-DIAG-3:

```text
APPLICATION_FAILED
  → HOST-DIAG-3
  → DiagnosticOrchestrator
  → ProblemLifecycleEngine
  → DiagnosticProblem
```

Confirmed by Scenario D and `test_worker_bootstrap_b6_failure_problem_visible_via_worker_read_side` — PASS.

---

## 11. Execution Engine impact

Pre-B5 worker bootstrap still occurs before canonical task `ExecutionRuntime`. Worker bootstrap does not require `ExecutionId` / `RunId` / `AttemptId`.

**Execution Engine impact:** NO IMPACT

Cross-system convergence gate (not B4 functional test, regression gate only):

`tests/unit/applications/architecture/test_npsc3g_application_runtime_convergence_gate.py` — 5/5 PASS

---

## 12. Multi-agent gap R1-01

`GAP-R1-01` (`parent_execution_id`, DIAG-2 read-side lineage) is **out of scope** for B4. No unexpected interaction observed.

---

## 13. Focused regression results

Authoritative run on `QUALIFICATION_BASELINE_HEAD` (`555ea245176eb26995074fb50254ce4807de710a`):

```text
tests/unit/runtime/architecture/test_dg001b4_pre_b5_integration_qualification.py     6 passed
tests/unit/scripts/proof/test_dg001b4_pre_b5_qualification.py                          3 passed
tests/unit/hosting/test_bootstrap_failure_record.py                                    7 passed
tests/unit/hosting/test_guarded_process_bootstrap.py                                  26 passed
tests/unit/applications/local_workspace_application/test_lkw_background_worker_bootstrap_conformance.py  14 passed
tests/unit/applications/local_workspace_application/test_lkw_background_worker_constructor_seam.py          8 passed
tests/unit/scripts/proof/test_dg001b_r5_bootstrap_failure_qualification.py             5 passed
tests/unit/applications/architecture/test_npsc3g_application_runtime_convergence_gate.py  5 passed
```

**Total: 74 passed, 0 failed**

Log: `.tmp/session/dg001b4-revalidation/pytest.log`

---

## 14. Remote advancement

**REMOTE_ADVANCED_AFTER_QUALIFICATION:** NO

Local and `origin/development` both at `555ea245176eb26995074fb50254ce4807de710a` at documentation time.

---

## 15. Confirmations

- Diagnostics core unchanged
- Execution Engine unchanged
- Decision System unchanged
- Agent Distribution unchanged
- LKW production unchanged
- Queue unchanged
- HOST-DIAG-3 unchanged
- ProblemLifecycleEngine unchanged
- No bypass
- No private API
- No getattr/setattr in qualification contracts
- No synthetic identity
- No branch / worktree / history rewrite

---

## 16. Final verdict

```text
DG-001B4 = REVALIDATED ✅
STATUS = PASS
HISTORICAL_B4 = VALID
```

Historical DG-001B4 semantic contracts (Scenarios A–D) remain satisfied on current `development` baseline with zero production changes.
