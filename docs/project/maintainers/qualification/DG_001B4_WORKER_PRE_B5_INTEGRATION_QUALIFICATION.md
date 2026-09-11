# DG-001B4 — Worker pre-B5 integration qualification

**Verdict:** QUALIFIED

**Date:** 2026-09-09

**Branch:** `development`

**START_HEAD:** `1af51065070dae2f5b777abc2e0bc9fc1821bb67`

**Task:** `DG-001B4-WORKER-PRE-B5-INTEGRATION-QUALIFICATION`

**Predecessors:** DG-001B3 (`HostedBootstrapFailureProducer`) · DG-001B2 contract design

---

## 1. Verdict

```text
DG-001B4 WORKER PRE-B5 INTEGRATION QUALIFICATION = QUALIFIED ✅
```

Scenarios **A–D PASS** on canonical worker bootstrap path. Pre-B5 failures emit `HostedBootstrapFailureRecord`; no `DiagnosticProblem` is created before B5.

---

## 2. Real path

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
  → [on failure] HostedBootstrapFailureRecord → LoggingBootstrapFailureReporter (+ qualification recorder)
  → primary exception re-raised (process exit / test propagation)
```

Qualification injects a controlled failure at the **production diagnostics segment** via `scripts/proof/dg001b4_pre_b5_qualification_contracts.py` — same signature as `build_local_workspace_worker_bootstrap_diagnostics`, no production seam changes.

Harness entrypoints:

| Artifact | Role |
|----------|------|
| `scripts/proof/dg001b4_pre_b5_qualification_support.py` | Mirrors `main()` pre-B5 guarded block |
| `scripts/proof/dg001b4_pre_b5_worker_child.py` | Subprocess child for manual evidence export |
| `tests/unit/runtime/architecture/test_dg001b4_pre_b5_integration_qualification.py` | Automated scenarios A–D |

---

## 3. Scenarios

### Scenario A — Pre-B5 failure

| Check | Expected | Observed |
|-------|----------|----------|
| Record creation | `HostedBootstrapFailureRecord` with schema `intergrax.hosting.bootstrap_failure_record` | PASS |
| Reporter delivery | `LoggingBootstrapFailureReporter` + qualification `RecordingBootstrapFailureReporter` receive record | PASS |
| Primary exception | `RuntimeError` with qualification sentinel preserved | PASS |
| Problem creation | None (`InMemoryProblemPersistence` empty) | PASS |

### Scenario B — Identity boundary

| Field | Expected | Observed |
|-------|----------|----------|
| `bootstrap_attempt_id` | Present, `bootstrap-attempt-*` | PASS |
| `application_id` | `local_workspace` | PASS |
| `process_role` | `background_worker` | PASS |
| `instance_id` | `None` (not fabricated) | PASS |
| `diagnostic_tenant_id` | `None` (not fabricated) | PASS |

### Scenario C — Reporter failure isolation

| Check | Expected | Observed |
|-------|----------|----------|
| Failing secondary reporter | Swallowed; logged by producer | PASS |
| Primary failure | `RuntimeError` still propagates | PASS |
| Record delivery | First reporter still receives record | PASS |

### Scenario D — Post-B5 boundary

| Check | Expected | Observed |
|-------|----------|----------|
| HOST-DIAG-3 flow after B5 | `APPLICATION_FAILED` on worker construction failure | PASS (no regression) |
| Pre-B5 record on post-B5 success | No bootstrap failure record when diagnostics succeed | PASS |

---

## 4. Problem absence proof

Pre-B5 guarded segment fails **before** `HostedProcessBootstrapContext.create()` and before a composed diagnostic publisher exists. Therefore:

- No `HostedApplicationEvent` / `APPLICATION_FAILED` from pre-B5 guard
- No `DiagnosticOrchestrator` projection
- No `DiagnosticProblem` in persistence

Verified in Scenario A via empty `InMemoryProblemPersistence` after controlled pre-B5 failure.

---

## 5. Tests

```text
tests/unit/runtime/architecture/test_dg001b4_pre_b5_integration_qualification.py
tests/unit/scripts/proof/test_dg001b4_pre_b5_qualification.py
tests/unit/hosting/test_bootstrap_failure_record.py
tests/unit/hosting/test_guarded_process_bootstrap.py
tests/unit/applications/local_workspace_application/test_lkw_background_worker_bootstrap_conformance.py
tests/unit/applications/local_workspace_application/test_lkw_background_worker_constructor_seam.py
tests/unit/scripts/proof/test_dg001b_r5_bootstrap_failure_qualification.py
```

---

## 6. Confirmations

- Diagnostics core unchanged
- LKW unchanged (qualification injection lives in `scripts/proof/`)
- Queue unchanged
- No bypass of guarded segment
- No private API in qualification contracts
- No branch / worktree / history rewrite

---

## 7. References

- [`DG_001B2_WORKER_PRE_B5_FAILURE_CONTRACT_DESIGN.md`](../architecture/DG_001B2_WORKER_PRE_B5_FAILURE_CONTRACT_DESIGN.md)
- [`DG_001B_REAL_CONTROLLED_BOOTSTRAP_FAILURE_QUALIFICATION_R5.md`](DG_001B_REAL_CONTROLLED_BOOTSTRAP_FAILURE_QUALIFICATION_R5.md) (post-B5 HOST-DIAG-3 — separate concern)
