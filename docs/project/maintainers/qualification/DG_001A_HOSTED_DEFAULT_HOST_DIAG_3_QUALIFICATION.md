# DG-001A Hosted Default Wiring — HOST-DIAG-3 Coverage Qualification

**Verdict:** PASS / QUALIFIED

**Date:** 2026-09-08

**Branch:** `development`

**Start HEAD:** `216616ecab74769307bf7a26b3bf1db91bd21569`

**Task:** `DG-001A-HOSTED-DEFAULT-WIRING-HOST-DIAG-3-COVERAGE` — qualification only; no diagnostics-core, LKW, queue, or default-runner production changes.

**Ancestor check:** `git merge-base --is-ancestor 8bf7daa4d65f901cc44061c4df6e1641abd5fba7 HEAD` → exit 0.

---

## 1. Verdict

```text
DG-001A HOSTED DEFAULT HOST-DIAG-3 WIRING = QUALIFIED
```

**Answer to the architectural question:**

| Question | Answer |
| -------- | ------ |
| Is bare `run_hosted_application()` automatically HOST-DIAG-3 wired? | **NO** — by design; platform default is observability-only |
| Is the supported hosted flow able to reach Central Diagnostics without manual bypass? | **YES** — via canonical `event_publisher_factory` override using `build_hosted_application_diagnostic_event_publisher` |
| Must the operator hand-create Problems or bypass Diagnostics? | **NO** — when product supplies tenant binding + orchestrator through the existing factory |

**DG-001A status after qualification:** `CLOSED / QUALIFIED` (intentional default boundary documented).

**DG-001 overall:** `PARTIALLY ADDRESSED` (DG-001C and B0–B5 remain open).

---

## 2. Canonical host wiring

```text
run_hosted_application(profile, event_publisher_factory=...)
        │
        ▼
HostedApplicationSupervisor
        │
        ├── engine_factory → HostedApplicationEngine(event_publisher=...)
        └── supervisor.event_publisher (same composed instance)
```

**Platform default (no override):**

| Component | Value |
| --------- | ----- |
| Composition root | `intergrax/hosting/runner.py` → `_default_runner_factories` |
| Publisher | `ObservabilityHostedApplicationEventPublisher` |
| HOST-DIAG-3 | **NO** |

**Supported product composition (canonical):**

| Component | Value |
| --------- | ----- |
| Public seam | `run_hosted_application(..., event_publisher_factory=...)` |
| Factory | `build_hosted_application_diagnostic_event_publisher` (`intergrax/applications/_shared/hosted_application_diagnostic_wiring.py`) |
| Reference product wiring | LKW foreground optional args; LKW background worker always composes at B5 |

---

## 3. HOST-DIAG-3 composition

```text
HostedApplicationSupervisor / HostedApplicationEngine
        │
        ▼
HostedApplicationEvent(APPLICATION_FAILED)
        │
        ▼
HostedApplicationDiagnosticEventPublisher
        ├── ObservabilityHostedApplicationEventPublisher (first)
        ├── hosted_application_failure_to_problem_signal
        └── DiagnosticOrchestrator → ProblemLifecycleEngine → persistence
```

Tenant binding is **product-owned** (`HostedDiagnosticTenantBinding`). Hosting core remains tenant-neutral per [`APPLICATION_HOSTING.md`](../../architecture/APPLICATION_HOSTING.md) § Central diagnostics boundary.

---

## 4. Identity flow

| Field | Source | Qualified |
| ----- | ------ | --------- |
| `application_id` | `HostedApplicationDefinition` / profile | YES |
| `instance_id` | `HostedApplicationSupervisor` minted per attempt | YES |
| `tenant_id` | `HostedDiagnosticTenantBinding` (product) | YES |
| Execution identity | **NONE** for hosting lifecycle | YES — `subject_ref.execution()` is `None` |

No new identity fields were added. Platform does not derive tenant from `application_id`.

---

## 5. Failure flow covered

| Failure mode | Default runner | Canonical HOST-DIAG-3 wiring |
| ------------ | -------------- | ---------------------------- |
| Engine `start()` failure → `APPLICATION_FAILED` | Observability export only | Observability → Problem lifecycle → read model |
| Supervisor pre-engine failure (DG-001D) | Observability export only when composed publisher wired | Same composed path when product overrides factory |
| Clean stop | Observability lifecycle events | No Problem |

**Not covered by DG-001A:** B0–B2 bootstrap before identity/tenant; DG-001C public launcher PYTHON-BOOTSTRAP; bare platform default without product factory override.

---

## 6. Test evidence

**Canonical test module:** `tests/unit/hosting/test_hosted_default_host_diag_3_qualification.py`

| Scenario | Result |
| -------- | ------ |
| `_default_runner_factories` → observability-only publisher | PASS |
| `run_hosted_application` exposes `event_publisher_factory` override | PASS |
| Default runner engine failure → 0 central Problems | PASS |
| Canonical factory override → Problem with `APPLICATION_INSTANCE` identity | PASS |

**Related existing suites (unchanged):**

- `tests/unit/applications/_shared/test_hosted_application_diagnostic_integration.py`
- `tests/unit/hosting/supervisor/test_supervisor_host_diag_3_conformance.py`
- `tests/unit/applications/architecture/test_host_diag_3_composition_gate.py`

---

## 7. Production changes

**None required.** Existing platform seam and product factory are the correct mechanism. Default observability-only runner is an intentional tenant-authority boundary, not a wiring defect.

---

## 8. Confirmations

- Diagnostics core unchanged
- LKW unchanged
- Queue unchanged
- No bypass of Central Diagnostics
- No private API usage in qualification harness
- No worktree / no new branch
