# DG-001 — Pre-execution / operator startup failure visibility audit (R1)

> **Task:** DG-001-PRE-EXECUTION-STARTUP-FAILURE-VISIBILITY-AUDIT-R1  
> **Mode:** review-only / architecture discovery — **no remediation**  
> **Start HEAD:** `8a0824c2434314c2746be9c690ce87fc0e51fb87`  
> **Audit date:** 2026-09-07

---

## 1. Verdict

**Audit R1: PASS** — root causes localized with evidence-backed boundaries. **DG-001 remains open** (not enterprise-closed).

DG-001 is **two capability slices**, not one monolithic gap:

| Slice | Scope | R1 status |
|-------|--------|-----------|
| **DG-001A** | Hosted startup failure projection (`APPLICATION_FAILED` → central diagnostics) | Mechanism **shipped** (HOST-DIAG-2/3); **production wiring optional**; qualification **partial** |
| **DG-001B** | Pre-host / bootstrap failure producer (worker composition, public launcher, host composition before publisher) | **Missing canonical producer**; observability/log only |

---

## 2. Historical evidence

Ledger entry DG-001 (`DIAGNOSTIC_GAP_LEDGER.md`) documents:

- LKW background worker container exit before TaskId/RunId (DG-A: `create_kafka_worker()` missing `causal_evidence_persistence`).
- Public Windows proof `.bat` historical `ModuleNotFoundError: local_workspace_application` before workload start (manual `PYTHONPATH` workaround pre-fix).

Related shipped work (not DG-001 closure):

- **HOST-DIAG-2:** `DiagnosticSubjectRef` / `DiagnosticSignalSubjectScope` / `signal_subjects` on `DiagnosticOrchestrationRequest`.
- **HOST-DIAG-3:** `HostedApplicationDiagnosticEventPublisher` + `hosted_application_failure_to_problem_signal`.

Architecture references: `docs/project/architecture/APPLICATION_HOSTING.md` (HOST-DIAG-3), `docs/project/architecture/OBSERVABILITY.md` (diagnostic spine).

---

## 3. Existing shipped capabilities

Frozen intended chain (unchanged):

```text
hosting/bootstrap/application lifecycle truth
        ↓
typed non-execution failure signal/event
        ↓
DiagnosticSignalSubjectScope
        ↓
DiagnosticOrchestrationRequest.signal_subjects
        ↓
DiagnosticOrchestrator
        ↓
Problem / diagnostic projection
```

**Shipped hosted path (when product wires HOST-DIAG-3):**

```text
HostedApplicationEvent (APPLICATION_FAILED)
    → ObservabilityHostedApplicationEventPublisher (observability export first)
    → hosted_application_failure_to_problem_signal
    → DiagnosticSignalSubjectScope (tenant_id + application_id + instance_id)
    → DiagnosticOrchestrator.run (sync)
    → ProblemLifecycleEngine.reconcile
    → Problem persistence
```

**Default platform foreground runner** (`run_hosted_application` / `_default_runner_factories`) uses **`ObservabilityHostedApplicationEventPublisher` only** — no central diagnostics projection unless `event_publisher_factory` overrides (LKW: `run_local_workspace_hosted_application` passes diagnostic wiring only when both `diagnostic_orchestrator` and `diagnostic_tenant_binding` are supplied).

---

## 4. Subject model

**Contracts:** `DiagnosticSubjectRef`, `DiagnosticSignalSubjectScope`, `DiagnosticOrchestrationRequest.signal_subjects`.

| Field | Requirement |
|-------|-------------|
| `tenant_id` | Required, normalized non-empty |
| `application_id` | Required for signal subjects |
| `instance_id` | Required for signal subjects |
| Subject kinds | `EXECUTION` (task_id + run_id) · `APPLICATION_INSTANCE` (application_id + instance_id) |
| TaskId / RunId / AttemptId | **Not required** for non-execution path |

**Can current subject contract represent?**

| Surface | Verdict |
|---------|---------|
| Application startup failure | **YES** — `APPLICATION_INSTANCE` via `application_id` + `instance_id` |
| Worker startup failure | **PARTIAL** — contract can hold signals **if** producer supplies application/worker identity + tenant; **no worker-specific kind**; no worker producer today |
| Launcher/bootstrap failure | **NO** — pre-Python / pre-host failures lack stable `application_id`/`instance_id` and often lack tenant |

**Synthetic execution identity:** **not observed.** `hosted_application_failure_to_problem_signal` leaves execution fields empty; orchestration uses `signal_subjects` only (`test_df4_hosted_failure_projection_has_no_execution_identity`).

---

## 5. Hosted app failure path

### 5.1 Lifecycle producer

`HostedApplicationEngine` emits `APPLICATION_FAILED` during `_startup_failure_cleanup` → `_execute_bounded_terminal_cleanup(record_lifecycle_failure_event=True)` → `_publish_lifecycle_failed_event`, with bounded payload from `hosted_failure_event_payload` (`phase`, `reason_code`, `source_kind`, `source_id`, `exception_type`).

### 5.2 Failure-point matrix

| Failure point | Caught by hosted lifecycle? | `APPLICATION_FAILED` emitted? |
|---------------|----------------------------|-------------------------------|
| App initialize / engine `start()` | YES (`HostedApplicationEngine`) | YES |
| Startup hooks / component start | YES | YES |
| Runtime factory / runtime start | YES | YES |
| Worker startup (separate process) | NO | NO |
| Config / profile validation before supervisor | NO (exception propagates) | NO |
| Supervisor `engine_factory` / contract mismatch | YES (`HostedApplicationSupervisor`) | **NO** — exit record only (`HostedApplicationExitRecord`), no lifecycle event |

### 5.3 `HostedApplicationDiagnosticEventPublisher`

- Observability publish **first** (`await self._observability_publisher.publish(event)`).
- Diagnostic projection only for `APPLICATION_FAILED` with bounded payload (`phase` + `reason_code` required).
- Broad `except Exception` around projection: **intentional isolation** — logs `hosted application diagnostic projection failed`; **does not rewrite** hosting observability truth already exported.
- **Answer:** If central diagnostic projection fails, canonical `APPLICATION_FAILED` observability export is **preserved** (YES).

### 5.4 Projection (`hosted_application_failure_to_problem_signal`)

Preserves: `application_id`, `instance_id`, `event_id`, `correlation_id`, `occurred_at`, bounded `phase`/`reason_code`/`exception_type`, severity, lifecycle state in `application_attributes`. Returns `None` without bounded failure facts. Deterministic grouping inputs via structural signal fields (DIAG-5B). **Generic enough for hosted application startup failures**; not for pre-host bootstrap without `HostedApplicationEvent`.

### 5.5 Problem grouping

Repeated startup failures with **same structural signature** → **one Problem, multiple occurrences** (across instances when signature matches — `test_recurrence_across_instances_and_replay`). Different signatures → separate Problems (`test_different_failure_signature_isolation`). No Task/Run required.

### 5.6 `instance_id` semantics

Minted per supervisor attempt (`uuid4` default). Stable for one failed instance lifetime; **new instance_id on restart attempt**. Sufficient for per-attempt operator diagnosis; recurring failures may group under one Problem when signature matches.

---

## 6. Worker startup path

**LKW composition root:** `background_worker_main.main()` → `activate_local_workspace_reference_production_authority()` → `build_local_workspace_background_worker_wiring()` → `create_kafka_worker(...)`.

Failures before execution identity (e.g. missing `causal_evidence_persistence`, missing `kv_store`, registry projection, production authority activation) surface as **Python exceptions / process exit** — **no** `HostedApplicationEvent`, **no** `APPLICATION_FAILED`, **no** `DiagnosticSignalSubjectScope`.

**Historical DG-A fixture (`create_kafka_worker` missing `causal_evidence_persistence`):**  
**Would CURRENT platform capture equivalent failure?** → **NO** (composition-time `TypeError` before worker host lifecycle; no canonical startup producer).

Worker path does not use `HostedApplicationSupervisor` or `HostedApplicationDiagnosticEventPublisher`.

---

## 7. Public bootstrap path

**Canonical wrappers** (e.g. `run-lkw-core-platform-proof-windows.bat`):

```text
shell (.bat)
  → uv run --project applications/local_workspace_application python run-lkw-core-platform-proof.py
```

Historical `ModuleNotFoundError: local_workspace_application` occurred at **PYTHON-BOOTSTRAP** (import / project layout), **before** host composition and **before** `HostedApplicationEventPublisher`.

**Classification:** `PYTHON-BOOTSTRAP` (not `PRE-PYTHON` for current wrappers that require `uv`; shell-only failures without `uv` are `PRE-PYTHON`).

**Canonical platform publisher at this layer:** **absent**. Fix was launcher/project invocation (`uv --project`), not diagnostics spine.

**Architectural ownership:** proof/bootstrap launcher owns failure fact; central diagnostics consumes typed signals only when a host/application composition root exists with tenant + subject identity. Shell scripts are **not** Diagnostic Engine producers (frozen principle: **producer owns failure fact**).

---

## 8. Tenant binding

| Surface | Tenant known? | Canonical source |
|---------|---------------|------------------|
| Hosted app (HOST-DIAG-3 wired) | YES when wired | `HostedDiagnosticTenantBinding` (product-owned; LKW may use `profile_id`) |
| Hosted app (default runner) | N/A for diagnostics | Observability export has no diagnostic tenant requirement |
| Background worker | **NOT at bootstrap failure boundary** | Tenant exists later in execution admission, not at composition-root crash |
| Public proof launcher | **NOT PROVEN** at PYTHON-BOOTSTRAP | Environment/profile may exist in settings but no diagnostic binding |
| Bootstrap before config | NO | Do not invent tenant |

---

## 9. Durability

`DiagnosticOrchestrator.run()` is **synchronous** through `ProblemLifecycleEngine.reconcile()` persistence writes.

| Deployment | Durability before process exit |
|------------|--------------------------------|
| Wired HOST-DIAG-3 + durable `DocumentStore` Problem persistence | **PROVEN** in unit/integration test stacks (orchestrator returns after persist) |
| In-memory / unwired diagnostics | **NOT PROVEN** for operator durability |
| Observability export only (default runner) | Hosting truth in observability envelope — **not** Problem persistence |

**If process exits immediately after `APPLICATION_FAILED` with diagnostics wired + durable store:** successful `orchestrator.run()` return implies Problem recorded — **PROVEN** in test harness; **NOT PROVEN** for all production compositions.

**Diagnostics unavailable paradox:** observability export runs first; diagnostic projection failure is non-fatal. Startup truth can survive diagnostics outage via observability — **YES** for hosted `APPLICATION_FAILED` when engine publishes.

---

## 10. Operator discovery

`DiagnosticReadService.list_problems(tenant_id=...)` discovers Problems created from `signal_subjects` (**proven** in `test_application_failure_creates_problem`).

`get_problem` occurrence views for `APPLICATION_INSTANCE` return `read_status=UNAVAILABLE`, `unavailable_reason=NON_EXECUTION_SUBJECT` — operator sees Problem + subject ref, not execution reconstruction (**by design**).

**Gaps:**

- No first-class **application_id / instance_id listing** filter on public read API (tenant-wide scan).
- Default LKW hosted path **without** diagnostic wiring → **no Problem** (observability only).
- Pre-host bootstrap failures → **no Problem**.

**Operator discovery for non-execution startup failure:** **PARTIAL** — proven when HOST-DIAG-3 wired; **MISSING** for default production wiring and all DG-001B surfaces.

---

## 11. Evidence ladders

### Hosted app (engine startup failure, HOST-DIAG-3 wired)

| Step | Status |
|------|--------|
| Failure occurs | PROVEN PASS |
| Typed canonical event (`APPLICATION_FAILED`) | PROVEN PASS |
| Tenant bound | PROVEN PASS (when product wires `HostedDiagnosticTenantBinding`) |
| Non-execution subject created | PROVEN PASS |
| `PlatformProblemSignal` created | PROVEN PASS |
| `DiagnosticOrchestrator` called | PROVEN PASS |
| Problem persisted | PROVEN PASS (test stack) |
| Operator read possible | PROVEN PASS (`list_problems`) |

### Hosted app (default runner — observability only)

| Step | Status |
|------|--------|
| Failure occurs | PROVEN PASS |
| Typed canonical event | PROVEN PASS |
| Tenant bound (diagnostics) | PROVEN FAIL (not wired) |
| Non-execution subject | PROVEN FAIL |
| PlatformProblemSignal / orchestrator / Problem | PROVEN FAIL |
| Operator read | PROVEN FAIL |

### Worker bootstrap (LKW `background_worker_main` / factory)

| Step | Status |
|------|--------|
| Failure occurs | PROVEN PASS |
| Typed canonical event | PROVEN FAIL |
| Tenant bound | NOT PROVEN |
| Non-execution subject | PROVEN FAIL |
| PlatformProblemSignal | PROVEN FAIL |
| DiagnosticOrchestrator | PROVEN FAIL |
| Problem persisted | PROVEN FAIL |
| Operator read | PROVEN FAIL |

### Public bootstrap (`ModuleNotFoundError` class)

| Step | Status |
|------|--------|
| Failure occurs | PROVEN PASS (historical) |
| Typed canonical event | PROVEN FAIL |
| Tenant bound | PROVEN FAIL |
| Non-execution subject | PROVEN FAIL |
| PlatformProblemSignal | PROVEN FAIL |
| DiagnosticOrchestrator | PROVEN FAIL |
| Problem persisted | PROVEN FAIL |
| Operator read | PROVEN FAIL |

---

## 12. First broken boundaries

| Surface | First proven fail |
|---------|-------------------|
| Hosted app (engine path, wired) | **None** in spine — operator discovery filter/limitations only |
| Hosted app (production default) | **Producer not wired** — `event_publisher_factory` defaults to observability-only |
| Hosted app (supervisor pre-engine) | **No `APPLICATION_FAILED`** — `HostedApplicationSupervisorError` exit record only |
| Host composition before publisher | **No canonical failure event** — config/path/definition exceptions |
| Worker bootstrap | **No canonical failure event** at `build_local_workspace_background_worker_wiring` / `main()` |
| Public bootstrap | **No canonical producer** — PYTHON-BOOTSTRAP / stderr |

---

## 13. Root-cause categories (per surface)

| Surface | Categories |
|---------|------------|
| Hosted engine startup (wired) | — (qualified path) |
| Hosted production default | **C** producer not wired · **F** operator discovery missing |
| Supervisor engine-factory failure | **A** no canonical `APPLICATION_FAILED` · **D** no diagnostic projection |
| Host pre-supervisor composition | **A** no canonical event |
| Worker bootstrap | **A** no canonical event · **C** no producer |
| Public launcher PYTHON-BOOTSTRAP | **A** no canonical event · **G** historical launcher gap remediated for success path only |
| Diagnostics projection outage (hosted) | **H** not applicable — observability preserves truth |

---

## 14. Architecture ownership

| Layer | Owns |
|-------|------|
| `HostedApplicationEvent` | Hosting lifecycle truth |
| Observability export | Fact export (vendor-optional) |
| `PlatformProblemSignal` | Functional problem projection |
| `DiagnosticOrchestrator` | Interpretation / grouping |
| Product composition | Tenant binding + wiring `HostedApplicationDiagnosticEventPublisher` |
| Worker / proof launcher bootstrap | **Should own** pre-execution typed failure facts (not diagnostics core) |

**Forbidden (confirmed not proposed):** global `sys.excepthook`, stderr scraping as primary contract, new `StartupFailureStore`, new orchestrator, synthetic TaskId/RunId.

---

## 15. Reuse / pluginability assessment

Current extension point **`typed subject signal producer → signal_subjects`** supports new host types **without** `DiagnosticOrchestrator` branches — **YES**.

Diagnostics core does **not** import LKW, Kafka, Celery, launchers — **PASS**.

Missing producers belong in **application/hosting/bootstrap** layers.

---

## 16. Frozen recommended slices (ordered)

1. **DG-001A — Hosted startup diagnostic wiring qualification:** Require product hosted entrypoints (LKW foreground) to wire HOST-DIAG-3 with durable Problem persistence; prove operator `list_problems` on real hosted startup failure.
2. **DG-001B — Worker bootstrap typed failure producer:** Explicit bounded bootstrap failure contract at worker composition root (`background_worker_main` / shared host queue wiring) → `signal_subjects` (no execution identity fabrication).
3. **DG-001C — Public proof launcher bootstrap producer:** PYTHON-BOOTSTRAP / host-composition failures before `HostedApplicationEventPublisher` — typed bounded record at correct semantic owner (hosting bootstrap or proof launcher), not diagnostics core.
4. **DG-001D — Supervisor pre-engine gap:** `HostedApplicationSupervisor` engine-factory / contract failures → `APPLICATION_FAILED` or equivalent typed hosting event (architecture decision; currently exit record only).

---

## 17. Non-claims

- Does **not** claim DG-001 enterprise-closed.
- Does **not** claim worker/container failures are visible today.
- Does **not** claim public launcher failures reach Central Diagnostics.
- Does **not** claim all production deployments use durable Problem persistence.
- Does **not** claim external-service monitoring (S5 broad scope).
- Does **not** implement remediation.

---

## 18. Next task

**Recommended single next task:** **DG-001B — Worker bootstrap typed failure producer qualification** (canonical producer at queue-enabled worker composition root; historical `create_kafka_worker` / `causal_evidence_persistence` fixture as qualification gate).

---

## Appendix A — Hard contract assessment (H1–H12)

| ID | Result |
|----|--------|
| H1 non-execution subject supported | **PASS** |
| H2 no identity fabrication | **PASS** |
| H3 explicit tenant ownership | **PASS** |
| H4 startup lifecycle truth typed | **PARTIAL** |
| H5 observability before diagnostic projection | **PASS** |
| H6 application startup → diagnostics | **PARTIAL** |
| H7 worker startup → diagnostics | **FAIL** |
| H8 bootstrap launcher → diagnostics | **FAIL** |
| H9 diagnostic result durable before exit | **PARTIAL** |
| H10 operator can discover non-execution failure | **PARTIAL** |
| H11 diagnostics failure does not erase startup truth | **PASS** |
| H12 core remains vendor-neutral | **PASS** |

---

## Appendix B — Broad exception catch (`HostedApplicationDiagnosticEventPublisher`)

**Classification: B — deliberate isolation of secondary diagnostic projection.**

Rationale: observability emitted first; exception logged; hosting lifecycle not aborted by diagnostics outage.

---

## Appendix C — Test evidence (R1 runs)

Environment: `uv sync --extra dev-ci --frozen`; `--basetemp=.tmp/session/dg001-audit-r1/pytest-basetemp`; **`--ignore` not used**.

| Suite | Command | Result |
|-------|---------|--------|
| Hosted diagnostics | `pytest tests/unit/applications/_shared/test_hosted_application_diagnostic_integration.py -q` | **8 passed** |
| Hosting lifecycle | `pytest tests/unit/hosting/supervisor/test_supervisor.py tests/unit/hosting/test_hosted_application_events.py tests/unit/hosting/engine/test_engine_startup.py tests/unit/hosting/engine/test_engine_w2_invariants.py -q` | **61 passed** |
| Worker startup | `pytest tests/unit/applications/local_workspace_application/test_lkw_background_worker_{authority,queue_dependencies}.py -q` | **2 passed, 9 failed** (environment: `LLMAdapterDependencyError` for `ollama` during worker wiring tests) |
| Worker regression (isolated) | `pytest .../test_lkw_background_worker_queue_dependencies.py::test_create_kafka_worker_requires_causal_evidence_persistence_regression -q` | **1 passed** |
| Operator / subject | `pytest tests/unit/runtime/diagnostics/test_application_diagnostic_subjects.py tests/unit/runtime/architecture/test_diag_foundation_4_entrypoint_consistency.py -q` | **21 passed** |

**Qualification evidence gap:** no dedicated unit test asserting diagnostic projection exception preserves observability envelopes (behavior established by code order + `test_observability_export_before_diagnostics`).

---

## Appendix D — Composition order (first diagnostics availability)

Actual LKW hosted foreground:

```text
resolve_hosted_application_definition(profile)
  → run_hosted_application factories (paths, clock, event_publisher)
  → [optional] build_hosted_application_diagnostic_event_publisher (requires injected orchestrator + tenant)
  → HostedApplicationSupervisor
  → HostedApplicationEngine
  → APPLICATION_FAILED possible
```

**First point canonical diagnostics become available:** construction of `HostedApplicationDiagnosticEventPublisher` (only when product injects orchestrator + tenant). Failures before that in the same process: **stderr / exception only**.

Worker: diagnostics orchestrator may exist inside `HarnessHostRuntime` for execution-time diagnosis, but **no bootstrap failure producer** connects composition exceptions to `signal_subjects`.
