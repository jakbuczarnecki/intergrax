# DG-001B — Worker bootstrap typed failure producer architecture (R1)

> **Task:** DG-001B-WORKER-BOOTSTRAP-TYPED-FAILURE-PRODUCER-ARCHITECTURE-AUDIT-R1  
> **Mode:** review-only / architecture freeze — **no implementation**  
> **Start HEAD:** `d200e4918ce9fe6f0923043ae9acf2441a391840`  
> **Audit date:** 2026-09-07  
> **Parent audit:** [`DG_001_PRE_EXECUTION_STARTUP_FAILURE_VISIBILITY_AUDIT.md`](DG_001_PRE_EXECUTION_STARTUP_FAILURE_VISIBILITY_AUDIT.md)

---

## 1. Verdict

**Audit R1: PASS** — worker pre-execution bootstrap failure gap confirmed; target architecture frozen with no blocking ambiguities.

| Decision | Frozen outcome |
|----------|----------------|
| HostedApplicationEvent reuse | **PARTIAL** — reuse envelope + `APPLICATION_FAILED`; bootstrap-specific bounded `phase` taxonomy |
| Generic new event contract | **NO** |
| New diagnostic subject kind | **NO** — reuse `APPLICATION_INSTANCE` |
| Reusable guarded bootstrap boundary | **YES** — platform primitive required |
| LKW-local `try/except` | **FORBIDDEN** as target architecture |
| Historical DG-A fixture coverage | **YES** at target boundary when diagnostic bootstrap context wired |

**DG-001 remains `PARTIALLY ADDRESSED`.** DG-001B implementation not yet shipped.

---

## 2. Current failure boundary

### 2.1 Proven LKW worker path (code-evidenced)

```text
background_worker_main.main()
  ↓
LocalWorkspaceBackendSettings.from_env()                    [B2]
  ↓
activate_local_workspace_reference_production_authority()   [B2–B5 heavy]
  ↓
build_local_workspace_background_worker_wiring()            [B6]
    → build_harness_host_runtime()
    → resolve_host_queue_execution_dependencies()
    → create_kafka_worker(...)                              [B6 failure locus]
  ↓
wiring.worker.start()                                       [B7]
```

**Sources:** `applications/local_workspace_application/host/background_worker_main.py`, `background_worker_factory.py`, `intergrax/integrations/providers/message_bus/kafka/bundle.py`.

### 2.2 Current failure behavior

| Property | Current state |
|----------|---------------|
| Exception propagation | YES — unhandled exception terminates `main()` |
| Process exit | Non-zero on uncaught exception; `return 1` only for message-bus gate |
| `HostedApplicationEvent` | **NO** |
| `APPLICATION_FAILED` | **NO** |
| `DiagnosticSignalSubjectScope` | **NO** |
| `Problem` persistence | **NO** |
| Execution identity (`TaskId`/`RunId`/`AttemptId`) | **NONE** — correct absence |

### 2.3 First broken boundary (historical + current)

**Historical DG-A fixture:** `TypeError: create_kafka_worker() missing 1 required keyword-only argument: 'causal_evidence_persistence'` at `create_kafka_worker` assembly inside `build_local_workspace_background_worker_wiring` — **before** `worker.start()`, **before** any execution identity.

**Current first failure loci (ordered):**

1. `main()` message-bus env gate → `return 1` (no exception; still no diagnostic spine).
2. `activate_local_workspace_reference_production_authority()` — registry projection, deploy/activate, settings parse.
3. `build_local_workspace_background_worker_wiring()` — harness runtime, queue dependency resolution, `create_kafka_worker`.
4. `worker.start()` — transport/worker runtime startup.

Regression test: `test_create_kafka_worker_requires_causal_evidence_persistence_regression` proves `create_kafka_worker` rejects missing `causal_evidence_persistence` at call boundary.

Authority test `test_bootstrap_failure_gates_worker_creation` proves earlier authority failure prevents `create_kafka_worker` — still no canonical failure fact.

---

## 3. Existing reusable capabilities

| Capability | Location | Reuse for DG-001B |
|------------|----------|-------------------|
| `HostedApplicationEvent` + `APPLICATION_FAILED` | `intergrax/hosting/contracts/events.py` | **YES** (envelope) |
| Bounded failure payload contract | `hosted_failure_event_payload` / projection requires `phase` + `reason_code` | **YES** (bootstrap phases as bounded strings) |
| `ObservabilityHostedApplicationEventPublisher` | `intergrax/hosting/eventing.py` | **YES** |
| `HostedApplicationDiagnosticEventPublisher` | `intergrax/applications/_shared/hosted_application_diagnostic_wiring.py` | **YES** (composed publisher pattern) |
| `hosted_application_failure_to_problem_signal` | `intergrax/applications/_shared/hosted_application_failure_projection.py` | **YES** (deterministic projection) |
| `DiagnosticSubjectRef` / `APPLICATION_INSTANCE` | `intergrax/runtime/diagnostics/diagnostic_subject.py` | **YES** |
| `DiagnosticOrchestrator` + Problem persistence | existing spine | **YES** — hard invariant |
| `HostedDiagnosticTenantBinding` | `hosted_application_diagnostic_wiring.py` | **YES** — product-owned tenant |
| `validate_instance_id` | `intergrax/hosting/contracts/public_data.py` | **YES** |
| Engine `HostedApplicationFailurePhase` | `intergrax/hosting/engine/diagnostics.py` | **NO** — engine lifecycle phases; bootstrap uses separate bounded phase taxonomy |
| `HostedApplicationSupervisor` / engine lifecycle | hosting runner | **NO** — worker bypasses supervisor |
| Generic `HostEvent` / `BootstrapEvent` / `ProcessLifecycleEvent` | — | **NOT FOUND** in repository |

**HOST-DIAG-3 ordering (frozen reuse):** observability export first → bounded diagnostic projection second → projection failure non-fatal.

---

## 4. Semantic ownership

| Artifact | Owner | Forbidden owner |
|----------|-------|-----------------|
| Canonical bootstrap failure fact | **`intergrax.hosting`** (process/bootstrap surface contracts) | `intergrax.runtime.diagnostics` |
| Bootstrap context + guarded execution primitive | **`intergrax.hosting`** | queue providers, LKW |
| Composed observability + diagnostic publisher | **`intergrax.applications._shared`** (product composition, same as HOST-DIAG-3) | diagnostics core |
| Problem projection adapter | **`intergrax.applications._shared`** (reuse/generalize `hosted_application_failure_to_problem_signal`) | diagnostics core |
| Product entrypoint wiring | **Tier-3 application** (`background_worker_main` supplies identity + bindings) | — |

**Frozen principle:** `FAILURE ORIGIN OWNS THE FACT` · `DIAGNOSTICS CONSUMES THE FACT`.

Diagnostics core **MUST NOT** import queue providers, worker factories, LKW, or hosting bootstrap concrete adapters.

---

## 5. Contract reuse decision

### 5.1 Option A — `HostedApplicationEvent` / `APPLICATION_FAILED`

| Question | Answer |
|----------|--------|
| Is worker semantically a hosted application instance for diagnostics? | **YES** — long-running application host process serving a declared `application_id`; distinct from foreground engine instance but same diagnostic subject domain |
| Can `application_id` be sourced canonically? | **YES** — caller-supplied (`LOCAL_WORKSPACE_APPLICATION_MANIFEST.app_id` = `local_workspace`) |
| Can `instance_id` exist before worker factory succeeds? | **YES** — minted by platform bootstrap wrapper at B1, before composition callback |
| Can supervisor semantics apply? | **NO** — worker does not use supervisor; **not required** for event truth |
| False semantics risk? | **PARTIAL** — no prior `APPLICATION_STARTING` lifecycle events; mitigated by `lifecycle_state=FAILED`, bootstrap `phase` in payload, optional `process_role` bounded field |

**Result: `REUSE HostedApplicationEvent = PARTIAL`**

Reuse the existing versioned hosting event envelope and `APPLICATION_FAILED` event type. Do **not** require `HostedApplicationEngine` or supervisor lifecycle to emit it. Bootstrap failures use a **separate bounded phase taxonomy** (not `HostedApplicationFailurePhase` engine enum values).

### 5.2 Option B — generic hosting/bootstrap event search

Repository search found **no** existing `HostEvent`, `BootstrapEvent`, `ProcessLifecycleEvent`, `ApplicationHostEvent`, `WorkerLifecycleEvent`, or `PlatformLifecycleEvent` contracts.

### 5.3 Option C — worker-specific event

**REJECTED** — not vendor-neutral, not reusable across Celery/Kafka/document-store/daemon hosts.

### 5.4 New contract gate

**Generic new event contract required: NO.**

Minimal new hosting types (R2) limited to:

- `HostedProcessBootstrapContext` (identity + bindings, immutable)
- `HostedProcessBootstrapPhase` (small bounded enum — see §9)
- `run_guarded_hosted_process_bootstrap(...)` primitive

Event payload remains `HostedApplicationEvent` with `APPLICATION_FAILED`.

### 5.5 Architecture comparison

| Candidate | Reusable | Semantically correct | Recommendation |
|-----------|:--------:|:--------------------:|----------------|
| Reuse `HostedApplicationEvent` directly | HIGH | PARTIAL (lifecycle stretch) | **ADOPT** with bootstrap phase taxonomy |
| Generic host/process failure event (new) | HIGH | HIGH | **DEFER** — unnecessary duplicate of hosting envelope |
| Worker-specific event | LOW | LOW | **REJECT** |
| LKW `try/except` wrapper | LOW | LOW | **REJECT** |
| Reusable guarded bootstrap boundary | HIGH | HIGH | **ADOPT** |

---

## 6. Subject identity

### 6.1 Subject kind

**Frozen: `APPLICATION_INSTANCE`** (`ApplicationDiagnosticSubjectRef`).

Worker host process maps truthfully to `application_id` + `instance_id` under product-owned diagnostic `tenant_id`. No new `HOST_INSTANCE` / `PROCESS_INSTANCE` subject kind required.

Optional bounded `process_role` in event payload (e.g. `background_worker`) distinguishes worker from foreground engine instances in operator views without new subject discriminator.

### 6.2 Identity matrix

| Field | Available before worker composition? | Canonical source | Required? |
|-------|:-----------------------------------:|------------------|:---------:|
| `application_id` | **YES** (B1) | Caller-supplied canonical context (`manifest.app_id` for LKW) | **YES** |
| `instance_id` | **YES** (B1) | Platform bootstrap wrapper mints once per process attempt (`uuid4` + `validate_instance_id`) | **YES** |
| `tenant_id` | **YES** (B3) when wired | Product-owned `HostedDiagnosticTenantBinding` — **not** execution tenant | **YES** for central diagnostics |
| `process_role` | **YES** (B1) | Caller-supplied bounded identifier in bootstrap context | **RECOMMENDED** |
| `environment/profile` | **YES** (B2) | Product environment profile (`profile_id`, e.g. `local_workspace.product`) | **OPTIONAL** (diagnostic tenant may equal `profile_id`) |

**Frozen principle:** `APPLICATION ID = CALLER-SUPPLIED CANONICAL CONTEXT`.

**`instance_id` minting owner:** platform `run_guarded_hosted_process_bootstrap` (or equivalent) at process bootstrap entry — **before** failure-prone composition callback. Diagnostic/hosting instance identity; **NOT** `ExecutionId`, `TaskId`, or `RunId`.

### 6.3 `instance_id` reuse

Reuse `validate_instance_id` and supervisor `InstanceIdGenerator` pattern (`uuid4` default). No requirement for file-guard lease acquisition at bootstrap failure boundary.

---

## 7. Tenant semantics

### 7.1 Audit conclusion

LKW background worker is a **platform/application service** that serves **multiple execution tenants** at task admission. It does **not** know execution `tenant_id` at bootstrap.

**Frozen rule: Option A — product-owned diagnostic tenant binding at process bootstrap.**

| Concept | Rule |
|---------|------|
| Execution tenant | **NOT USED** at bootstrap |
| Diagnostic tenant | **REQUIRED** for central diagnostics — explicit `HostedDiagnosticTenantBinding` supplied by product composition |
| LKW canonical binding | `profile_id` from `ApplicationEnvironmentProfile` (`local_workspace.product`) or dedicated product constant — **deployment/operator domain**, not per-task tenant |
| Fabrication | **FORBIDDEN** — do not invent `tenant_id` from env heuristics inside reusable contract |

### 7.2 Tenant-unavailable case

When diagnostic tenant binding is absent:

- Bootstrap failure remains **observability/log truth only**
- No central `Problem` projection
- Process failure semantics unchanged

This matches HOST-DIAG-3 optional wiring semantics for foreground host.

---

## 8. Bootstrap dependency ladder B0–B7

| Level | Definition | Observability | Central Diagnostics |
|-------|------------|:-------------:|:-------------------:|
| **B0** | Python process alive | stderr / process exit | **NO** |
| **B1** | Application identity + `instance_id` minted | structured log possible | **NO** |
| **B2** | Config / profile parsed | log + optional early export | **NO** |
| **B3** | Diagnostic tenant binding known | log | **NO** (tenant required but orchestrator may be absent) |
| **B4** | Observability publisher available | **YES** — canonical event export | **NO** |
| **B5** | `DiagnosticOrchestrator` + Problem persistence available | YES | **YES** — projection possible |
| **B6** | Worker runtime composed (`create_kafka_worker` etc.) | YES | **YES** *if B3–B5 wired before callback* |
| **B7** | Worker started (`worker.start()`) | YES | **YES** *if B3–B5 wired* |

### 8.1 Coverage contract (frozen)

```text
Once explicit diagnostic bootstrap context (B3) + observability publisher (B4)
+ DiagnosticOrchestrator stack (B5) are available, all subsequent worker
composition/startup failures (B6–B7) are canonically projected into central diagnostics.

Earlier failures (B0–B2, or B3–B5 not wired) remain bootstrap observability/log
facts only. DG-001B does NOT claim ALL bootstrap failures always enter central diagnostics.
```

### 8.2 Dependency paradox

`DiagnosticOrchestrator` requires Problem persistence (typically document store). Full `HarnessHostRuntime` is **not** required for orchestrator construction — `build_diagnostic_orchestrator` over `HostDiagnosticReadDependencies` suffices.

**LKW feasible wiring:** `activate_local_workspace_reference_production_authority()` materializes `ProductionProcessComposition` with stores **before** `build_local_workspace_background_worker_wiring()`. Product can resolve B5 from composition stores, then run guarded composition — covering historical `create_kafka_worker` failure at B6.

Failures during B2 authority activation **before** stores exist: observability/log only unless product supplies lighter diagnostic bootstrap.

**Central diagnostic coverage begins at: B5** (all of B3 + B4 + B5 satisfied).

---

## 9. Canonical event model

### 9.1 Event type

**Frozen:** `HostedApplicationEvent` with `event_type = APPLICATION_FAILED`.

### 9.2 Required fields

| Field | Value at bootstrap failure |
|-------|---------------------------|
| `application_id` | From bootstrap context |
| `instance_id` | Minted at B1 |
| `lifecycle_state` | `FAILED` |
| `severity` | `CRITICAL` or `ERROR` (producer-classified) |
| `occurred_at` | Captured once at failure detection |
| `event_id` | Minted once per startup attempt |
| `payload.phase` | Bounded bootstrap phase (see below) |
| `payload.reason_code` | Deterministic classifier output |
| `payload.exception_type` | `type(exc).__name__` |
| `payload.process_role` | Optional bounded caller field |
| `payload.source_kind` / `source_id` | Optional bounded composition locus |

### 9.3 Bootstrap phase taxonomy (frozen, minimal)

| Phase | Typical failure |
|-------|-----------------|
| `configuration` | Settings/env parse, authority config |
| `composition` | Harness/runtime wiring, registry |
| `dependency_resolution` | Queue/KV/causal store resolution |
| `worker_construction` | Worker factory (`create_kafka_worker`) |
| `startup` | `worker.start()` |

**Historical DG-A mapping:**

```text
create_kafka_worker missing causal_evidence_persistence
  → phase = worker_construction
  → exception_type = TypeError
  → reason_code = bootstrap_unhandled_exception (or composition_missing_dependency when classifier can detect)
```

### 9.4 Reason-code ownership

**Producer/composition layer** assigns `reason_code` via deterministic classifier. Unknown failures: `bootstrap_unhandled_exception`. **Forbidden:** raw `str(exc)` as grouping truth.

### 9.5 Sensitive data

**Frozen:** canonical payload **MUST NOT** persist arbitrary exception message text. Allowed: `phase`, `reason_code`, `exception_type`, `process_role`, bounded `source_kind`/`source_id`.

---

## 10. Producer boundary

### 10.1 Frozen installation point

**Platform reusable primitive** in `intergrax.hosting` (illustrative name: `run_guarded_hosted_process_bootstrap`):

```text
Product process entrypoint (e.g. background_worker_main.main)
    ↓
build HostedProcessBootstrapContext (application_id, process_role, tenant_binding, publishers)
    ↓
run_guarded_hosted_process_bootstrap(context, startup_callback)
    ↓  try: startup_callback()
    ↓  except Exception: emit canonical APPLICATION_FAILED → publishers → re-raise
```

**NOT** LKW-local `try/except` around `build_local_workspace_background_worker_wiring` except as thin composition of the platform primitive.

### 10.2 Callback scope

`startup_callback` wraps:

```text
settings → composition → worker wiring → worker.start
```

Product may split authority activation outside callback only if failures there are either (a) separately guarded, or (b) accepted as pre-B5 observability-only per coverage contract.

**Recommended LKW shape:** guard from post-B5 point through `worker.start()`; optionally outer guard for authority activation when B5 resolvable from composition.

---

## 11. Publisher/projection boundary

### 11.1 Publisher abstraction

**Reuse** `build_hosted_application_diagnostic_event_publisher` pattern:

```text
BootstrapFailurePublisher (conceptual) =
  ObservabilityHostedApplicationEventPublisher
  + optional HostedApplicationDiagnosticEventPublisher
```

Product injects via bootstrap context — no service locator, no hidden globals, no env reads inside reusable contract.

### 11.2 Signal conversion

```text
HostedApplicationEvent (APPLICATION_FAILED)
    → hosted_application_failure_to_problem_signal (existing adapter)
    → PlatformProblemSignal
    → DiagnosticSignalSubjectScope (tenant_id, application_id, instance_id)
    → DiagnosticOrchestrator.run (sync)
```

Projection adapter remains **deterministic, bounded, diagnostics-neutral**. Generalize naming only if needed; behavior unchanged.

### 11.3 Ordering

**Frozen:** observability publish **BEFORE** diagnostic projection. Projection failure **MUST NOT** rewrite canonical observability truth.

---

## 12. Failure/exit semantics

| Rule | Frozen value |
|------|--------------|
| Original bootstrap failure authoritative | **YES** |
| Diagnostic projection failure changes process failure | **NO** |
| Original exception preserved | **YES** — publish then re-raise |
| Exit code preserved | **YES** — non-zero on bootstrap failure |
| Swallowing forbidden | `except Exception: return 0` — **FORBIDDEN** |

### 12.1 Exception class boundary

**Catch `Exception` only** — do **not** classify `KeyboardInterrupt`, `SystemExit`, or `asyncio.CancelledError` (subclass of `BaseException`, not `Exception`) as application bootstrap failures. Aligns with HOST-DIAG-3 projection isolation (`except Exception` around diagnostics only).

---

## 13. Durability

| Rule | Frozen value |
|------|--------------|
| Synchronous projection before re-raise | **YES** when B5 available — `orchestrator.run()` completes in guarded path before re-raise |
| Unbounded shutdown wait | **FORBIDDEN** |
| Diagnostics outage blocks process exit indefinitely | **FORBIDDEN** |
| Diagnostics unavailable | Process fails correctly; observability/log fact remains; projection failure logged separately |

---

## 14. Grouping/idempotency

**Reuse DIAG-5B deterministic structural grouping** via existing `hosted_application_failure_to_problem_signal` fields:

- Grouping keys include: `error_code` (reason_code), `exception_type`, `source_component` (phase), signal structural fields
- **`instance_id` excluded** from stable Problem fingerprint — recurrence across process restarts groups under one Problem (proven: `test_recurrence_across_instances_and_replay` in hosted diagnostic integration)
- Individual occurrence retains: `instance_id`, `event_id`, `occurred_at`

**Event identity:** one `event_id` minted per process startup attempt at failure time; no regeneration on publisher retry.

---

## 15. Security/bounded payload

See §9.5. Raw traceback and exception message remain in controlled logs/observability per existing policy — **not** canonical Problem attributes.

---

## 16. Pluginability

Future entrypoints supply:

| Input | Required |
|-------|----------|
| `HostedProcessBootstrapContext` | `application_id`, `instance_id`, `process_role`, optional `tenant_binding` |
| `startup_callback` | Product composition |
| `failure_publisher` | Observability ± diagnostic composed publisher |

**No** Kafka/Celery/LKW parameters in platform contract.

**Pluginability verdict: PASS.**

---

## 17. Historical fixture

### 17.1 Fixture qualification design (future R5)

```text
REAL PROCESS START
  → REAL BOOTSTRAP CONTEXT (application_id, instance_id, tenant_binding, B5 wired)
  → CONTROLLED COMPOSITION FAILURE (injected factory omitting dependency before worker.start)
  → REAL APPLICATION_FAILED EVENT
  → REAL OBSERVABILITY EXPORT
  → REAL DIAGNOSTIC PROJECTION
  → REAL PROBLEM PERSISTENCE
  → REAL OPERATOR READ (list_problems)
  → PROCESS STILL FAILS (non-zero exit)
```

**Allowed:** deterministic injected dependency failure at composition boundary.  
**Forbidden:** synthetic Problem insertion, post-hoc persistence edits, manual publisher calls after failure.

### 17.2 Historical coverage

**`HISTORICAL DG-A FIXTURE COVERED BY TARGET DG-001B BOUNDARY = YES`**

When product wires B3–B5 from `ProductionProcessComposition` before guarded `build_local_workspace_background_worker_wiring`, `create_kafka_worker` `TypeError` at B6 is inside guarded callback and emits canonical `APPLICATION_FAILED`.

---

## 18. Frozen target architecture

```text
Product process entrypoint
    ↓
HostedProcessBootstrapContext (caller identity + tenant binding + publishers)
    ↓
intergrax.hosting.run_guarded_hosted_process_bootstrap(...)
    ↓
startup_callback: composition → worker factory → worker.start
    ↓ on Exception
HostedApplicationEvent (APPLICATION_FAILED, lifecycle_state=FAILED, bounded bootstrap phase)
    ↓
ObservabilityHostedApplicationEventPublisher          [B4+]
    ↓
hosted_application_failure_to_problem_signal          [B5+]
    ↓
DiagnosticSignalSubjectScope (APPLICATION_INSTANCE)
    ↓
DiagnosticOrchestrator (existing)
    ↓
Problem persistence (existing)
    ↓
operator discovery via DiagnosticReadService.list_problems
    ↓
re-raise original exception → non-zero process exit
```

**Invariants:**

- Execution identity: **NONE**
- New diagnostic store: **NONE**
- New orchestrator: **NONE**
- Worker-specific diagnostic engine: **FORBIDDEN**
- Queue-specific logic in diagnostics core: **FORBIDDEN**

---

## 19. Implementation slices

| Slice | Scope |
|-------|-------|
| **R2** | `HostedProcessBootstrapContext`, `HostedProcessBootstrapPhase`, `run_guarded_hosted_process_bootstrap`, bootstrap `APPLICATION_FAILED` factory helper in `intergrax.hosting` |
| **R3** | Bootstrap composed publisher (reuse HOST-DIAG-3 wiring); architecture test: diagnostics core isolation |
| **R4** | LKW `background_worker_main` conformance — supply context, wire B5 from composition, guard composition/start |
| **R5** | Real controlled-failure qualification fixture + operator read proof |

**Next task:** **DG-001B IMPLEMENTATION R2** (contract + guarded bootstrap primitive).

---

## 20. Non-claims

DG-001B architecture does **NOT** claim:

- All bootstrap failures always enter central diagnostics (B0–B2, unwired B3–B5)
- Execution tenant semantics at worker bootstrap
- Kafka/Celery/queue internals coverage
- DG-004 causal evidence or DG-005 RuntimeEvent topology changes
- DG-003 operator story projection
- Supervisor pre-engine failures (DG-001D scope)
- Public proof PYTHON-BOOTSTRAP failures (DG-001C scope)
- Default LKW production wiring includes diagnostic publisher without explicit product composition

---

## Test evidence (R1)

| Suite | Result |
|-------|--------|
| `test_hosted_application_diagnostic_integration.py` | **passed** (subset of 73 total) |
| `test_lkw_background_worker_queue_dependencies.py` | **passed** |
| `test_lkw_background_worker_authority.py` | **passed** |
| `test_hosted_application_events.py` | **passed** |
| `test_w3_supervisor_regression.py` | **passed** |
| `test_application_diagnostic_subjects.py` | **passed** |
| **Total** | **73 passed**, 0 failed |
| `--ignore` | **NO** |
| New skips | **NONE** |

Environment-dependent LLM/Ollama authority tests not in required set; no unrelated failures observed.
