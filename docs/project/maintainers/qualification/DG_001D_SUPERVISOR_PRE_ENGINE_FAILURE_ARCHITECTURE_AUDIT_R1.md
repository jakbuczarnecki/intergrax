# DG-001D — Supervisor pre-engine failure architecture audit (R1)

> **Task:** DG-001D-SUPERVISOR-PRE-ENGINE-FAILURE-ARCHITECTURE-AUDIT-R1  
> **Mode:** review-only / architecture freeze — **no implementation**  
> **Start HEAD:** `cf8370d2d65cb26578a9872d55f200076aa303be`  
> **Audit date:** 2026-09-07  
> **Parent:** DG-001 — [`DIAGNOSTIC_GAP_LEDGER.md`](DIAGNOSTIC_GAP_LEDGER.md) · prioritization — [`DG_001_REMAINING_PRE_EXECUTION_SURFACES_PRIORITIZATION.md`](DG_001_REMAINING_PRE_EXECUTION_SURFACES_PRIORITIZATION.md)  
> **Sibling qualified slice:** DG-001B — [`DG_001B_FINAL_CLOSURE_R6.md`](DG_001B_FINAL_CLOSURE_R6.md)

---

## 1. Verdict

**Audit R1: PASS** — supervisor pre-engine failure diagnostic gap confirmed; target architecture frozen with no blocking ambiguities.

| Decision | Frozen outcome |
|----------|----------------|
| Semantic owner | **`intergrax.hosting.supervisor`** detects; **`intergrax.hosting`** owns generic failure representation |
| Event reuse | **YES** — `HostedApplicationEvent` + `APPLICATION_FAILED` + `lifecycle_state=FAILED` |
| New event type | **NO** |
| Subject kind | **YES** — reuse `DiagnosticSubjectKind.APPLICATION_INSTANCE` |
| New subject kind | **NO** |
| Execution identity | **NONE** — no TaskId/RunId/AttemptId |
| Supervisor truth | **`HostedApplicationExitRecord`** preserved |
| Diagnostic evidence | **`HostedApplicationEvent(APPLICATION_FAILED)`** derived deterministically |
| R2 bootstrap primitive wrap | **NO** — supervisor state machine + projection helper |
| New guarded process-launch primitive | **NO** |
| Fault seam | **Existing `HostedApplicationEngineFactory`** (proof-owned failing factory) |
| Diagnostics core changes | **NONE** |
| DG-001D vs DG-001A | **Separate** — producer vs default composition coverage |

**DG-001 remains `PARTIALLY ADDRESSED`.** DG-001D implementation not yet shipped.

---

## 2. Current supervisor flow

### 2.1 Exact code path (`intergrax/hosting/supervisor/supervisor.py`)

```text
HostedApplicationSupervisor.run()
  ↓
instance_id = validate_instance_id(instance_id_generator())     [minted]
  ↓
HostedApplicationSupervisorLaunchContext(
    definition, instance_id, attempt_number, control)
  ↓
restart_evaluator.record_launch(...)
  ↓
try:
    engine = await _build_engine(launch)                        [engine_factory]
    _verify_engine_contract(engine, launch)
except HostedApplicationSupervisorError:
    exit_record = classifier.classify_exception(...)
    attempt_records.append(...)
    _maybe_schedule_restart(...)                                [RESTART_* events]
  ↓
[on success]
engine.run_until_stopped()                                      [out of DG-001D scope]
```

### 2.2 `_build_engine` failure translation

```text
engine_factory(launch)
  ↓ Exception
HostedApplicationSupervisorError("engine factory failed") from exc

invalid return type
  ↓
HostedApplicationSupervisorError("engine factory returned invalid type")

_verify_engine_contract mismatch
  ↓
HostedApplicationSupervisorError("<field> mismatch")
```

### 2.3 Identity availability before `engine_factory`

| Identity / dependency | Available before `_build_engine`? | Source |
|-------------------------|-----------------------------------|--------|
| `application_id` | **YES** | `self.definition.application_id` |
| `instance_id` | **YES** | minted at line 120, validated, passed in `launch` |
| `event_publisher` | **YES** | `HostedApplicationSupervisor.event_publisher` (constructor field) |
| `attempt_number` | **YES** | supervisor loop counter (restart context, not execution identity) |

### 2.4 Current `event_publisher` usage in supervisor

| Event type | Emitted? | Method |
|------------|----------|--------|
| `RESTART_REQUESTED` | YES | `_publish_restart_event_safe` |
| `RESTART_SCHEDULED` | YES | `_publish_restart_event_safe` |
| `RESTART_STARTED` | YES | `_publish_restart_event_safe` |
| `RESTART_EXHAUSTED` | YES | `_publish_restart_event_safe` |
| `APPLICATION_FAILED` | **NO** | — |
| Any other failure event | **NO** | — |

Restart events use `instance_id="supervisor"` (supervisor role context). Failed application instance identity is **not** published on any event in the pre-engine catch path.

**Conclusion:**

```text
SUPERVISOR PRE-ENGINE FAILURE = NO CANONICAL FAILURE EVENT
```

---

## 3. Exact gap

| Property | Current state |
|----------|---------------|
| Detection | **YES** — `HostedApplicationSupervisorError` caught |
| Supervisor lifecycle truth | **YES** — `HostedApplicationExitRecord` (`exit_kind=supervisor_error`, `reason_code=supervisor_error`, `retryable=false`) |
| Canonical `APPLICATION_FAILED` | **NO** |
| HOST-DIAG-3 projection | **NO** — nothing to project |
| Central Diagnostics `Problem` | **NO** |
| Operator visibility | Supervisor result / logs only |

The platform knows the supervisor attempt failed but emits **no** canonical hosting failure evidence consumable by Central Diagnostics.

---

## 4. Scope boundary

### 4.1 In scope (DG-001D)

Failures **after** `instance_id` is minted and **before** `HostedApplicationEngine.run_until_stopped()` begins:

| Failure locus | In scope | Notes |
|---------------|----------|-------|
| `engine_factory` raises | **YES** | Primary case |
| `engine_factory` returns invalid type | **YES** | `_build_engine` |
| `engine.instance_id` mismatch | **YES** | `_verify_engine_contract` |
| `profile_digest` mismatch | **YES** | contract validation |
| `definition_digest` mismatch | **YES** | contract validation |
| `application_id` mismatch | **YES** | contract validation |

### 4.2 Out of scope

| Boundary | Rationale |
|----------|-----------|
| `engine.run_until_stopped()` failures | Engine lifecycle already publishes `APPLICATION_FAILED` via `_publish_lifecycle_failed_event` |
| `stop_before_launch` | Clean stop — `exit_kind=clean_stop`, `reason_code=stop_before_launch`; **no** `APPLICATION_FAILED` |
| Scheduler / queue / Kafka | Explicitly forbidden |
| Execution Engine / LKW feature work | Explicitly forbidden |
| DG-001A default runner composition | Separate slice |

### 4.3 Double-event risk

Pre-engine catch path does **not** construct a running engine; engine failure publisher is unreachable. **No duplicate producer** exists today for this boundary. R2 must preserve **one canonical producer** (supervisor only) for pre-engine failures.

---

## 5. Semantic ownership

| Concern | Owner | Forbidden |
|---------|-------|-----------|
| Failure detection + restart truth | **`intergrax.hosting.supervisor`** | product apps, runtime/diagnostics, LKW |
| Bounded failure event contract | **`intergrax.hosting`** | diagnostics core |
| HOST-DIAG-3 composition | **`intergrax.applications._shared`** (product wiring) | supervisor imports |
| Problem interpretation | **Central Diagnostics** (`DiagnosticOrchestrator`) | supervisor |

**Frozen principle:** supervisor owns detection and lifecycle truth; hosting owns canonical failure evidence; Central Diagnostics owns interpretation.

---

## 6. Identity model

| Field | Authority | Notes |
|-------|-----------|-------|
| `application_id` | `HostedApplicationDefinition.application_id` | stable per definition |
| `instance_id` | `supervisor.instance_id_generator()` → `validate_instance_id` | **per supervisor attempt**; new ID on restart |
| `attempt_number` | supervisor loop counter | recurrence context only; **≠** execution `AttemptId` |
| TaskId / RunId / AttemptId / ExecutionId | **NOT MINTED** | correct absence |

Restart policy mints new `instance_id` per attempt — each failed attempt is a distinct `APPLICATION_INSTANCE` occurrence.

---

## 7. Tenant / publisher composition

### 7.1 Who constructs the publisher

| Layer | Responsibility |
|-------|----------------|
| `intergrax/hosting/runner.py` | `_default_runner_factories().create_event_publisher` → `ObservabilityHostedApplicationEventPublisher` |
| Product foreground (e.g. LKW) | Optional `event_publisher_factory` → `build_hosted_application_diagnostic_event_publisher(tenant_binding, orchestrator)` |
| `HostedApplicationSupervisor` | Receives `HostedApplicationEventPublisher` only — **tenant-agnostic** |

### 7.2 Frozen composition model

```text
supervisor remains tenant-agnostic
event_publisher already carries tenant-bound diagnostic composition when product wires HOST-DIAG-3
```

Supervisor **must not** import `HostedDiagnosticTenantBinding`, `DiagnosticOrchestrator`, or persistence.

Default runner path: observability-only publisher (DG-001A). Failure event publication is still required at producer level; Central Diagnostics ingestion depends on composed publisher (product / qualification).

---

## 8. Existing contracts reuse

| Capability | Location | Reuse for DG-001D |
|------------|----------|-------------------|
| `HostedApplicationEvent` + `APPLICATION_FAILED` | `intergrax/hosting/contracts/events.py` | **YES** |
| `HostedApplicationEventPublisher` | `intergrax/hosting/contracts/context.py` | **YES** |
| `hosted_application_failure_to_problem_signal` | `intergrax/applications/_shared/hosted_application_failure_projection.py` | **YES** — requires `phase` + `reason_code` in payload |
| `HostedApplicationDiagnosticEventPublisher` | `intergrax/applications/_shared/hosted_application_diagnostic_wiring.py` | **YES** — composition only |
| `HostedProcessBootstrapFailureFacts` + payload helper | `intergrax/hosting/process_bootstrap.py` | **YES** — extend phase enum; reuse payload shape (`phase`, `reason_code`, `exception_type`, `process_role`) |
| `run_guarded_hosted_process_bootstrap` | `intergrax/hosting/process_bootstrap.py` | **NO** — see §9 |
| `HostedApplicationFailurePhase` | `intergrax/hosting/engine/diagnostics.py` | **NO** — engine-internal lifecycle phases |
| `HostedApplicationExitRecord` | `intergrax/hosting/supervisor/classification.py` | **YES** — supervisor truth; separate from event payload |

### 8.1 Projector compatibility

`hosted_application_failure_to_problem_signal` requires:

- `event.event_type == APPLICATION_FAILED`
- bounded `payload.phase` (string)
- bounded `payload.reason_code` (string)
- optional `exception_type`, `source_kind`, `source_id`

`process_role` in bootstrap payload is **not** required by projector. Supervisor producer **must** supply `phase` + `reason_code` at minimum.

`DiagnosticSignalSubjectScope` uses `application_id` + `instance_id` → **`APPLICATION_INSTANCE`** subject (HOST-DIAG-3 wiring).

**No diagnostics core changes required** for supervisor failure shape if producer follows bounded facts contract.

---

## 9. Event reuse decision

**REUSE EXISTING EVENT** — confirmed.

`APPLICATION_FAILED` with `lifecycle_state=FAILED` semantically represents failure of a concrete hosted application instance before successful engine execution. Pre-engine supervisor failure is still an application instance failure.

**Do not introduce:** `ENGINE_CREATION_FAILED`, `SUPERVISOR_FAILED`, or parallel diagnostic event hierarchy.

---

## 10. Subject reuse decision

**REUSE `DiagnosticSubjectKind.APPLICATION_INSTANCE`.**

Before `engine_factory` completes:

- `application_id` exists
- `instance_id` exists
- no Task/Run exists

No new subject kind required.

---

## 11. R2 primitive decision

### 11.1 Option A — reuse `run_guarded_hosted_process_bootstrap` unchanged

**REJECTED.**

Supervisor already owns:

- exception handling
- `HostedApplicationSupervisorError` translation
- `HostedApplicationExitClassifier`
- restart policy state machine

Wrapping `engine_factory` with R2 primitive would risk:

- double classification
- duplicated `APPLICATION_FAILED` if both layers emit
- interference with restart semantics and error identity
- re-raise vs supervisor error contract mismatch

### 11.2 Option B — generalize R2

**NOT REQUIRED** for DG-001D. No broader primitive emerges from audit evidence.

### 11.3 Option C — supervisor-owned deterministic projection helper

**SELECTED (architecture).**

Conceptual split:

```text
supervisor_pre_engine_failure_to_hosted_event(...)   # pure, deterministic, no IO
_publish_failure_event_safe(...)                     # isolated publisher IO
```

Helper ownership: `intergrax/hosting/supervisor/` or sibling hosting failure module (not `applications/`, not `runtime/diagnostics`).

**No `run_guarded_hosted_process_launch` primitive required.**

---

## 12. Supervisor truth vs diagnostic evidence

| Artifact | Role | Mutability |
|----------|------|------------|
| `HostedApplicationExitRecord` | Supervisor restart / terminal decision truth | Unchanged by diagnostic publication |
| `HostedApplicationEvent(APPLICATION_FAILED)` | Canonical observability + diagnostic evidence | Derived from same detected failure |

Neither rewrites the other. Diagnostic `reason_code` values (e.g. `engine_factory_failed`) are **bounded diagnostic facts** — distinct from supervisor exit record `reason_code=supervisor_error`.

---

## 13. Failure projection point and ordering

### 13.1 Frozen insertion point

Inside pre-engine `except HostedApplicationSupervisorError` block, **after** classification, **before** `_maybe_schedule_restart`.

### 13.2 Frozen ordering

```text
classify → HostedApplicationExitRecord
  ↓
publish APPLICATION_FAILED (safe, isolated)
  ↓
append HostedApplicationSupervisorAttemptRecord
  ↓
_maybe_schedule_restart → RESTART_*
```

### 13.3 Causal event ordering invariant

```text
APPLICATION_FAILED (attempt N)
  → RESTART_REQUESTED
  → RESTART_SCHEDULED
  → RESTART_STARTED
```

when restart occurs for attempt N.

### 13.4 Recurrence invariant

```text
one detected pre-engine failure → exactly one APPLICATION_FAILED per supervisor attempt / instance_id
```

No duplicate during classification, restart scheduling, or result construction.

---

## 14. Restart semantics

| Property | Frozen behavior |
|----------|-----------------|
| `HostedApplicationExitClassifier` output | **Unchanged** by failure publication |
| `HostedApplicationRestartPolicyEvaluator` | **Unchanged** — uses `exit_record.retryable`, `exit_kind`, policy mode |
| `HostedApplicationSupervisorError` classification | `exit_kind=supervisor_error`, `retryable=false` |
| Restart on supervisor error | Only when policy mode is `ALWAYS` or custom classifier allows — publication does not alter |

Failure publication **must not** change retryability, exit kind, attempt numbering, backoff, or exhaustion.

---

## 15. Publication failure isolation

Existing patterns:

| Pattern | Location | Behavior |
|---------|----------|----------|
| `_publish_restart_event_safe` | `supervisor.py` | Swallows publisher exceptions (silent) |
| bootstrap failure publish | `process_bootstrap.py` | Logs error with bounded fields; re-raises bootstrap exception |

**Frozen for DG-001D:** publication failure **must not** replace supervisor failure or prevent restart policy. Prefer **logged secondary failure** (bootstrap pattern) over silent swallow for `APPLICATION_FAILED`.

Do not recursively invoke Central Diagnostics when publisher fails.

---

## 16. Payload / security

### 16.1 Bounded target payload

| Field | Value |
|-------|-------|
| `phase` | `engine_construction` or `engine_contract_validation` (see §17) |
| `reason_code` | deterministic bounded codes (e.g. `engine_factory_failed`, `engine_factory_invalid_type`, `engine_instance_id_mismatch`, …) |
| `exception_type` | `HostedApplicationSupervisorError` (public wrapper type) |
| `process_role` | `hosted_application_supervisor` |

Optional: supervisor `attempt_number` in payload if bounded and useful for operator context — **not** for Problem fingerprint root.

### 16.2 Forbidden in payload

- raw exception message
- stack trace
- config / environment dump
- engine factory repr
- `__cause__` traversal in production projection

### 16.3 Error wrapping audit

Current `_build_engine`:

```python
raise HostedApplicationSupervisorError("engine factory failed") from exc
```

Public typed original cause is **not** available except via exception chaining. **R2 preference:** deterministic `reason_code` mapping at catch site from known `HostedApplicationSupervisorError` message patterns (or future typed fields on supervisor error). **Do not** depend on ad hoc `__cause__` reflection in projection.

Exit classifier uses generic `reason_code=supervisor_error` — diagnostic payload uses **separate** bounded diagnostic reason codes.

---

## 17. Failure phase taxonomy

### 17.1 Audit of existing enums

| Enum | Suitability |
|------|-------------|
| `HostedProcessBootstrapPhase` | **Extend** — generic hosting pre-run failure taxonomy (DG-001B bootstrap + DG-001D supervisor) |
| `HostedApplicationFailurePhase` | **NO** — engine runtime lifecycle only |
| New supervisor-local strings | **FORBIDDEN** |

### 17.2 Frozen phase values (R2 enum extension)

| Failure | Phase |
|---------|-------|
| `engine_factory` exception / invalid type | `engine_construction` |
| `_verify_engine_contract` mismatches | `engine_contract_validation` |

Do **not** use `worker_construction`, `process_launch`, or `startup` for supervisor in-process engine minting.

R2 may add to `HostedProcessBootstrapPhase` or rename enum to broader hosting name — **one taxonomy only**, no duplicates.

---

## 18. Grouping / recurrence

Deterministic Problem grouping (via existing HOST-DIAG-3 projector + grouping strategy):

```text
same application_id
+ same structural phase
+ same reason_code
+ same exception_type
→ one Problem
```

Different `instance_id` → separate occurrences. **No grouping by supervisor `attempt_number`.**

Repeated engine factory failures across restarts: one structural Problem, multiple `APPLICATION_INSTANCE` occurrences (ideal R4 qualification scenario).

---

## 19. Implementation slices

| Slice | Scope |
|-------|-------|
| **R2 — Producer** | `DG-001D-SUPERVISOR-PRE-ENGINE-FAILURE-PRODUCER-R2` — projection helper, supervisor emission, phase enum extension, unit/architecture tests |
| **R3 — Conformance** | HOST-DIAG-3 composed publisher + supervisor event shape |
| **R4 — Qualification** | Real supervisor + failing `HostedApplicationEngineFactory` + durable Problem + operator read + restart recurrence |

**No production fault flags.** Fault injection via proof-owned failing factory only.

### 19.1 Future test matrix (R2+)

1. engine factory raises → `APPLICATION_FAILED`
2. invalid engine type → `APPLICATION_FAILED`
3. instance mismatch → `APPLICATION_FAILED`
4. profile mismatch → `APPLICATION_FAILED`
5. definition mismatch → `APPLICATION_FAILED`
6. application_id mismatch → `APPLICATION_FAILED`
7. stop-before-launch → no failure event
8. successful engine creation → no pre-engine failure event
9. publisher failure → restart unaffected
10. failure event before restart events
11. new `instance_id` per restart
12. structural recurrence grouping
13. no execution identity
14. no raw exception message in payload
15. no diagnostics-core import in supervisor

### 19.2 Real qualification target (R4)

```text
real supervisor
→ controlled engine_factory failure
→ canonical APPLICATION_FAILED
→ observability
→ HOST-DIAG-3
→ durable Problem
→ operator read
→ restart semantics preserved
```

Platform hosting-level proof preferred over LKW-specific path.

---

## 20. Frozen decisions

| Decision | Outcome |
|----------|---------|
| Semantic owner | `intergrax.hosting.supervisor` |
| Event | Reuse `HostedApplicationEvent` / `APPLICATION_FAILED` |
| Subject | `APPLICATION_INSTANCE` |
| New event / subject | **NO** |
| Execution identity | **NONE** |
| Supervisor truth | `HostedApplicationExitRecord` |
| Diagnostic evidence | `HostedApplicationEvent` |
| Fault seam | `HostedApplicationEngineFactory` |
| New fault injection framework | **NO** |
| R2 primitive direct reuse | **NO** |
| New process-launch primitive | **NO** |
| Helper approach | **Option C** — deterministic projection in supervisor flow |
| Phase taxonomy | Extend `HostedProcessBootstrapPhase` with `engine_construction`, `engine_contract_validation` |
| `process_role` | `hosted_application_supervisor` |
| Boundary | After `instance_id` minted, before `run_until_stopped()` |
| `stop_before_launch` | No `APPLICATION_FAILED` |
| Runtime engine failures | Out of scope (existing engine producer) |
| Restart semantics | Unchanged |
| Event ordering | `APPLICATION_FAILED` before `RESTART_*` |
| One event per attempt | **YES** |
| Publication failure | Isolated; restart continues; log secondary failure |
| Tenant in supervisor | **NO** direct injection |
| Publisher composition | **YES** — product/HOST-DIAG-3 |
| Diagnostics core | **NONE** |
| Queue | **NONE** |
| DG-001D vs DG-001A | **Separate concerns** |
| Next task | `DG-001D-SUPERVISOR-PRE-ENGINE-FAILURE-PRODUCER-R2` |

---

## Non-claims

- This audit does **not** implement the producer.
- This audit does **not** close DG-001D or DG-001.
- This audit does **not** qualify HOST-DIAG-3 on default runner (DG-001A).
- This audit does **not** change restart event `instance_id="supervisor"` convention.
