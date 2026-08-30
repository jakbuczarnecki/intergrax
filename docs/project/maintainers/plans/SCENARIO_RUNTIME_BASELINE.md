# SCENARIO-PLATFORM-2 — Scenario Runtime Baseline & Lifecycle Contract

**Contract ID:** SCENARIO-PLATFORM-2  
**Date:** 2026-08-27  
**Branch:** `development`  
**Mode:** architecture / contract only — **no generator implementation**  
**Repository:** jakbuczarnecki/intergrax  
**Supersedes (partial):** SCENARIO-PLATFORM-1 §5 baseline assumptions that implied GraphExecutor + manual identity as default

---

## 1. Executive decision

**Recommendation: C — shared Scenario runtime facade (default), delegating to NexusLoop.**

| Option | Verdict |
|--------|---------|
| **A. GraphExecutor direct** | **Not** canonical Scenario application entry |
| **B. NexusLoop direct** | Correct underlying runtime, but callers must not duplicate `HarnessHostRuntime` wiring |
| **C. Shared facade over platform primitives** | **Canonical baseline** — thin composition only, no second orchestrator |

**Default underlying runtime:** `NexusLoop` (which already owns an internal `GraphExecutor`).  
**GraphExecutor direct** remains a legitimate **lower-level primitive** for unit/integration tests, conformance proofs, and internal Nexus mechanics — not for generated Scenario application entry.

---

## 2. Why Nexus-backed facade (not GraphExecutor direct)

Evidence from current `development` spine:

| Requirement | GraphExecutor direct (`ai_incident` today) | NexusLoop + factory + diagnostic wiring |
|-------------|-------------------------------------------|----------------------------------------|
| Canonical terminal `RuntimeEvent` truth | **Absent** — no `NexusRuntimeEventPublisher.publish_terminal` |
| Persisted execution truth | **Manual** — requires explicit `event_bus` + store wiring |
| Terminal diagnostic bridge | **Absent** — no `_publish_terminal_runtime_event` → `invoke_terminal_execution_diagnostics` |
| Execution identity ownership | **Manual** — `mint_run_id`, `mint_attempt_id`, `bind_active_execution_identity` in scenario code |
| Full task lifecycle | **Absent** — graph execution only |
| Policy / guardrail / security | **Manual / partial** — not applied via `nexus_factory` |
| Retry / budget machinery | **Partial** — retry engine present; budget slot absent |
| ToolRuntime (UAEP path) | **Manual** — scenario builds `ApplicationBuildContext` locally |
| Observability subscriptions | **Absent** unless explicitly bridged |

`ai_incident_investigation` (`platform_proofs/scenarios/ai_incident_investigation/scenario.py`) is a **behavior reference**, not the desired future baseline. Its manual orchestration boundary work is **legacy composition debt**, not a pattern to cargo-cult.

---

## 3. Capability comparison matrix

Classification: **NATIVE** · **MANUAL** · **OPTIONAL** · **ABSENT**

| Invariant / capability | GraphExecutor direct | NexusLoop | Shared facade possible? |
|------------------------|---------------------|-----------|-------------------------|
| Task lifecycle (intake → plan → execute → finalize) | ABSENT | NATIVE | NATIVE (delegates) |
| `TaskId` | NATIVE (`Task()` mints) | NATIVE | NATIVE |
| `RunId` | MANUAL (`mint_run_id` + bind) | MANUAL at `handle_task` boundary; platform owns mint in executor | NATIVE (facade mints once) |
| `AttemptId` | MANUAL | MANUAL at boundary; platform owns mint in executor | NATIVE |
| Active execution identity | MANUAL (`bind_active_execution_identity`) | NATIVE in `handle_task` | NATIVE |
| `RuntimeEvent` persistence | OPTIONAL (explicit bus + store) | OPTIONAL (`runtime_events_db_path` / env / explicit store) | NATIVE when profile selects store |
| Terminal `RuntimeEvent` publication | ABSENT | NATIVE (`_publish_terminal_runtime_event`) | NATIVE |
| Diagnostic trigger (`TerminalExecutionDiagnosticTrigger`) | ABSENT | OPTIONAL (`attach_terminal_diagnostic_trigger`) | NATIVE when persistence prerequisites met |
| `ToolRuntime` / UAEP tool trace | OPTIONAL (`declarative_tool_invoker` + `RuntimeContext`) | OPTIONAL (factory + `ApplicationBuildContext`) | NATIVE via environment wiring |
| `RuntimeState` / `trace_event` | OPTIONAL (agent `build_context` path) | OPTIONAL (same agent path) | NATIVE when agent path used |
| `TraceEvent` / run trace persistence | OPTIONAL | OPTIONAL (`trace_store` via observability wiring) | NATIVE per profile |
| `ToolCallTrace` | OPTIONAL (ToolRuntime path) | OPTIONAL | NATIVE when tools enabled |
| Retry engine | NATIVE (`RetryEngine`) | NATIVE | NATIVE |
| Policy engine | OPTIONAL (engine wiring) | NATIVE (`PolicyEngine` + pre-output bridge) | NATIVE |
| Guardrails | ABSENT | OPTIONAL (`apply_application_guardrail_wiring`) | NATIVE (permissive default in lab) |
| Security wiring | ABSENT | OPTIONAL (`apply_application_security_wiring`) | NATIVE |
| Critic | OPTIONAL (`critic_graph_hooks`) | OPTIONAL (`ApplicationCriticWiring`) | OPTIONAL (capability-driven) |
| `RunBudget` | ABSENT | OPTIONAL | OPTIONAL |
| Checkpoint / recovery | OPTIONAL (graph metadata) | OPTIONAL (`reliability_profile`) | OPTIONAL |
| `ContextManager` / tenant in events | OPTIONAL | NATIVE when composed | NATIVE |
| Tenant propagation (`Task.tenant_id`) | MANUAL (caller sets `Task`) | NATIVE (publisher injects from task) | NATIVE (validated at facade) |
| Classify / plan orchestration | ABSENT | NATIVE | NATIVE (configurable via `ApplicationEnvironmentProfile`) |
| HITL / hosting / memory / RAG / web | ABSENT | OPTIONAL (profile-driven) | OPTIONAL |

**Facade rule:** the facade **composes** existing platform mechanisms; it does **not** implement orchestration, event publishing, or diagnostic logic.

---

## 4. Canonical execution truth (`RuntimeEvent`)

### 4.1 Nexus guarantees (when correctly wired)

When `NexusLoop` completes a task, `_finish_task` calls `_publish_terminal_runtime_event`, which:

1. Publishes a terminal `RuntimeEvent` via `NexusRuntimeEventPublisher.publish_terminal`
2. If `terminal_diagnostic_trigger` is attached, invokes `invoke_terminal_execution_diagnostics`

**Nexus already guarantees canonical terminal execution truth** — scenario code must **not** manually publish terminal events.

### 4.2 Required factory / composition arguments

`resolve_runtime_event_persistence()` returns **`None` by default** when neither `runtime_events_db_path` nor `INTERGRAX_RUNTIME_EVENTS_DB` is set (`intergrax/runtime/events/store.py`).  
Therefore: **passing `NexusLoop` alone does not imply persisted `RuntimeEvent` truth.**

| Profile | Required composition |
|---------|---------------------|
| **LAB / synthetic proof** | `wire_nexus_observability(..., runtime_events_db_path=<scoped temp path>)` **or** explicit in-memory `RuntimeEventPersistence` for isolated tests |
| **PRODUCTION-ATTACHED** | `wire_application_observability` / `wire_nexus_observability` with integration profile + durable paths (same as `build_harness_host_runtime`) |

`runtime_event_bus=None` is acceptable: `NexusLoop` constructs `RuntimeEventBus(persistence=self._runtime_event_store)` internally. Persistence depends on **`runtime_event_store` resolution**, not on an externally supplied bus.

### 4.3 Forbidden

- Scenario-specific `RuntimeEvent` terminal publishers
- Scenario code assuming persistence without explicit observability profile

---

## 5. Diagnostic write path

### 5.1 Production composition chain (verified)

```
NexusLoop._finish_task
  → _publish_terminal_runtime_event
    → NexusRuntimeEventPublisher.publish_terminal  (persisted RuntimeEvent)
    → invoke_terminal_execution_diagnostics
      → TerminalExecutionDiagnosticTrigger.trigger_for_terminal_execution
        → DiagnosticOrchestrator.run
          → ProblemLifecycleEngine  (Problem write)
```

Shared wiring today: `build_harness_host_runtime` (`intergrax/applications/_shared/harness_host_runtime.py`) calls:

- `try_build_terminal_execution_diagnostic_trigger(env_wiring, observability)`
- `nexus_loop.attach_terminal_diagnostic_trigger(...)` when dependencies resolve

Dependencies (`diagnostic_runtime_wiring.py`):

- `env_wiring.build_context.tool_wiring_context.document_store` (non-null)
- `observability.runtime_event_store` (non-null)

### 5.2 Scenario baseline requirement

Future scenario runtime must call **one generic shared builder** (extracted from harness host path) that returns:

- runtime event store (observability)
- terminal diagnostic trigger (when prerequisites met)
- shared Problem persistence (via document store)

**No scenario-local knowledge** of `DiagnosticOrchestrator` assembly.

When prerequisites are missing (lab without document store), diagnostic write is **skipped gracefully** (`trigger=None`) — execution truth may still persist if runtime event store is configured.

### 5.3 Domain diagnostic payload

Typed `DiagnosticPayload` subclasses remain **scenario-owned**. Runtime baseline exposes emission via standard agent / `RuntimeState.trace_event` path — not generic generated payload classes.

---

## 6. Execution identity ownership

| Identity | Owner | Scenario application code |
|----------|-------|---------------------------|
| `TaskId` | `Task()` default factory **or** caller-supplied validated id | May accept optional `task_id` in request; must not invent ad-hoc formats |
| `RunId` | **Platform composition boundary** (`NexusLoopTaskExecutor`, facade, or `ExecutionBoundary`) | **Must not** call `mint_run_id()` |
| `AttemptId` | **Platform** (`NexusLoop.handle_task` mints when omitted) | **Must not** call `mint_attempt_id()` |
| Active identity binding | **NexusLoop.handle_task** (or `ExecutionBoundary`) | **Must not** call `bind_active_execution_identity()` |

`ai_incident` manual binding (`scenario.py` L315–326) is **legacy** — generator must not reproduce it.

---

## 7. Task / request contract

### 7.1 Application-facing input

**Canonical request envelope: `Task`** with **required explicit `tenant_id`.**

```python
task = Task(
    tenant_id=tenant_id,  # required — no ContextVar fallback
    user_id=...,
    message=...,
    context=TaskContext(capability=...),
)
```

Optional wrapper for generator ergonomics (facade-owned, not a new identity model):

```python
@dataclass(frozen=True)
class ScenarioExecutionRequest:
    tenant_id: str
    message: str
    user_id: str = "scenario-user"
    capability: str | None = None
    task_id: TaskId | None = None  # optional explicit id
```

Facade converts → `Task` and invokes `nexus_loop.handle_task(task, run_id=mint_run_id())`.

### 7.2 Agent runtime context

Agents use existing **`RuntimeRequest`** + `build_runtime_context_from_environment` (as in `ai_incident` `runtime_composition.py`).  
Do **not** invent `ScenarioRunRequest` as a parallel task identity model.

### 7.3 Tenant invariant

```
caller tenant_id
  = Task.tenant_id
  = runtime tenant (events, traces)
  = observability tenant
  = diagnostic tenant
```

- **Standalone synthetic proof:** explicit named constant (e.g. `scenario-tenant-synthetic`) — documented, not implicit.
- **No** `ContextVar` / global tenant source.
- Facade validates non-empty `tenant_id` before execution.

---

## 8. Nexus factory ownership & dependency direction

| Layer | Responsibility |
|-------|----------------|
| `intergrax/applications/_shared/nexus_factory.py` | `build_nexus_loop_from_environment` — **reuse directly** |
| `intergrax/applications/_shared/harness_host_runtime.py` | Full Tier-3 host spine — **reference extraction source** |
| **New (SCENARIO-PLATFORM-3A):** `intergrax/applications/_shared/scenario_runtime_baseline.py` | Lighter scenario facade: environment → observability → nexus → diagnostic trigger |
| `platform_proofs/scenarios/<slug>/` | Domain agents, tools, payloads, proof projection — **imports platform shared only** |

**Forbidden dependency:** `intergrax/` → `platform_proofs/`.

**Forbidden:** copying `nexus_factory` / `harness_host_runtime` internals into `platform_proofs/_shared/`.

Scenarios may import `intergrax.applications._shared.*` — same as Tier-3 applications.

---

## 9. Proposed facade surface (contract only — not implemented in -2)

Minimal types for SCENARIO-PLATFORM-3A:

```python
@dataclass(frozen=True)
class ScenarioRuntimeComposition:
    environment: ApplicationEnvironmentProfile
    env_wiring: ApplicationEnvironmentWiring
    observability: NexusObservabilityStores
    nexus_loop: NexusLoop

@dataclass(frozen=True)
class ScenarioExecutionResult:
    task_result: TaskResult
    run_id: RunId
    task_id: TaskId
    tenant_id: str
    # domain projection remains scenario-owned

def build_scenario_runtime_from_environment(
    *,
    environment: ApplicationEnvironmentProfile,
    registry: AgentRegistry,
    tenant_id: str,
    runtime_events_db_path: Path | None = None,
    trace_db_path: Path | None = None,
    lab_profile: bool = True,
    ...
) -> ScenarioRuntimeComposition: ...

async def execute_scenario_task(
    composition: ScenarioRuntimeComposition,
    request: ScenarioExecutionRequest,
) -> ScenarioExecutionResult: ...
```

**Not allowed:** `ScenarioEngine`, `ScenarioLoop`, `ScenarioScheduler`, or any type that reimplements Nexus intake/plan/execute/finalize.

---

## 10. Mandatory baseline invariants

Every production-capable Scenario run **must** provide:

| # | Invariant | Mechanism |
|---|-----------|-----------|
| B1 | Explicit `tenant_id` on `Task` | `ScenarioExecutionRequest` / caller |
| B2 | Platform-owned `RunId` / `AttemptId` | Facade → `handle_task` |
| B3 | `ApplicationEnvironmentProfile` (no duplicate manifest) | `lab_defaults` or `harness_production_defaults` + scenario overrides |
| B4 | `ApplicationBuildContext` with `RuntimePolicyBundle` slot | Even if empty/default |
| B5 | Observability store selection per profile | `wire_nexus_observability` / `wire_application_observability` |
| B6 | Terminal `RuntimeEvent` truth on success/failure | Nexus finalize path |
| B7 | Standard operational failure semantics | `TaskResult` / `TaskState` — separate from domain outcome |
| B8 | Tool declaration via `ToolProfile` when tools used | No manual trace reproduction |
| B9 | Security + guardrail wiring on same path as production | Permissive config in lab, same code path |
| B10 | No scenario terminal event publisher | Platform only |

---

## 11. Optional capabilities (opt-in via `ApplicationEnvironmentProfile`)

Reuse existing profile bundles — **no `ScenarioCapabilityManifest`:**

| Capability | Profile / wiring hook |
|------------|----------------------|
| Critic | `CriticProfile` + `ApplicationCriticWiring` via factory |
| EvaluatorLoop | Critic hooks + `EvaluatorLoopSpec` on nodes (domain) |
| HITL | `reliability_profile` / human decision store |
| Problem read | `DiagnosticReadService` at composition seam (scenario-owned DTO) |
| `IncidentInvestigationInput` | Domain contract — investigation scenarios only |
| RAG / web / memory | `ContextProfile`, `MemoryProfile` |
| LLM | `resolve_environment_llm_adapter` / `ApplicationEnvironmentProfile` cognition bundle |
| Hosting | `HarnessHostRuntime` full path — hosting scenarios only |
| Root-cause adjudication | Domain + Critic — not baseline |
| Causal evidence | Observability persistence + document store |
| Long-running / checkpoint | `ReliabilityProfile` |
| Budget claims | `CostProfile` / `RunBudget` |

Enabling optional capabilities = typed platform composition, not local re-wiring.

---

## 12. Observability ownership

| Class | Owner | Automatic in baseline? |
|-------|-------|------------------------|
| Execution / tool / runtime lifecycle (`RuntimeEvent`, `TraceEvent`, `ToolCallTrace`) | Platform via Nexus + observability wiring | **Yes** (per profile) |
| Domain decisions (planner objective, claim proposal, investigation conclusion) | Scenario agents via `RuntimeState.trace_event` + typed `DiagnosticPayload` | **No** — domain-owned |
| Proof projection (`PlatformProofEvidence`) | `scripts/proof` + scenario `evidence_builder.py` | Proof layer only |

---

## 13. Failure semantics

Keep layers separate:

| Layer | Examples |
|-------|----------|
| **Platform operational** | `TaskState.FAILED`, `TaskState.CANCELLED`, validation errors, policy denial |
| **Scenario domain** | `UNRESOLVED`, `SUPPORTED`, `NOT_ACCEPTED` (investigation-specific) |
| **Business negative** | "hypothesis rejected" ≠ runtime failure |

Do **not** create one universal scenario outcome enum.

---

## 14. Runtime profiles (LAB vs PRODUCTION-ATTACHED)

| Aspect | LAB / controlled proof | PRODUCTION-ATTACHED |
|--------|------------------------|---------------------|
| `ApplicationEnvironmentProfile` | `lab_defaults(profile_id="scenario.<slug>")` | `harness_production_defaults` or app-specific strict profile |
| Runtime events | Temp SQLite under `.tmp/` or explicit in-memory for unit tests | Durable `runtime_events_db_path` / integration profile |
| Trace store | `use_in_memory_trace=True` allowed for fast proofs | Durable trace store |
| Diagnostic Problem write | Skipped if no document store | `try_build_terminal_execution_diagnostic_trigger` → full chain |
| Tenant | `scenario-tenant-synthetic` (explicit constant) | Caller-supplied real tenant |
| Security / guardrail | Same wiring, permissive policy config | Strict `ExecutionMode.STRICT` |

Both profiles use **identical contracts** — only configuration differs.

---

## 15. GraphExecutor — legitimate roles

| Role | Allowed |
|------|---------|
| Internal Nexus mechanism | Yes (default) |
| Unit / integration test primitive | Yes |
| Conformance / low-level proof | Yes, when testing graph semantics in isolation |
| **Generated Scenario application entry** | **No** |

Fixed-topology scenarios (single-agent proof graphs) still enter via **NexusLoop** with environment profile configuring deterministic classifier/planner — or register capability routing so Nexus produces the intended graph. Bypassing Nexus for convenience forfeits terminal truth and diagnostic spine.

---

## 16. `ai_incident_investigation` — reusable vs legacy

| Artifact | Status |
|----------|--------|
| `runtime_composition.py` (`ApplicationEnvironmentProfile` + `ApplicationBuildContext`) | **Reusable pattern** — migrate to shared facade |
| `build_agent_runtime_context` / `resolve_scenario_llm_adapter` | **Reusable** — LLM opt-in seam |
| `scenario_composition.py` (diagnostic read) | **Reusable seam pattern** — domain DTO stays investigation-specific |
| `investigation_observability.py` (`DiagnosticPayload`) | **Domain-owned** |
| `scenario.py` GraphExecutor + manual identity | **Legacy** — do not generate |
| Incident-specific Critic adapter | **Domain-owned** |
| `STANDALONE_SCENARIO_TENANT_ID` | **Acceptable** only as named synthetic tenant constant via facade contract |

---

## 17. Target flow (canonical)

```text
ScenarioExecutionRequest (tenant_id, message, capability)
  → build_scenario_runtime_from_environment(
        ApplicationEnvironmentProfile,
        AgentRegistry,
        observability profile,
    )
  → wire_application_environment / observability / reliability / security / guardrail
  → build_nexus_loop_from_environment(...)
  → try_build_terminal_execution_diagnostic_trigger → attach_terminal_diagnostic_trigger
  → Task(tenant_id=..., ...)
  → nexus_loop.handle_task(task, run_id=mint_run_id())   # identity owned here
      → classify → plan → GraphExecutor (internal)
      → _finish_task → terminal RuntimeEvent → diagnostic trigger (if wired)
  → ScenarioExecutionResult(task_result, run_id, task_id, tenant_id)
  → scenario domain projection (outcome, DiagnosticPayload, proof evidence)
  → run_proof / PlatformProofEvidence (proof layer)
```

---

## 18. Lifecycle & promotion contract

### 18.1 Lifecycle states (conceptual)

| State | Meaning | Generator |
|-------|---------|-----------|
| `DESIGN / NOT YET ACCEPTED` | Design package only | `create_scenario_proof.py` (today) |
| `ACCEPTED FOR IMPLEMENTATION` | Human quality gate passed | — |
| `IMPLEMENTATION INITIALIZED` | Skeleton generated | `init_scenario_implementation.py` (**SCENARIO-PLATFORM-3B — implemented**) |
| `EXECUTABLE / NOT YET VERIFIED` | Runnable stub + proof descriptor | future |
| `VERIFIED` | Public Library proof accepted | manual / CI |

Constant `LIFECYCLE_ACCEPTED_FOR_IMPLEMENTATION` exists in `create_scenario_proof.py` but is **not enforced** today.

### 18.2 `init_scenario_implementation` (future contract)

**Inputs:** `scenario_slug`, optional capability / config profile selection.

**Preconditions (machine-checkable):**

| Check | Method |
|-------|--------|
| Design package exists | `platform_proofs/scenarios/<slug>/SCENARIO_SPEC.md` |
| Accepted for implementation | See §18.3 |
| INTERGRAX FIT ≠ `NOT YET PERFORMED` | Structured section or frontmatter field |
| GAP DECISION resolved | Structured section or frontmatter field |
| Observability / diagnostics contract complete | Section presence + required headings (existing doc contract tests) |
| APPLICATION vs PROOF section filled | Table in `SCENARIO_SPEC.md` |

**Output:** implementation skeleton only (rails, stubs, gates) — **no domain logic**.

Initialized scenario implementations are automatically subject to universal architecture
conformance via `scripts/proof/scenario_architecture_conformance.py` before lifecycle
promotion to `IMPLEMENTATION_INITIALIZED`.

### 18.3 Machine-readable lifecycle metadata (recommended)

String search for `ACCEPTED FOR IMPLEMENTATION` in markdown is **fragile** (false positives, manual edits).

**Recommendation for SCENARIO-PLATFORM-3B:** minimal YAML frontmatter on `SCENARIO_SPEC.md`:

```yaml
---
scenario_slug: my_scenario
lifecycle_status: ACCEPTED_FOR_IMPLEMENTATION
intergrax_fit: COMPLETED
gap_decision: RESOLVED
---
```

**SCENARIO-PLATFORM-3B — implemented:** minimal YAML frontmatter on `SCENARIO_SPEC.md` (see `scripts/proof/scenario_lifecycle.py`). Legacy specs without frontmatter parse as `LEGACY`; `init_scenario_implementation` fails with `lifecycle metadata required`.

---

## 19. Generator prerequisites (SCENARIO-PLATFORM-3)

Before `init_scenario_implementation` runs:

1. `lifecycle_status: ACCEPTED FOR IMPLEMENTATION` (or equivalent frontmatter)
2. `intergrax_fit: COMPLETED`
3. `gap_decision: RESOLVED`
4. Observability / diagnostics / APPLICATION-vs-PROOF sections present (extend existing `test_create_scenario_proof` patterns)
5. Scenario slug matches directory name
6. No implementation artifacts from `DESIGN_STAGE_FORBIDDEN_ARTIFACT_NAMES`

Generated implementation must include:

- `runtime_composition.py` stub calling shared `build_scenario_runtime_from_environment`
- `scenario.py` entry using facade `execute_scenario_task` — **not** GraphExecutor direct
- explicit `tenant_id` parameter on public entrypoints
- `proof.json` stub (v3 schema)
- architecture gate test module template

---

## 20. Platform gaps (do not patch in -2)

| Gap | Impact | Target slice |
|-----|--------|--------------|
| ~~No `scenario_runtime_baseline.py` shared builder~~ | ~~Every scenario would duplicate `harness_host_runtime` subset~~ | **SCENARIO-PLATFORM-3A — implemented** |
| `init_scenario_implementation` absent | No gated skeleton generation | **SCENARIO-PLATFORM-3B — implemented** |
| Lifecycle frontmatter not defined in generator | Fragile promotion gate | **SCENARIO-PLATFORM-3B — implemented** |
| Lab observability profile not codified for scenarios | Authors may omit runtime event store | **SCENARIO-PLATFORM-4** |
| `ai_incident` still on GraphExecutor path | Reference diverges from contract | **SCENARIO-PLATFORM-5** (migration) |
| Universal scenario architecture CI gates | Only ai_incident tested | **SCENARIO-PLATFORM-6** |
| Diagnostic write requires document store | Lab proofs skip Problem write unless store configured | Document in profile; optional lab document store stub in -4 |

---

## 21. Implementation slices (derived)

| Slice | Scope | Status |
|-------|-------|--------|
| **SCENARIO-PLATFORM-3A** | `intergrax/applications/_shared/scenario_runtime_baseline.py` — `build_scenario_runtime_from_environment`, `execute_scenario_task` | **Implemented** |
| **SCENARIO-PLATFORM-3B** | `init_scenario_implementation.py` + lifecycle frontmatter spec + precondition gates | **Implemented** |
| **SCENARIO-PLATFORM-4** | Observability + diagnostic baseline profiles (lab temp DB vs production-attached); document store optional lab stub |
| **SCENARIO-PLATFORM-5** | Migrate `ai_incident` to facade baseline (behavior parity proofs) |
| **SCENARIO-PLATFORM-6** | Universal `platform_proofs/scenarios/*` architecture gates in CI |

---

## 22. P0 findings

1. **GraphExecutor-only scenario path cannot produce canonical execution truth** without reimplementing Nexus terminal publishing.
2. **`runtime_events_db_path` is not implicit** — Nexus default persistence is `None` without path/env.
3. **Diagnostic write already exists on harness path** — scenarios must reuse `try_build_terminal_execution_diagnostic_trigger`, not invent bridges.
4. **Identity minting in scenario code is a bug pattern** — platform executor must own it.
5. **SCENARIO-PLATFORM-1 §5 incorrectly listed manual identity minting as AUTO baseline** — corrected here.
6. **No shared scenario composition primitive** — highest delivery risk for -3 generator.

---

## 23. Remaining questions

1. Should fixed-topology proof scenarios use a **deterministic planner stub** registered in `AgentRegistry`, or a **profile flag** to skip dynamic replanning only?
2. Minimum lab document store: in-memory stub for diagnostic write proofs, or accept skip in lab?
3. Should `ScenarioRuntimeComposition` expose `HarnessHostRuntime` directly for hosting scenarios, or compose a subset?

---

## 24. References (code)

| Path | Role |
|------|------|
| `intergrax/runtime/nexus/nexus_loop.py` | Terminal event + diagnostic trigger |
| `intergrax/applications/_shared/nexus_factory.py` | Nexus composition from profile |
| `intergrax/applications/_shared/harness_host_runtime.py` | Production spine reference |
| `intergrax/applications/_shared/diagnostic_runtime_wiring.py` | Diagnostic trigger builder |
| `intergrax/runtime/diagnostics/terminal_execution_diagnostic_bridge.py` | Post-terminal invoke |
| `intergrax/runtime/nexus/observability_wiring.py` | Runtime event store resolution |
| `intergrax/runtime/events/store.py` | Persistence default (`None`) |
| `intergrax/runtime/interactions/task_executor.py` | `mint_run_id` at executor boundary |
| `platform_proofs/scenarios/ai_incident_investigation/scenario.py` | Legacy GraphExecutor path |

---

---

## 25. SCENARIO-PLATFORM-3A — implemented public API

**Module:** `intergrax/applications/_shared/scenario_runtime_baseline.py`

### Types

| Type | Role |
|------|------|
| `ScenarioRuntimeComposition` | Immutable build result: `environment`, `env_wiring`, `observability`, `registry`, `nexus_loop`, `tenant_id`, `security_wiring`, `guardrail_wiring`; status via `has_runtime_event_store`, `has_terminal_diagnostic_trigger` |
| `ScenarioExecutionRequest` | `tenant_id`, `message`, `user_id`, optional `capability`, optional `task_id` |
| `ScenarioRuntimeExecutionResult` | `task_result`, `task_id`, `run_id`, `tenant_id` |
| `ScenarioRuntimeBuildError` | Fail-closed build when RuntimeEvent persistence is required but unavailable |

### Functions

| Function | Role |
|----------|------|
| `validate_scenario_tenant_id(tenant_id)` | Non-empty `str`, no leading/trailing whitespace |
| `build_scenario_runtime_from_environment(...)` | Compose environment → observability → reliability/security/guardrail/cost/evaluation/critic → `NexusLoop` → optional terminal diagnostic trigger |
| `execute_scenario_task(composition, request)` | Validate tenant, build `Task`, `mint_run_id()`, `nexus_loop.handle_task`, return platform envelope |

### LAB behavior

| Aspect | LAB profile |
|--------|-------------|
| Storage | Automatic scoped workspace (`runtime_events.db`, `trace.db`) via `build_scenario_lab_runtime` |
| Tenant | Explicit synthetic tenant constant passed by caller (not hidden global) |
| Nexus | Same `build_scenario_runtime_from_environment` baseline as production-attached |
| RuntimeEvent | ON — persistence required |
| Diagnostics | ON when shared `InMemoryDocumentStore` is wired (LAB default); unavailable without document store |
| API | `build_scenario_lab_runtime(registry=..., tenant_id=...)` — no manual DB paths |

Authors of ordinary generated proofs **do not** configure `runtime_events_db_path`, `trace_db_path`, `use_in_memory_trace`, `require_runtime_event_persistence`, or diagnostic storage rules manually.

Optional `workspace_root` may reuse a proof artifact directory when proof runner owns lifecycle.

### Production-attached behavior

| Aspect | PRODUCTION_ATTACHED profile |
|--------|----------------------------|
| Storage | Caller-supplied durable `runtime_events_db_path` / `trace_db_path` |
| Tenant | Real `tenant_id` required (validated, non-empty) |
| Manifest | Explicit `ApplicationManifest` required |
| Nexus | Same baseline |
| RuntimeEvent | Required — fail closed when missing |
| Diagnostics | Fail closed when `diagnostics_required=True` and `document_store` missing |
| API | `build_scenario_production_runtime(...)` — explicit critical configuration |

Forbidden without explicit caller decision: temp SQLite fallback, synthetic tenant, in-memory document store, `ApplicationManifest.lab` synthesis.

### Diagnostics prerequisites

| Prerequisite | Effect |
|--------------|--------|
| `observability.runtime_event_store` | Required when `require_runtime_event_persistence=True` |
| `env_wiring.build_context.tool_wiring_context.document_store` | Required for terminal diagnostic trigger |
| Both present | `nexus_loop.attach_terminal_diagnostic_trigger(...)` |

### Deliberately excluded

- `ApplicationHost`, `HarnessHostRuntime`, registry projection, control-plane governance, hosting lifecycle
- Direct imports of `DiagnosticOrchestrator`, `ProblemGroupingEngine`, `ProblemLifecycleEngine`, `ExecutionReconstructor`
- `platform_proofs`, domain scenario code, proof evaluators

### Tests

- `tests/unit/applications/test_scenario_runtime_baseline.py` — build, execution identity, tenant, RuntimeEvent persistence, diagnostics attachment, LAB without document store, architecture gate
- `tests/unit/applications/test_scenario_runtime_profiles.py` — LAB zero-config, execution, diagnostics default, storage isolation, production fail-closed, production success

---

*End of SCENARIO-PLATFORM contract (3A + 4 implemented).*
