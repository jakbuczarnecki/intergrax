# Observability

**Status:** Canonical architecture (domain pair 1:1)  
**Hub:** [`intergrax_runtime_architecture.md`](intergrax_runtime_architecture.md)
**Plan (1:1):** [`plan/OBSERVABILITY.md`](../maintainers/plans/OBSERVABILITY.md)
**Target:** [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../technical/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md)
**Audit layers:** 21, 30  
**Platform audit:** [`docs/audit_results/AUDIT_PROTOCOL.md`](../../audit_results/AUDIT_PROTOCOL.md)**Last updated:** 2026-08-17 — **TRACE-BITEMP-3** K-only knowledge reconstruction **Done / Closed** (`5c2eedca75fc32101ea7a35e332c2abb3af24985`) · **TRACE-ASOF-2** logical execution projection **Done / Closed** (`d0cfad1eeecbf3167e3955b93d4a2ef82de09b4f`) · **TRACE-BITEMP-2** canonical revision ordering provider **Planned / In Review** · **TRACE-BITEMP-1** typed bitemporal contracts + tenant ordering scope + transactional provider strategy **Done / Closed** (`d68c72177403fb634fd4ede2d0252e9814d7adee`) · **TRACE-ASOF-1** execution position + `AsOfBoundary` **Done / Closed** (`02462d96897daa4ea19d96dce776768a03cbbf53`) · **TRACE-BITEMP-ARCH-SYNC-R7** acceptance linearization + fenced-out/orphaned durable commit semantics · **TRACE-BITEMP-ARCH-SYNC-R6** unresolved position resolution ownership + lease/fencing + auditable terminalization · **TRACE-BITEMP-ARCH-SYNC-R5** watermark finality + gap semantics + idempotent acceptance requirements · **TRACE-BITEMP-ARCH-SYNC-R4** domain-owned revision ordering authority + pluggable provider contract · **TRACE-BITEMP-ARCH-SYNC-R3** revision watermark semantics + serialization decision boundary · **TRACE-BITEMP-ARCH-SYNC-R2** deterministic correction / knowledge revision ordering · **TRACE-BITEMP-ARCH-SYNC-R1** temporal axes semantic correction · pre-production clean-cut policy

---

## Cursor read scope (token budget)

**Do not read this entire file in one session** (OBSERVABILITY canon).

- **Implement / audit default:** trace spine + HOS + signal planes (§1–§4); execution identity + journal + as-of + bitemporal state (§5–§10). Extended depth: [`satellites/OBSERVABILITY_extended_depth.md`](satellites/OBSERVABILITY_extended_depth.md).
- **Use** table of contents below — `Read` with offset/limit per §.
- **Plan hub:** [`plan/OBSERVABILITY.md`](../maintainers/plans/OBSERVABILITY.md) (scoped §6 only).
- **Platform audit:** [`docs/audit_results/AUDIT_PROTOCOL.md`](../../audit_results/AUDIT_PROTOCOL.md).
- **Max reads:** at most **one** file >5k tokens per session unless RESUME cites more.

---


## Architecture satellites (read on demand)

Large § blocks moved out of the architecture hub to reduce Cursor context use.
Load **only** the satellite matching your task or cited §.

| Satellite | Contents |
|-----------|----------|
| [`satellites/OBSERVABILITY_extended_depth.md`](satellites/OBSERVABILITY_extended_depth.md) | extended depth · **OECP** target architecture (Evidence Ledger, Eval Registry v2, custom telemetry, L5–L7) |

> **Cursor context budget:** read hub read-scope block + **at most one** satellite per session.

## 1. Purpose and scope

### 1.1 What this document defines

This is the **single source of truth** for how observability works across the Intergrax Harness:

- **Harness (Tier-0 / Tier-1)** — Nexus, AgentEngine, ToolRuntime, policy, critic, adaptive loops
- **Applications (Tier-3)** — composition roots that wire stores and profiles; no parallel telemetry stack
- **Agents (Tier-2)** — domain logic that **extends** platform contracts; never implements a private trace pipeline

### 1.2 What observability must answer

For every user interaction (question → answer), an operator MUST be able to reconstruct:

| Question | Required evidence |
|----------|-------------------|
| What entered the system? | Intake events, ingestion, normalized task |
| How was the agent chosen? | Agent selection record (capability, score, fallback) |
| What plan was produced? | Plan events + planner diagnostics |
| What context was assembled? | Context/RAG/memory events (metadata; content redacted in prod) |
| What did each step do? | Step start/complete/fail, tool calls, LLM calls |
| What did policy/critic decide? | Policy decisions, validation layers, verdicts |
| What failed or retried? | Error taxonomy, retry schedule/start, handoff |
| What did it cost? | Token/cost aggregation per run |
| Why did the run stop? | Terminal event + reason codes |

### 1.3 Non-goals

- Replacing external APM (Datadog, Honeycomb) as the **only** store — Intergrax owns the canonical journal; external systems are **optional sinks**
- Storing raw prompts/completions in production traces (redaction is mandatory)
- Per-agent custom SQLite trace databases
- Raw `dict` payloads without `payload_schema_id` / registry (see §8.2 residual evolution)

---

## 2. Design principles

| Principle | Meaning |
|-----------|---------|
| **Harness-provided spine** | One observability mechanism ships with the platform. Applications and agents **configure and extend** it — they do not rebuild it. |
| **Event-first** | `RuntimeEvent` is the primary audit signal (canon §42.1). Traces and metrics are derived views. |
| **Typed extension** | Platform steps use `DiagnosticPayload` subclasses with stable `schema_id`. Domain extensions inherit the same contract. |
| **Emit at the boundary** | Signals are recorded where the Harness enforces policy (ToolRuntime, AgentRouter, GraphExecutor) — not inside ad-hoc agent helpers. |
| **Correlation by construction** | `TaskId`, `RunId`, `AttemptId`, `EventId`, `correlation_id`, `parent_event_id` are set by the spine — not passed manually in business code. |
| **Redact before persist** | `DiagnosticPayload.redact()` + `production_mode` run before any store append. |
| **Pluggable persistence** | SQLite default; Cassandra/Elasticsearch/OTLP as integration profiles — same API, different backend. |
| **Read-model unification** | Operators consume **one chronological journal** per run (`build_unified_run_journal`) — a derived read model, not the persistence source of truth (§6). |
| **Modular sinks** | Metrics, logs, and external trace UIs subscribe to the bus or journal — they do not fork emission. |

---

## 3. The Harness Observability Spine (HOS)

The **Harness Observability Spine** is the universal “bus” through which all tiers publish execution signals.

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                    HARNESS OBSERVABILITY SPINE (HOS)                     │
├─────────────────────────────────────────────────────────────────────────┤
│  EMIT (write)                                                            │
│    ObservabilityEmitter.emit_step()     ← single developer-facing API    │
│    RuntimeState.trace_event()           ← pipeline internal (today)      │
│    RuntimeEventBus.record() / publish() ← canonical envelope             │
├─────────────────────────────────────────────────────────────────────────┤
│  NORMALIZE                                                               │
│    trace_bridge                         TraceEvent → RuntimeEvent        │
│    payload_registry + schema_guard      schema_id → typed payload        │
├─────────────────────────────────────────────────────────────────────────┤
│  PERSIST (write path)                                                    │
│    RunTraceWriter                       TraceEvent timeline (SQLite…)    │
│    RuntimeEventPersistence              RuntimeEvent journal (SQLite…)     │
├─────────────────────────────────────────────────────────────────────────┤
│  READ (query path)                                                       │
│    build_unified_run_journal()          merged chronological timeline      │
│    export_run_metrics()                 aggregates per run                 │
│    Debug API / CLI                      operator inspection                │
├─────────────────────────────────────────────────────────────────────────┤
│  SINKS (optional, subscribe/export)                                      │
│    journal_export plugin                unified journal OTLP snapshot      │
│    OTLP / Prometheus                    LLM/RAG metrics plugins            │
│    ObservabilityBackend tools           Langfuse, Sentry, Phoenix…         │
│    Custom RuntimeEventBus handlers      alerting, webhooks               │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key rule:** Harness, applications, and agents all use the **same spine**. Differences are only in **which steps emit** and **which `DiagnosticPayload` schemas** are registered — not in transport or storage mechanics.

---

## Observability Event Spine

**Normative rule:** `RuntimeEvent` is the canonical runtime event and audit envelope for meaningful execution transitions.

The Harness Observability Spine (§3) is the write/read/export path; this section defines **what each signal type owns** so agents, tools, integrations, and applications do not fork parallel observability pipelines.

| Signal | Role | Must not become |
|--------|------|-----------------|
| **`RuntimeEvent`** | Canonical event/audit envelope on `RuntimeEventBus`; primary source of execution truth for lifecycle, policy, HITL, and operator reconstruction | A optional add-on beside private agent logs |
| **`TraceEvent`** | Compatibility / read-model / diagnostic view (Plane B); fine-grained timeline via `RuntimeState.trace_event()` and `RunTraceWriter`; bridged to the bus by `trace_bridge` | A competing event bus or private audit store |
| **Logs** | Local diagnostic output (stdlib logging, host logs, integration transport traces) | Canonical audit evidence or execution history |
| **Metrics** | Aggregated operational signals (Prometheus, OTLP counters, SLO ratios) derived from events or counters | A substitute for the unified run journal |
| **External sinks** | Destinations for normalized events, logs, or metrics (Langfuse, Sentry, Datadog, OTLP export) | Semantic owners of Intergrax event vocabulary |
| **`DiagnosticPayload`** | Typed payload detail carried by Plane B trace rows or domain-signal envelopes (`payload_schema_id` + `redact()`) | An independent lifecycle channel with its own persistence contract |
| **`PlatformProblemSignal`** | Vendor-neutral problem/error plane for classified failures requiring operator attention; exported via `ObservabilityExportPolicy` | A substitute for `RuntimeEvent` execution history or a generic lifecycle channel |
| **Platform observability signal** | Non-execution platform/domain lifecycle signal on HOS (application instance, component, infrastructure) with its own identity and correlation — **no** `TaskId`/`RunId`/`AttemptId` | A `RuntimeEvent` with synthetic execution identity |

**Implementation detail:** Plane A/B/C breakdown, field catalog, and bridge mechanics — §4. Correlation identifiers — §6 and [Required correlation fields](.#required-correlation-fields) below. Layered `event_type` / `event_kind` governance — §4.4 and [Event type governance](.#event-type-governance) below.

**Cross-layer canon:** [`SYSTEM_INVARIANTS.md`](../technical/guides/SYSTEM_INVARIANTS.md) §7 · [`UNIFIED_EXECUTION_RUNTIME.md`](UNIFIED_EXECUTION_RUNTIME.md) §42.1 · [`NEXUS_EXECUTION_FLOW.md`](NEXUS_EXECUTION_FLOW.md) §12.2 · [`RELIABILITY_FAILURE_AND_HITL.md`](RELIABILITY_FAILURE_AND_HITL.md#attempt-ledger) · [`TOOLS.md`](TOOLS.md) · [`INTEGRATIONS.md`](INTEGRATIONS.md) · [`AGENT_CONTRACTS_AND_ASSEMBLY.md`](AGENT_CONTRACTS_AND_ASSEMBLY.md) §31 · [`CRITIC_VERIFICATION.md`](CRITIC_VERIFICATION.md#boundary-with-observability--evaluation-control-plane-oecp) · [`ADAPTIVE_HARNESS_INTELLIGENCE.md`](ADAPTIVE_HARNESS_INTELLIGENCE.md#governance-boundary) · [`ELASTIC_CAPACITY_AND_SCALING.md`](ELASTIC_CAPACITY_AND_SCALING.md#scaling-action-governance) · [`CODE_CRAFT.md`](CODE_CRAFT.md#codecraft-safety-boundary)

---

## Observability & Evaluation Control Plane

Intergrax observability is **not** limited to traces and metrics. The **Harness Observability Spine (HOS)** remains the **only** canonical observability spine. **Observability & Evaluation Control Plane (OECP)** operates **above** HOS — it consumes `RuntimeEvent`, `TraceEvent`, unified journal, and evidence refs; it **must not** create a parallel trace system.

OECP transforms spine data into eval-grade artifacts: **evidence ledger** records, **eval snapshots**, **metric results**, **regression gates**, and **perturbation suites**. External workbenches (Langfuse, LangSmith, OTLP, Sentry, Phoenix, Braintrust, Datadog, …) are optional sinks — not semantic owners.

**Target architecture:** [`satellites/OBSERVABILITY_extended_depth.md`](satellites/OBSERVABILITY_extended_depth.md) (OECP sections). **Plan:** [`plan/satellites/OBSERVABILITY_eval_control_plane.md`](../maintainers/plans/satellites/OBSERVABILITY_eval_control_plane.md). **Audit source:** [`audit/OBSERVABILITY_EVALUATION_CONTROL_PLANE_AUDIT.md`](../../audit_results/OBSERVABILITY_EVALUATION_CONTROL_PLANE_AUDIT.md).

---

## Event ownership rules

| Rule | Requirement |
|------|-------------|
| Runtime emission | New runtime components **SHOULD** emit meaningful execution transitions through `RuntimeEventBus` or the approved observability spine (§3). |
| Agent trace stores | Agents **MUST NOT** create private trace stores. |
| Agent logging pipelines | Agents **MUST NOT** create private logging pipelines for execution state. |
| Tool side effects | Tools **MUST NOT** bypass runtime observability for side effects — `TOOL_*` and bridged diagnostics **MUST** be visible through the spine ([`TOOLS.md`](TOOLS.md)). |
| Integration diagnostics | Integrations **MAY** log transport/backend diagnostics; they **MUST NOT** own harness execution trace semantics ([`INTEGRATIONS.md`](INTEGRATIONS.md)). |
| Application summaries | Applications **MAY** add product-level summaries (e.g. `ApplicationRunSummary`); they **MUST NOT** replace runtime event history. |
| External sinks | External sinks **MUST** receive normalized signals; they **MUST NOT** define canonical Intergrax event semantics. |
| Secrets | Event payloads **MUST NOT** contain secrets. |
| Redaction | Redaction **MUST** happen before persistence or external export where required (`DiagnosticPayload.redact()`, `production_mode`). |
| Domain extension | Domain-specific events **SHOULD** use namespaced `event_kind` / payload schemas instead of expanding platform lifecycle enums unnecessarily (§4.4). |

Audit stores (`RuntimeEventPersistence`, `RunTraceWriter`) persist spine-normalized records — they are **not** alternate semantic owners. Custom `RuntimeEventBus` handlers and journal export plugins are subscribers/sinks, not parallel buses.

---

## Execution-scoped vs non-execution observability signals

**Normative rule:** `RuntimeEvent` is **execution-scoped only**. Every canonical `RuntimeEvent` **MUST** carry full execution identity: `TaskId`, `RunId`, `AttemptId`, and `EventId` — all required, none optional, no synthetic placeholders to admit unrelated signals.

The Harness Observability Spine (HOS) is **broader** than `RuntimeEvent`. HOS is the single approved write/read/export path; it carries **multiple semantic envelope families** through the same spine infrastructure. Semantic contract and transport/storage/export mechanism are separate concerns — **one spine, not two buses**.

```text
Harness Observability Spine (HOS)
├── Execution event          → RuntimeEvent (TaskId + RunId + AttemptId + EventId)
├── Platform observability   → non-execution platform/domain lifecycle signal
├── Problem plane            → PlatformProblemSignal (failures / operator attention)
├── Diagnostic detail        → DiagnosticPayload (payload on trace or domain-signal envelopes)
├── Read model               → TraceEvent (Plane B compatibility / reconstruction)
└── Export projection        → ObservabilityExportEnvelope (policy-safe export record)
```

### A. Execution-scoped signals (`RuntimeEvent`)

| Property | Requirement |
|----------|-------------|
| Scope | Meaningful **execution** transitions inside a Task → Run → Attempt lifecycle |
| Identity | `event_id`, `task_id`, `run_id`, `attempt_id` — all required |
| `AttemptId` semantics | One concrete execution try inside a Run; **one arbitrary observable event ≠ one Attempt** |
| Source of execution truth | `RuntimeEvent` persistence is canonical execution history; Unified Run Journal reconstructs execution from `RuntimeEvent` only |
| Forbidden | Optional execution identity; multiplexed identity modes; synthetic `TaskId`/`RunId`/`AttemptId` for non-execution events |

`emit_domain_signal()` and `RuntimeEventType.DOMAIN_SIGNAL` are **execution-attached** in practice: both require `EmitContext` with validated `TaskId`, `RunId`, and `AttemptId`. A domain signal on the bus is a `RuntimeEvent` carrying a namespaced `event_kind` and typed payload **within an active execution correlation** — not a generic non-execution lifecycle channel. Platform lifecycle facts that occur **during** execution (for example `platform.adaptive.*` on `DOMAIN_SIGNAL`) remain execution-scoped because they are correlated to a real attempt.

### B. Non-execution platform observability signals

**Platform observability signal** is the canonical semantic family for observable platform/domain lifecycle facts that do **not** belong to Task/Run/Attempt execution history.

| Property | Requirement |
|----------|-------------|
| Scope | Application hosting lifecycle, component health, instance acquisition/release, infrastructure lifecycle, and similar platform facts **outside** execution attempt boundaries |
| Identity | Signal-local `event_id` (or equivalent), `correlation_id`, `causation_id`, source/component identity (`application_id`, `instance_id`, … as applicable), typed payload, severity/category |
| Execution identity | **MUST NOT** include `TaskId`, `RunId`, or `AttemptId`; **MUST NOT** mint `AttemptId` per signal |
| Source of truth | Describes platform/application observability — **not** execution history; does not replace Unified Run Journal reconstruction |
| Transport | Published through the **existing HOS spine/export path** — not a second bus, not `RuntimeEventBus.record()` with fake execution identity |

`ObservabilityExportEnvelope` is an **export projection / transport envelope** only (`record_kind`, sanitized fields). It is **not** the semantic owner of platform lifecycle facts — do not promote it to domain semantics.

`DiagnosticPayload` is **payload detail** (`schema_id`, `redact()`) carried by Plane B `TraceEvent` rows or execution-attached `DOMAIN_SIGNAL` envelopes. It is **not** an independent non-execution lifecycle channel.

`TraceEvent` remains a **compatibility / read-model / diagnostic view** (Plane B). It **MUST NOT** become the canonical non-execution signal bus or execution truth source.

`PlatformProblemSignal` remains the specialized **problem/error plane** (`what broke / requires attention`). It **MUST NOT** be abused for routine hosting lifecycle events.

### C. Application hosting classification

`HostedApplicationEvent` (`intergrax/hosting/contracts/events.py`) is the typed authoring envelope for **application-hosting platform observability signals**. Its semantics are:

| Lifecycle | Examples |
|-----------|----------|
| Application instance | `APPLICATION_STARTING`, `APPLICATION_READY`, `APPLICATION_STOPPED`, `APPLICATION_FAILED` |
| Component | `COMPONENT_STARTED`, `COMPONENT_HEALTH_CHANGED`, `COMPONENT_FAILED` |
| Instance guard | `INSTANCE_ACQUIRED`, `INSTANCE_RELEASED`, `INSTANCE_STALE_RECOVERED` |
| Restart / hooks / plugins | `RESTART_*`, `HOOK_*`, `PLUGIN_*` |

These events describe **hosted application/platform lifecycle** — not Intergrax Task, Run, or Attempt lifecycle. `HostedApplicationEvent` already carries the correct non-execution identity (`event_id`, `correlation_id`, `causation_id`, `application_id`, `instance_id`).

**Target (canonical):**

```text
HostedApplicationEvent
  → platform observability signal (hosting domain)
  → existing HOS spine / export infrastructure
```

**Architecture debt (implementation status: Planned — TRACE-1B-HOS-FIX):** `RuntimeSpineHostedApplicationEventPublisher` (`intergrax/hosting/eventing.py`) currently adapts hosting events into `RuntimeEvent` via `emit_domain_signal()` after synthesizing `TaskId`, `RunId`, and a fresh `AttemptId` (`mint_attempt_id()`) per event. This is **architecturally invalid** under TRACE-1B: it falsely equates each hosting event with a new execution attempt. Pre-production clean-cut: **remove/replace** this adapter in the implementation slice — no compatibility alias, dual path, or historical migration.

### D. Author decision supplement (see also §4.4.1)

```text
Need a new signal?
├── No Task/Run/Attempt lifecycle (hosting, infra, app instance)?
│     → platform observability signal on HOS (not RuntimeEvent)
├── Debug / reconstruction only?     → DiagnosticPayload (Plane B)
├── Product/domain fact during execution? → emit_domain_signal (requires real EmitContext)
├── Nexus lifecycle transition?      → emit_platform_event (requires real EmitContext)
└── Classified failure / operator attention? → PlatformProblemSignal (problem plane)
```

---

## Problem signal emission boundary

**Normative rule:** `RuntimeEvent` answers **what happened**; `PlatformProblemSignal` answers **what broke and requires attention**. Problem signals are an explicit semantic classification at an **owned emission boundary** — not an automatic conversion from every `RuntimeEvent` or exception.

`ProblemReporter` / `report_problem` (`intergrax/runtime/observability/problem_reporter.py`) is the developer-facing helper for building and exporting problems through the existing observability export path (`PlatformProblemSignal` → `ObservabilityExportEnvelope` → `ObservabilityExportPolicy` → `try_export_observability_envelope`). This section defines **where** that helper may be called. It does **not** add automatic runtime emission, routing/fanout, Sentry, Elastic, OTLP, or vendor-specific behavior.

### A. ProblemSignal role

| Property | Requirement |
|----------|-------------|
| Semantic model | `PlatformProblemSignal` is the vendor-neutral problem/error signal model (`problem_signal.py`). |
| Not a replacement for `RuntimeEvent` | Execution/audit history remains on the spine; problems are a separate explicit plane. |
| Not a generic log record | Problems require classified taxonomy fields — not unstructured diagnostic text. |
| Not vendor-specific | No Sentry/Elastic/OTLP semantics in the platform model; vendors project sanitized envelopes only. |
| Attention signal | Represents a classified failure/problem requiring operator or developer attention. |

### B. Allowed emitters

Problem signals **MAY** be emitted only from boundaries that **own failure classification** for a run/task and can preserve correlation identifiers:

| Boundary | Examples |
|----------|----------|
| **Application** | Tier-3 endpoint handler, command handler, pipeline boundary, or composition root that owns product-level failure classification. |
| **Runtime** | Runtime executor, graph boundary, agent run boundary, tool runtime wrapper, or policy-enforced runtime boundary. |
| **Integration** | Platform integration wrapper that classifies a provider/backend failure into a platform problem without leaking vendor SDK details. |
| **Explicit tool wrapper** | `ToolRuntime` or an approved wrapper around tool execution — **not** arbitrary tool internals. |

The owning boundary **SHOULD** call `report_problem(...)` or `ProblemReporter(...).report(...)` once it has decided the failure is reportable and has stable `problem_kind`, `severity`, `source_layer`, `source_component`, and `error_code` when available.

### C. Discouraged or forbidden emitters

| Location | Rule |
|----------|------|
| Low-level utility functions | **MUST NOT** own platform problem taxonomy. |
| Raw model/provider client code | **MUST NOT** emit `PlatformProblemSignal`; raise typed errors or return structured failures instead. |
| Ad-hoc agent helper functions | **MUST NOT** report problems outside an agent run boundary that owns classification. |
| LKW-only private logging code | **MUST NOT** define an LKW-only issue model or bypass the platform helper. |
| Vendor provider internals | **MUST NOT** be semantic owners of platform problems or platform taxonomy. |
| External sink/provider code | **MUST NOT** define Intergrax problem kinds or severities. |
| Code without run/task/correlation ownership | **MUST NOT** call `report_problem` — use `ProblemReportContext` with available correlation fields at the owning boundary. |

### D. Duplicate prevention

| Rule | Detail |
|------|--------|
| One owner per failure | One failure **SHOULD** have one owning emission boundary. |
| Lower layers raise, upper layers classify | Lower layers **MAY** raise typed errors or attach typed context; they **SHOULD NOT** double-report if a higher boundary owns classification. |
| Correlation preservation | The boundary that reports **MUST** populate `run_id`, `task_id`, `correlation_id`, and related fields when available (`ProblemReportContext`). |
| Export failure isolation | `try_export_observability_envelope` failure isolation **MUST NOT** recursively create an unbounded chain of problem signals — export failures are isolated; optional single observability-plane report is a separate explicit decision at an observability boundary. |

### E. RuntimeEvent relationship

| Rule | Detail |
|------|--------|
| No automatic conversion | Not every `RuntimeEvent` becomes a `PlatformProblemSignal`. |
| No automatic exception mapping | Not every exception automatically becomes a problem signal. |
| Retries/fallbacks | Not every retry or fallback is a problem — only semantically classified failures. |
| Required classification | A problem signal **MUST** include explicit taxonomy: `problem_kind`, `severity`, `source_layer`, `source_component`, and `error_code` when available. |
| Spine remains canonical | `RuntimeEvent` remains the canonical execution/audit history on `RuntimeEventBus`. |
| Export plane | `PlatformProblemSignal` is the explicit problem/error plane exported via `ObservabilityExportPolicy` and existing envelope mapping (`problem_export.py`). |

A boundary **MAY** correlate a problem to a spine `event_id` when both exist; correlation does **not** imply automatic creation from the event.

### F. Safety rules

Problem signals and their export envelopes **MUST** follow the same content-safety posture as observability export:

| Forbidden | Required alternative |
|-----------|---------------------|
| Raw exception serialization (stack traces, `str(exc)` bodies) | `error_code`, `exception_type` (class name only) when applicable |
| Raw prompt/query/content/chunks/tool_args | Typed `ApplicationObservabilityAttributes` with declared safe fields only |
| Raw local file paths | `ObservabilityArtifactReference` (`artifact_ref`, `sha256`, `safe_relative_path`, `schema_id`) |
| Raw `dict` payload/context/details/metadata | Typed attributes and reference-only artifacts |
| Secrets | Never — policy drops or hashes forbidden fields |

`ObservabilityExportPolicy` owns redaction/sanitization before export. Vendor providers receive **only** policy-safe envelopes.

### G. Developer-facing examples

**Application boundary — explicit classification:**

```python
from intergrax.runtime.observability.problem_reporter import ProblemReportContext, report_problem
from intergrax.runtime.observability.problem_signal import PROBLEM_SOURCE_LAYER_APPLICATION

context = ProblemReportContext(
    run_id="run-fake-001",
    task_id="task-fake-001",
    correlation_id="corr-fake-001",
    agent_id="agent-lkw",
    capability="local.workspace.search",
)

await report_problem(
    context=context,
    problem_kind="lkw.retrieve_failed",
    severity="error",
    error_code="LKW_RETRIEVE_FAILED",
    source_layer=PROBLEM_SOURCE_LAYER_APPLICATION,
    source_component="local_workspace_search_handler",
    tool_id="rag.retrieve",
)
```

**Runtime/tool boundary — bound reporter:**

```python
from intergrax.runtime.observability.problem_reporter import ProblemReportContext, ProblemReporter
from intergrax.runtime.observability.problem_signal import PROBLEM_SOURCE_LAYER_TOOL

reporter = ProblemReporter(
    context=ProblemReportContext(
        run_id="run-fake-002",
        task_id="task-fake-002",
        correlation_id="corr-fake-002",
    ),
)

await reporter.report(
    problem_kind="platform.tool_failure",
    severity="error",
    error_code="TOOL_EXECUTION_FAILED",
    source_layer=PROBLEM_SOURCE_LAYER_TOOL,
    source_component="tool_runtime_wrapper",
    tool_id="web.search",
)
```

### H. Anti-examples (do not)

- Do **not** call `sentry_sdk` (or any vendor SDK) from runtime, application, agent, or tool code.
- Do **not** map LKW or domain code directly to Sentry, Elastic, or OTLP — use the platform export envelope and policy.
- Do **not** emit the same failure from tool internals, agent helper, **and** endpoint (pick one owning boundary).
- Do **not** serialize raw exception objects, raw context dicts, or query/content into problem fields.
- Do **not** turn every `RuntimeEvent` (or every `ObservabilityEmitter.emit_step`) into a `PlatformProblemSignal`.
- Do **not** add `ObservabilityEmitter.emit_problem`, automatic global exception hooks, or `RuntimeEventBus` subscribers that auto-emit problems (deferred / out of scope for OBS-PROBLEM-3).

**Code references:** `problem_signal.py` · `problem_export.py` · `problem_reporter.py` · `export_boundary.py` · `export_policy.py`. **Plan:** OBS-PROBLEM-3 in [`plan/OBSERVABILITY.md`](../maintainers/plans/OBSERVABILITY.md).

---

## Problem signal routing/fanout boundary

**Normative rule:** routing operates only on **policy-safe** `ObservabilityExportEnvelope` records — typically after `ObservabilityExportPolicy` and `try_export_observability_envelope`. Routing selects logical destinations; it does **not** decide problem semantics, sanitize raw data, or call vendor SDKs.

### A. Routing role

| Property | Requirement |
|----------|-------------|
| Input plane | Policy-safe envelopes only — not raw `PlatformProblemSignal`, exceptions, or unsanitized attributes. |
| No semantic classification | Routing **MUST NOT** decide `problem_kind`, severity, or error taxonomy. |
| No sanitization | Routing **MUST NOT** apply redaction or replace `ObservabilityExportPolicy`. |
| No vendor SDKs | Routing **MUST NOT** import or call Sentry, Elastic, OTLP, or other vendor clients. |
| Post-policy selection | Operator/platform wiring selects destinations **after** policy has allowed export. |

### B. Ownership split

| Owner | Responsibility |
|-------|----------------|
| **Producer** | Semantic signal — `problem_kind`, severity, source context, correlation. |
| **Policy** | Safety/redaction — `ObservabilityExportPolicy`, sanitized attributes, forbidden-field drops. |
| **Operator routing** | Destination selection — which logical routes receive a policy-safe envelope. |
| **Vendor provider** | Delivery format/projection — Sentry issue, Elastic document, OTLP span, etc. (future tasks). |

### C. Routing criteria

Allowed route filters (empty filter tuple = match all):

| Criterion | Source |
|-----------|--------|
| `record_kind` | Envelope `record_kind` (e.g. `problem_signal`). |
| `problem_kind` | Envelope `problem_kind`. |
| `problem_severity` | Envelope `problem_severity`. |
| `problem_error_code` | Envelope `problem_error_code`. |
| Source fields | Envelope fields already present (`run_id`, `agent_id`, `capability`, `tool_id`, …). |
| `source_layer` / `source_component` | Only when present on envelope or a future envelope extension. |
| `tenant_id` / `workspace_id` | Only after policy allows them. |
| Operator config flags | Later tasks — not routing module construction. |

### D. Fanout behavior

| Rule | Detail |
|------|--------|
| One input | One policy-safe envelope fans out to zero/one/many selected routes. |
| Disabled routes | `enabled=False` routes are skipped. |
| Filter skip | Non-matching filters skip a route without error. |
| Per-route isolation | Exporter failure on one route **MUST NOT** block other routes. |
| No propagation | Fanout exporter failures **MUST NOT** raise to callers by default. |
| No recursive problems | Fanout **MUST NOT** recursively emit new `problem_signal` records by default. |

Platform contract: `FanoutObservabilityExporter` + `ObservabilityExportRoute` (`export_routing.py`). Operators that need per-route diagnostics may call `FanoutObservabilityExporter.export_with_result(...)`; `export(...)` remains the `ObservabilityExporter`-compatible method.

### E. Vendor boundary

Sentry is a **provider-owned projection** for `ObservabilityVendorPayload` with `ObservabilityVendorSignal.PROBLEMS`: the Sentry provider maps policy-safe problem metadata to Sentry issue-shaped events. **Sentry SDK is used only inside Sentry provider transport/client/factory code** (`intergrax/integrations/providers/observability_backend/sentry`). Runtime, LKW, agents, and tools **MUST NOT** import or call `sentry_sdk`.

Elastic, OTLP, Langfuse, and similar backends are operator-selectable projections that receive policy-safe envelopes from configured route exporters. Runtime, application, agent, tool, and LKW code **MUST NOT** choose vendor destinations directly.

Problem/error/issue information flows through the shared `ObservabilityVendorIntegrationContract`: vendors receive policy-safe `ObservabilityVendorPayload` with platform problem metadata (`problem_kind`, `problem_severity`, `problem_error_code`). Sentry projects problems to Sentry issues; Elasticsearch projects problems to indexed error/problem documents. The producer does not choose the backend — operator routing decides whether `problem_signal` goes to Sentry, Elasticsearch, both, or another backend; the vendor provider decides delivery projection.

**Deferred:** LKW endpoint proof, docker compose, live Sentry proof, and operator bootstrap wiring are separate follow-on tasks.

### F. Out of scope (this boundary)

- No runtime automatic problem emission.
- No `ObservabilityEmitter.emit_problem`.
- No `RuntimeEventBus` subscriber for problems.
- No LKW endpoint wiring or operator bootstrap config.

**Code references:** `export_routing.py` · `export_boundary.py` · `export_policy.py`. **Plan:** OBS-ROUTING-0 in [`plan/OBSERVABILITY.md`](../maintainers/plans/OBSERVABILITY.md).

---

## Required correlation fields

Meaningful runtime events **SHOULD** preserve all correlation identifiers available at the emission boundary:

| Field | Purpose |
|-------|---------|
| `task_id` (`TaskId`) | User-facing work unit / intent — **WHAT** task |
| `run_id` (`RunId`) | Single execution of the task — **WHICH** run |
| `attempt_id` (`AttemptId`) | Attempt within the run — **WHICH** attempt (target canon §5) |
| `node_id` | Graph node placement |
| `agent_id` | Responsible agent |
| `step_id` | UAEP / pipeline step |
| `tool_call_id` | Tool invocation chain (when applicable) |
| `correlation_id` | Cross-agent/tool chain (default: `task_id`) |
| `parent_event_id` | Causal parent in the spine tree |
| `event_id` (`EventId`) | Unique runtime event identity — **WHICH** event |
| `timestamp` | UTC ordering |
| `schema_version` | Envelope version (e.g. `runtime_event.v1`) |

`EmitContext` and `TraceScope` (§6.2) populate these by construction on approved emit paths. A component that **drops** correlation identifiers **MUST** document **why** and **what observability is lost** (e.g. in PR description or module docstring).

---

## Event type governance

| Rule | Detail |
|------|--------|
| Spine rarity | New high-level lifecycle `RuntimeEventType` members **SHOULD** be rare — publication budget ~56 (§4.4.13). |
| Domain detail | Prefer namespaced `event_kind` or typed payload schemas (`emit_domain_signal`) for domain-specific detail. |
| Platform changes | Adding a new platform-level spine `event_type` requires updating this document (§4.4), `EventCatalogEntry`, relevant ADR, and observability checks if they exist (`check_event_catalog.py`, `check_observability_gates.py`). |
| No product-only categories | Do **not** create new event categories only to support one product-specific use case. |
| Product payloads | Product-specific events **SHOULD** remain in product/domain payloads (`agents.*`, `applications.*` kinds) unless they represent a general harness lifecycle concept. |

See also §4.4.6 anti-patterns and [`UNIFIED_EXECUTION_RUNTIME.md`](UNIFIED_EXECUTION_RUNTIME.md) §42.1.7.

---

## Cursor review checklist

Before adding or modifying observability behavior, Cursor **MUST** verify:

- [ ] Is this a meaningful execution transition?
- [ ] Should it be represented as `RuntimeEvent` (direct bus emit or trace bridge)?
- [ ] Are correlation identifiers preserved (`EmitContext` / `TraceScope`)?
- [ ] Are secrets redacted before persistence/export?
- [ ] Is this a platform lifecycle `event_type` or a domain-specific `event_kind`/payload?
- [ ] Does this create a parallel private trace/log system?
- [ ] Are tool side effects visible through the runtime spine?
- [ ] Are integration/backend logs clearly separated from harness execution events?
- [ ] Are metrics derived from events or operational counters rather than replacing event history?
- [ ] Are external sinks treated as destinations, not semantic owners?

---

## 4. Three signal planes

Intergrax observability deliberately separates three planes (pattern: event sourcing + structured logging + metrics).

### 4.1 Plane A — Canonical events (`RuntimeEvent`)

| Field | Role |
|-------|------|
| `event_type` | `RuntimeEventType` enum — **spine** lifecycle vocabulary (§4.4) |
| `event_kind` | Namespaced semantic id — **primary for domain extensions** (defaults to `event_type.value`) |
| `event_category` | Derived ops grouping (`tool`, `agent`, `plan`, …) — §4.4.2 |
| `phase` | `ExecutionPhase` — where in the Nexus lifecycle |
| `severity` | `EventSeverity` — alert routing |
| `task_id` | Logical work unit (user request scope) — **target:** `TaskId` |
| `run_id` | Single execution of the task — **target:** `RunId`; retries mint new `AttemptId` under the same `RunId` (§5.4) |
| `attempt_id` | Attempt within the run — **target:** `AttemptId` on every canonical `RuntimeEvent` (§5) |
| `correlation_id` | Cross-agent/tool chain (default: `task_id`) |
| `parent_event_id` | Causal parent in the spine tree (**target:** populated by `TraceScope`) |
| `node_id` / `agent_id` / `step_id` | Graph and UAEP placement |
| `payload` | Structured facts (**today:** `dict`; **target:** typed `RuntimeEventPayload`) |
| `schema_version` | Envelope version (`runtime_event.v1`) |

**Code:** `intergrax/runtime/events/runtime_event.py`, `phase_coverage.py`, `event_bus.py`

**Catalog:** **56** `RuntimeEventType` spine members (publication budget; OBS-EVOL-9.7). Platform adaptive/capacity/hook/recovery signals emit on `DOMAIN_SIGNAL` + `platform.*` `event_kind` — see §4.4.13.

### 4.4 Layered event identity (P1-ARCH-02 · OBS-EVOL-9)

**Status:** Architecture **accepted** (2026-06-17) · implementation **Done** (OBS-EVOL-9 register) · **ADR:** [`ADR-OBS-003`](../technical/adr/entries/2026-06-17/ADR-OBS-003.md) · **SAR:** accepted 2026-06-17 (§4.4.7–4.4.13)

HOS uses **three levels of identity** so the spine scales without forcing developers through platform enum changes:

```text
RuntimeEvent
├── event_type      RuntimeEventType   # spine — platform lifecycle (~50 at publication)
├── event_kind      str                # semantic — namespaced domain id (unbounded)
├── event_category  EventCategory      # derived — ops/metrics/hook grouping
├── phase           ExecutionPhase     # when in Nexus lifecycle
├── ops_hint        str                # trace/alert routing token
└── payload         envelope           # payload_schema_id + data (registry-backed)
```

| Level | Owner | Examples | Growth |
|-------|-------|----------|--------|
| **Spine** `event_type` | Platform (Tier-0/1) | `TASK_CREATED`, `TOOL_COMPLETED`, `HUMAN_APPROVAL_REQUESTED`, `DOMAIN_SIGNAL` | Frozen ~50; ADR to add |
| **Kind** `event_kind` | Platform + agents + apps | `agents.legal.clause_flagged`, `platform.adaptive.signal_recorded` | Unbounded; registry |
| **Trace** Plane B | Agents (preferred for debug) | `agents.legal.diag.clause_parse` | Unbounded; extension SDK |

**Default rule:** `event_kind` defaults to `event_type.value` for spine events.

#### 4.4.1 Author decision tree

```text
Need a new signal?
├── Debug / reconstruction only?     → DiagnosticPayload (Plane B)
├── Product/domain fact on bus?      → emit_domain_signal(kind, payload)
│                                      event_type = DOMAIN_SIGNAL
├── Nexus lifecycle transition?      → emit_platform_event(event_type, payload)
│                                      (platform PR + EventCatalog entry + ADR if new spine)
└── Must trigger platform HITL?      → Tier-3 adapter maps kind → existing spine
                                       (e.g. kind → HUMAN_APPROVAL_REQUESTED)
```

#### 4.4.2 `EventCategory` (derived, not a second enum root)

Categories group kinds for subscribers and metrics — **not** a replacement for `event_type`:

| Category | Spine examples | Kind prefix examples |
|----------|----------------|----------------------|
| `task` | `TASK_*` | `platform.task.*` |
| `plan` | `PLAN_*` | `platform.plan.*` |
| `tool` | `TOOL_*` | `agents.*.tool_*` |
| `agent` | `AGENT_SELECTED`, `STEP_*` | `agents.<slug>.*` |
| `context` | `CONTEXT_*`, `MEMORY_*` | `platform.context.*` |
| `human` | `HUMAN_*`, `PAUSE_*` | — |
| `policy` | `POLICY_DECISION`, `GUARDRAIL_BLOCKED` | `platform.policy.*` |
| `platform` | `DOMAIN_SIGNAL` carrier | `platform.adaptive.*`, `platform.capacity.*` |

Ops subscribes to `ops_hint` and `event_category`; developers subscribe to `kind_prefix`.

#### 4.4.3 Target spine at publication (pre-release consolidation)

Before external v1, consolidate **non-lifecycle** enum members into `DOMAIN_SIGNAL` + `platform.*` kinds:

| Keep on spine | Consolidate to `DOMAIN_SIGNAL` + kind |
|---------------|---------------------------------------|
| `TASK_*`, `PLAN_*`, `STEP_*` | — |
| `TOOL_*`, `VALIDATION_*`, `DECISION_EMITTED` | — |
| `HUMAN_*`, `INTERRUPT_*`, `PAUSE_*`, `RETRY_*` | — |
| `CONTEXT_*`, `MEMORY_*`, `SKILL_*`, `INGESTION_FAILED` | — |
| `HANDOFF_*`, `DELEGATION_GRANTED`, `GRAPH_BACKPRESSURE` | — |
| `POLICY_DECISION`, `GUARDRAIL_BLOCKED`, `BUDGET_*` | — |
| `TASK_PROGRESS`, `LLM_CALL`, `TRACE_PERSISTED` | — |
| `RUNTIME_HANDLER_FAILED`, `CANCELLED`, `CANCELLATION_REQUESTED` | — |
| — | `ADAPTIVE_*` → `platform.adaptive.*` |
| — | `SCALE_*`, `CAPACITY_*`, `AUTONOMY_*` → `platform.capacity.*` |
| — | `HOOK_*` → `platform.hook.*` |
| — | `RECOVERY_REBOOT` → `platform.recovery.reboot` |

**Code target:** `intergrax/runtime/events/event_catalog.py` (single registry); `phase_coverage.py` becomes a view until removed.

#### 4.4.4 Public emit APIs (target)

```python
# Tier-2/3 — primary extension path
emit_domain_signal(ctx, kind="agents.legal.clause_flagged", payload=LegalClauseFlaggedPayloadV1(...))

# Platform only — lifecycle spine
emit_platform_event(ctx, event_type=RuntimeEventType.TOOL_COMPLETED, payload=ToolPayloadV1(...))
```

Tier-2 agents **must not** import `RuntimeEventType` for product semantics.

#### 4.4.5 Bus subscription (additive)

```python
bus.subscribe(handler, event_types={RuntimeEventType.TOOL_COMPLETED})  # legacy
bus.subscribe(handler, categories={EventCategory.TOOL})                # preferred
bus.subscribe(handler, kind_prefix="agents.legal.")                     # product hooks
```

#### 4.4.6 Anti-patterns

| ID | Anti-pattern | Correct |
|----|--------------|---------|
| EVT-AP-01 | Tier-2 adds `RuntimeEventType` member | `emit_domain_signal` + extension payload |
| EVT-AP-02 | Raw dict on bus without `payload_schema_id` | `RuntimeEventPayload.to_envelope()` |
| EVT-AP-03 | Per-agent trace SQLite | Plane B via `AgentEngine` |
| EVT-AP-04 | Duplicate semantics in enum and kind | Kind is authoritative for domain; spine for lifecycle |
| EVT-AP-05 | High-cardinality `event_kind` in Prometheus labels | Aggregate by `event_category`; kind in journal only |
| EVT-AP-06 | Reuse `event_kind` name for LLM stream chunks and HOS bus signals | Stream: `intergrax.llm.stream.*`; bus: `platform.llm.*` / `agents.*` |

#### 4.4.7 Production metadata (`EventCatalogEntry` · SAR accepted)

Each spine type is described by a single **`EventCatalogEntry`** in `event_catalog.py` (SSOT):

| Field | Role |
|-------|------|
| `phase` | `ExecutionPhase` — Nexus lifecycle placement |
| `ops_hint` | Stable ops scrape / alert routing token |
| `category` | `EventCategory` — subscriber and metrics grouping |
| `preferred_payload_schema_id` | Merged from payload registry |
| `sample_rate` | `1.0` default; `<1.0` for high-volume types (`TASK_PROGRESS`) — enforced at bus persist (OBS-EVOL-9.6) |
| `retention_class` | `operational` \| `audit` \| `debug` — ties to data classification retention (IDEAL-23.5) |
| `consolidation_kind` | Target `platform.*` kind when spine member moves to `DOMAIN_SIGNAL` (OBS-EVOL-9.7) |

`phase_coverage.py` is a **deprecated view** — import catalog helpers instead.

#### 4.4.8 `EmitContext` (OBS-EVOL-9.3)

All public emit APIs accept a typed **`EmitContext`** carrying `task_id`, `run_id`, `attempt_id` (target), `tenant_id`, `correlation_id`, and active `TraceScope` — correlation by construction (SAR-01). **Target:** `TaskId`/`RunId`/`AttemptId` typed carriers (§5.3).

#### 4.4.9 Domain signal redaction (OBS-EVOL-9.3)

`emit_domain_signal()` **must** call `payload.redact()` and respect `production_mode` before `RuntimeEventBus.record` — same bar as Plane B `DiagnosticPayload` (SAR-09).

#### 4.4.10 `JournalQuery` (OBS-EVOL-9.5)

Read-model API over unified journal:

```python
query_journal(run_id, categories={EventCategory.TOOL}, kind_prefix="agents.legal.")
```

Replaces ad-hoc enum-list filtering in debug tooling (SAR-07).

#### 4.4.11 Declarative profile subscriptions (OBS-EVOL-9.10)

`ObservabilityProfile.event_subscriptions: list[EventSubscriptionSpec]` — Tier-3 declares `kind_prefix`, `categories`, `ops_hints`, and/or `event_types` plus a `handler_id`. Handlers register via `register_event_subscription_handler()`; `wire_observability_event_subscriptions()` attaches them at host bootstrap (`harness_host_runtime`). **Code:** `sub_profiles.py`, `event_subscription_registry.py`, `observability_wiring.py`.

#### 4.4.12 W3C Trace Context (OBS-EVOL-9.11)

Optional `traceparent` / `tracestate` on `RuntimeEvent` for external APM correlation. `EmitContext` propagates inbound headers; `NexusRuntimeEventPublisher` injects per-event spans from task metadata; OTLP journal export prefers W3C trace/span ids when present. **Code:** `w3c_trace_context.py`, `journal_export.py`, `export_bridge.py`.

#### 4.4.13 Spine consolidation shim (OBS-EVOL-9.7)

Nineteen legacy flat spine members (adaptive, capacity/scale, autonomy, recovery, hook) were removed from `RuntimeEventType`. Emitters use `build_platform_signal_event()` → `DOMAIN_SIGNAL` + namespaced `platform.*` kind. Persisted journals with legacy `event_type` values are coerced on read via `migrate_legacy_spine_payload()` (payload retains `legacy_spine_type`). Publication gate: `assert_publication_spine_budget()` (max 56). **Code:** `spine_consolidation.py`, `scripts/maintenance/check_event_catalog.py`.

### 4.2 Plane B — Diagnostic trace (`TraceEvent` + `DiagnosticPayload`)

Fine-grained, append-only timeline optimized for **reconstruction** and **evaluation**.

| Field | Role |
|-------|------|
| `seq` | Monotonic per `run_id` |
| `component` | `TraceComponent` (ENGINE, TOOLS, RAG, CRITIC, …) |
| `step` | Stable step identifier (e.g. `tool_invocation_start`, `critic.l1_judge`) |
| `payload` | `DiagnosticPayload` instance (typed, `schema_id`, `redact()`) |
| `tags` | Correlation metadata (`tenant_id`, `task_id`, `agent_id`) |

**Code:** `intergrax/runtime/nexus/tracing/trace_models.py`, `RuntimeState.trace_event()`

**Guard:** Non-`DiagnosticPayload` payloads are rejected at emission (gate: `test_runtime_state_trace_event_payload_guard.py`).

### 4.3 Plane C — Metrics and aggregates

| Source | What | When |
|--------|------|------|
| `RunStats.llm_usage` | Tokens, cost per run | Run finalize |
| `export_run_metrics()` | Behavioral ratios, modality summary | Debug `/metrics` |
| LLM metrics collector | Prometheus / OTLP JSON | `TASK_COMPLETED` plugin |
| RAG metrics | Retrieval latency, hit rate | `TASK_COMPLETED` / RAG plugin |
| Modality metrics | Vision/audio/tool modality counters | Trace payload aggregation |

Metrics are **third** in priority (canon §42.24): derived from events/trace, not a substitute for the journal.

---

## 5. Canonical execution identity (TRACE-ARCH-SYNC-1)

**Status:** Target canon (**accepted** 2026-08-15) · implementation **Planned** (TRACE-1A–TRACE-1C)  
**Plan:** [`plan/OBSERVABILITY.md`](../maintainers/plans/OBSERVABILITY.md) — Phase TRACE  
**Cross-layer:** [`UNIFIED_EXECUTION_RUNTIME.md`](UNIFIED_EXECUTION_RUNTIME.md) §42.1.8 (identity ownership cross-ref)

### 5.1 Identity hierarchy

Canonical execution identity is a **frozen** four-level hierarchy:

```text
Task
  1:N
Run
  1:N
Attempt
  1:N
RuntimeEvent
```

| Identifier | Meaning |
|------------|---------|
| `TaskId` | **WHAT** task / intent |
| `RunId` | **WHICH** execution of the task |
| `AttemptId` | **WHICH** attempt inside the run |
| `EventId` | **WHICH** runtime event |

Every canonical `RuntimeEvent` **MUST** carry all four: `TaskId`, `RunId`, `AttemptId`, `EventId` — without exception.

### 5.2 Strong typing (target canon)

Canonical in-process identifiers **`TaskId`**, **`RunId`**, **`AttemptId`** **MUST** be non-interchangeable typed identifiers.

**Normative implementation pattern:**

```python
TaskId = typing.NewType("TaskId", str)
RunId = typing.NewType("RunId", str)
AttemptId = typing.NewType("AttemptId", str)
```

Wire representation remains a flat string. Architecture describes this as the normative target; the plan marks it **planned until implementation exists** (TRACE-1A).

`EventId` is the unique identity of a single persisted runtime event.

### 5.3 Identity carrier matrix (target canon)

Canonical identity **MUST NOT** come from metadata. Forbidden patterns include `metadata["run_id"]`, `task_id or run_id`, `run_id or task_id`, fallback of one identity into another, dynamic identity binding, and `dict[str, Any]` as the canonical identity carrier.

| Carrier | `TaskId` | `RunId` | `AttemptId` |
|---------|----------|---------|-------------|
| Task | REQUIRED | NOT PRESENT | NOT PRESENT |
| `RuntimeRequest` execute boundary | REQUIRED | REQUIRED | NOT PRESENT |
| `RuntimeExecutionContext` | REQUIRED | REQUIRED | REQUIRED |
| `EmitContext` | REQUIRED | REQUIRED | REQUIRED |
| `RuntimeEvent` | REQUIRED | REQUIRED | REQUIRED |

**Mint ownership (lifecycle boundary, not observability):**

| Identifier | Minted by |
|------------|-----------|
| `TaskId` | Task lifecycle owner |
| `RunId` | Run lifecycle owner |
| `AttemptId` | Attempt lifecycle owner |
| `EventId` | RuntimeEvent / event creation owner |

`AttemptId` is minted by the owning attempt lifecycle boundary. The observability spine **receives and propagates** canonical identity by construction — it does **not** mint `TaskId`, `RunId`, or `AttemptId` as lifecycle owner. Observability records and propagates them; carriers receive identity by construction — not by ad-hoc metadata lookup.

### 5.4 Attempt lifecycle, retry, resume, replay

```text
Run starts
  ↓
AttemptId A1 minted
  ↓
all RuntimeEvents belong to A1
  ↓
retry
  ↓
same TaskId
same RunId
new AttemptId A2
```

| Scenario | `TaskId` | `RunId` | `AttemptId` |
|----------|----------|---------|-------------|
| **Retry** | same | same | **new** |
| **Resume** (without retry) | same | same | **same** |
| **Explicit new execution** of same task | same | **new** | **new** A1 |

Replay semantics are attempt-scoped: reconstruction and as-of projections respect attempt boundaries (§7).

### 5.5 `TASK_CREATED` semantics

`TASK_CREATED` is the **first runtime journal event** inside Run R1 / Attempt A1. It does **not** denote the moment the `Task` object was created in memory or registered in a product store.

### 5.6 Implementation gap (documentation truth)

The **target canon** above is **not** fully implemented. Known legacy gaps in code today include:

- `run_id == task_id` aliasing
- identity carried in metadata instead of typed carriers
- missing `AttemptId` on emit paths and persisted events
- identity fallbacks (`task_id or run_id`, `run_id or task_id`)
- loose journal adapters that tolerate missing or aliased identity

TRACE-1A–TRACE-1C close these gaps. Do **not** treat current runtime behavior as satisfying §5.

### 5.7 Pre-production clean-cut policy

Intergrax is **pre-production** — there are no active production platform users. Canonical TRACE delivery therefore uses a **clean cut** to target architecture:

```text
Unused legacy contracts are removed rather than preserved.
```

**Consequences for identity, journal, and checkpoint paths:**

- no compatibility aliases for unused legacy identity
- no dual canonical schemas (old + new)
- no deprecated-but-supported identity contracts kept indefinitely
- no migrations for unused persisted formats (including old `RuntimeCheckpoint` shapes)
- no fallback to old metadata identity
- no silent interpretation of old identity semantics
- no permanent parallel old/new ownership

If an old capability is still genuinely used by the current repo runtime, tests, or product path: migrate that live code directly to the canonical contract, then **delete** the old path. Do **not** preserve both.

Temporary recognition of legacy shapes is acceptable only during a bounded implementation step when technically unavoidable — it is **not** target architecture.

---

## 6. Unified Run Journal (canonical run read model)

The **Unified Run Journal** is the canonical **run-scoped read model** for operator reconstruction and downstream narrative surfaces.

```text
RuntimeEvent / persistence
        ↓
Unified Run Journal
        ↓
query / derived read models
```

| Property | Requirement |
|----------|-------------|
| Role | Chronological derived execution timeline — **WHAT happened** |
| Source of truth | **NOT** — persistence of `RuntimeEvent` remains authoritative |
| Replaces event store | **MUST NOT** |
| Scope | Composes chronological history per run (attempt-aware ordering) |
| Execution Story | Canonical foundation for Execution Story read surfaces (§10) |
| Construction | `build_unified_run_journal()` merges spine-normalized events into one timeline |

The journal is a **derived view**. Metrics, external APM, and product summaries subscribe to or export from it — they do not fork a competing timeline.

---

## 7. First-class as-of projections (TRACE-ARCH-SYNC-1)

**Status:** Target canon (**accepted** 2026-08-15) · **TRACE-ASOF-1** execution position + `AsOfBoundary` **Done / Closed** (`02462d96897daa4ea19d96dce776768a03cbbf53`) · **TRACE-ASOF-2** run execution lifecycle projection **Done / Closed** (`d0cfad1eeecbf3167e3955b93d4a2ef82de09b4f`) · query/materialization surfaces **Planned** (TRACE-ASOF-3–TRACE-ASOF-4) · compatible with bitemporal knowledge basis (§8)

**TRACE-ASOF-1 evidence chain:** `ae618fc81817497dbbcf018d92c95856f2d44115` → `d88253dbcfaa470597f93d91eec6a80a30e77007` → `98a2d186d9b512048c01024b67f1e707d72240ee` → `a7a931c6a5c4356e9bd49d7d9f8b5787e9a826b6` → `02462d96897daa4ea19d96dce776768a03cbbf53`.

### 7.1 Capability definition

A **First-Class As-Of Projection** is a typed, deterministic reconstruction of execution state at an explicit historical execution boundary.

> Typowana, deterministyczna rekonstrukcja stanu wykonania dokładnie na wskazanej granicy historycznej.

### 7.2 Journal vs as-of

| Surface | Question |
|---------|----------|
| **Unified Run Journal** | **WHAT happened?** — chronological facts |
| **As-Of Projection** | **WHAT did this execution see / do by boundary X?** — execution state at a deterministic journal boundary |
| **Bitemporal State** (§8) | **WHAT was valid, according to knowledge recorded by system time S?** — valid-time + system-time basis only (execution boundary is separate, §7) |

Conceptual example (Run R1):

```text
Attempt A1
  E1 intake
  E2 agent = Agent-A
  E3 context revision = C12
  E4 policy = ALLOW
  E5 tool
  E6 validation = FAILED
  E7 retry

Attempt A2
  E8 agent = Agent-B
  E9 validation = PASS
```

`as-of(E6)` may represent:

```text
Task = T1
Run = R1
Attempt = A1
Agent = Agent-A
ContextRevision = C12
Policy = ALLOW
Validation = FAILED
```

**TRACE-ASOF-2** freezes the first canonical logical projection contract in §7.3.1.

### 7.3.1 Run execution lifecycle projection (`TRACE-ASOF-2`)

**Status:** **Planned / In Review** (independent audit closes it).

| Concept | Type / API | Owner | Semantics |
|---------|------------|-------|-----------|
| **Projection result** | `RunExecutionAsOfProjection` | `intergrax.runtime.events.asof_projection` | Immutable run-scoped execution/lifecycle state at inclusive `AsOfBoundary` |
| **Lifecycle status** | `RunExecutionLifecycleStatus` | same | Closed enum derived from `RuntimeEventType` only — `CREATED`, `RUNNING`, `PAUSE_REQUESTED`, `PAUSED`, `CANCELLATION_REQUESTED`, `COMPLETED`, `FAILED`, `CANCELLED` |
| **Attempt summary** | `AttemptAsOfSummary` | same | Per-attempt first/last position + event count reconstructed from prefix |
| **Source provenance** | `HistoricalEventReference` | same | `EventId` + `ExecutionEventPosition` + `AttemptId` + `RuntimeEventType` — no payload copies |
| **Pure reducer** | `project_run_execution_as_of(...)` | same | Deterministic fold over positioned prefix; no persistence, clock, or live state |
| **Read orchestration** | `reconstruct_run_execution_as_of(...)` | same | Tenant-scoped load via `load_positioned_run_journal_through` then reducer |
| **Positioned journal load** | `load_positioned_run_journal_through(...)` | `unified_run_journal` | Single authority for prefix completeness and exact boundary existence; paginates by increasing `limit` until prefix complete |

**Rules (TRACE-ASOF-2):**

1. Canonical input is `list_positioned_through(boundary)` semantics — `position <= boundary.position`, strict increasing positions, same `RunId`.
2. Reducer **MUST NOT** parse `RuntimeEvent.payload` or use timestamp ordering.
3. `PAUSE_REQUESTED` ≠ `PAUSED`; `CANCELLATION_REQUESTED` ≠ `CANCELLED`; `HUMAN_APPROVAL_REQUESTED` does not imply run failure.
4. Attempt history is first-seen execution-position order; `RETRY_STARTED` introduces a new `AttemptId`; `RESUMED` preserves attempt identity.
5. `AsOfBoundary` is a **stable historical coordinate**: reconstruction is valid only when the boundary position corresponds to an accepted `PositionedRuntimeEvent` in canonical history. A nonexistent future position **MUST** fail (`RunExecutionBoundaryNotFoundError`). On success, `last_included_position == boundary.position`.
6. Unknown run with no positioned history **MUST** fail (`RunExecutionHistoryNotFoundError`) — not an empty projection.
7. Prefix reads **MUST NOT** silently truncate: incomplete reads fail closed (`RunExecutionHistoryTruncatedError`).
8. Logical-only — **no** projection persistence, store, or materialized view (TRACE-ASOF-3).
9. **No** `KnowledgeRevisionPosition` / bitemporal types in the execution reducer — E-only reconstruction.
10. `TASK_COMPLETED` and `CANCELLED` are irreversible run-terminal statuses; `TASK_FAILED` / `PLAN_FAILED` represent the current retryable failure state (not canonical finality); `RETRY_SCHEDULED` preserves `FAILED`; `RETRY_STARTED` transitions `FAILED` → `RUNNING`.
11. `RunExecutionAsOfProjection.is_terminal` is `True` only for `COMPLETED` and `CANCELLED`. `FAILED` is **not** terminal — canonical runtime permits `FAILED` → `RETRY_SCHEDULED` → `RETRY_STARTED` → `RUNNING`. There is currently **no** distinct canonical `RuntimeEventType` for final non-retryable run failure (e.g. retries exhausted); if one is introduced later, projection finality may be extended under a separately reviewed contract change.

**Forbidden:** `dict[str, Any]` projection fields; dynamic projection registry; payload-key lifecycle inference; timestamp-ordered reducer input; second source of truth.

### 7.3 Execution position and as-of boundary (`AsOfBoundary`)

Canonical execution-history ordering is **not** timestamp-based. For one accepted `RunId`, every persisted `RuntimeEvent` receives exactly one **execution position** at the persistence acceptance boundary.

| Concept | Type | Owner | Semantics |
|---------|------|-------|-----------|
| **Execution position** | `ExecutionEventPosition` | `RuntimeEventPersistence.append` | Positive, immutable, tenant + run-scoped, unique among accepted events, strictly monotonic acceptance order, stable after acceptance, gap-tolerant, non-recyclable |
| **Positioned event** | `PositionedRuntimeEvent` | persistence read APIs | Semantic `RuntimeEvent` + authoritative position — position is **not** stored on `RuntimeEvent` |
| **As-of boundary** | `AsOfBoundary` | query / projection callers | `RunId` + inclusive execution position (`<= position`) |

**Rules (TRACE-ASOF-1):**

1. Scope is **per `RunId`** (tenant-scoped store partition) — not global across runs.
2. Producers own `EventId`, identity fields, and semantic `timestamp`; persistence owns position allocation.
3. Idempotent append on the same `EventId` returns the **same** position — no duplicate allocation.
4. Concurrent acceptance for the same run yields one total order with distinct positions (store-level transaction/lock semantics).
5. `AttemptId` does **not** reset position — retries and resumes continue the run-level sequence.
6. `EventId` is identity only — **not** ordering authority.
7. `RuntimeEvent.timestamp` remains diagnostic/display — it does **not** define canonical execution order.
8. Execution position is independent of valid-time and system-time (§8) — do **not** call it bitemporal.
9. **Monotonic** means strictly increasing for accepted events — it does **not** mean contiguous. Positions **may** contain gaps (for example P1 → accepted event, P2 → unused reservation, P3 → accepted event). An unused position is **never** recycled.
10. Gaps may occur when concurrent duplicate `EventId` acceptance races allocate a candidate position before one writer wins, when candidate acceptance fails after allocation, or when backend retries/reservations consume sequence slots without producing an accepted event.

**Persistence contract (minimum):**

- `append(...) -> PositionedRuntimeEvent`
- `list_positioned_for_run(..., through: ExecutionEventPosition | None = None)` — oldest position first; `through` selects the inclusive prefix for `AsOfBoundary`
- `list_for_run` derives event order from positioned reads

**Forbidden:** `AsOfBoundary(timestamp=...)`, `ORDER BY RuntimeEvent.timestamp`, `(timestamp, event_id)` tie-break as authoritative order, producer-side position minting, exposing backend row ids as the public position type.

Execution position + boundary semantics are **TRACE-ASOF-1**. Run execution lifecycle logical reconstruction is **TRACE-ASOF-2** (§7.3.1).

### 7.4 Projection properties

Canonical as-of projection **MUST** be:

- derived
- deterministic
- typed
- run-scoped
- attempt-aware
- immutable as a historical result
- reconstructable from canonical history
- traceable back to source `RuntimeEvent` references
- free from metadata identity fallback

**MUST NOT** be:

- a new source of truth
- a new event store
- an arbitrary mutable snapshot
- `dict[str, Any]`
- dynamic projection binding

Projection **MUST NOT** be named or classified as proof or evidence. Projection **SHOULD** contain or enable resolution to source event references.

### 7.5 Logical vs materialized projection

| Kind | Meaning |
|------|---------|
| **Logical projection** | Deterministically derived from `RuntimeEvent` history |
| **Materialized projection** | Optional performance optimization — **MUST NOT** change semantics; **MUST** be rebuildable; **MUST NOT** become a competing source of truth |

Materialization is **not** mandatory for as-of capability (TRACE-ASOF-3 is conditional).

### 7.6 Revision / supersedes (materialized only)

For **persisted / materialized** projection revisions only (not every `RuntimeEvent`):

```text
If an as-of projection is persisted/materialized,
each materialized revision SHOULD have explicit immutable revision identity
and MAY reference the revision it supersedes.
```

```text
ProjectionRevision P1
   ↓ superseded by
ProjectionRevision P2
   ↓ superseded by
ProjectionRevision P3
```

Goals: projection history is not overwritten; operators can audit which revision was available; the current revision does not destroy earlier ones. Field-level schema is deferred to TRACE-ASOF-3.

### 7.7 Relationship to bitemporal state (§8)

As-of projections and bitemporal historical state answer **different questions**. Execution as-of is accepted and planned (this section). Bitemporal valid-time / system-time semantics are also **accepted target capability** with **planned implementation** (§8, TRACE-BITEMP-1–TRACE-BITEMP-5). Neither replaces the other.

---

## 8. First-class bitemporal historical state (TRACE-BITEMP-ARCH-SYNC)

**Status:** Target canon (**accepted** 2026-08-15; acceptance linearization + fenced-out/orphaned durable commit semantics **TRACE-BITEMP-ARCH-SYNC-R7** 2026-08-17; unresolved position resolution / lease / fencing / auditable terminalization **TRACE-BITEMP-ARCH-SYNC-R6** 2026-08-17; watermark finality / gap semantics **TRACE-BITEMP-ARCH-SYNC-R5** 2026-08-16; revision-ordering authority / provider contract **TRACE-BITEMP-ARCH-SYNC-R4** 2026-08-16) · **TRACE-BITEMP-1** typed contracts **Done / Closed** (`d68c72177403fb634fd4ede2d0252e9814d7adee`) · **TRACE-BITEMP-2** canonical provider **Planned / In Review** · **TRACE-BITEMP-3** K-only knowledge reconstruction at finalized watermark **Done / Closed** (`5c2eedca75fc32101ea7a35e332c2abb3af24985`) · **TRACE-BITEMP-4** temporal query/audit (Valid Time + System Time, T→K, optional E+K composition) **Planned** · TRACE-BITEMP-5 **Planned**

### 8.1 Capability definition

**Bitemporal Historical State** (also: **Bitemporal Knowledge Reconstruction**) is a typed, deterministic, immutable-history-oriented capability for selecting and reconstructing facts using **both** temporal axes. It is correction-preserving, queryable across valid-time and system-time, provenance-linked, compatible with as-of projections (§7), rebuildable where derived, and never dependent on mutable current state alone.

Bitemporality is a **semantic model** — not merely two datetime fields. TRACE-BITEMP-1 freezes the typed bases in `intergrax.contracts.bitemporal_knowledge`: `ValidTimeBasis` and `SystemTimeBasis` (instant or half-open interval; `end is None` = open-ended; timezone-aware instants only). This is **not** a storage schema.

### 8.2 Valid time

**Valid time** (`ValidTimeBasis`) answers: **when was a fact actually valid / effective in the modeled domain?**

Domain/effective truth — independent of when Intergrax learned or recorded it. Supports retrospective corrections, backdating, and future-effective changes without collapsing “what was true on date D” into “when we wrote it down.”

### 8.3 System time

**System time** (`SystemTimeBasis`) answers: **when did Intergrax know, record, or accept that version of the fact?**

Recorded/known-by-Intergrax truth — the knowledge history of the platform. A later correction **must not** destroy what Intergrax previously believed; queries must eventually distinguish **history as currently known** from **history as believed at system-time S1**.

Conceptual example:

```text
Aug 10 — Intergrax records Policy P1 (valid from Aug 1)
Aug 15 — correction Policy P2 (actually valid from Jul 28)

A) "What did Intergrax believe on Aug 10?"  → system-time historical truth
B) "What do we now know was valid on Aug 10?" → valid-time truth using current knowledge
```

Where deterministic knowledge ordering is required, a wall-clock system-time question **SHOULD** resolve to an authoritative knowledge/revision watermark (§8.4) **before** reconstruction. Wall-clock time remains the query input / temporal basis; it does **not** define acceptance order.

### 8.4 Independent reconstruction coordinates and ordering primitives

Architecture distinguishes **independent reconstruction coordinates and ordering primitives** — do **not** collapse them:

| Primitive | Kind | Question |
|-----------|------|----------|
| **Execution AsOfBoundary** (§7) | Execution-history position | **WHERE** in this run / journal are we reconstructing? (`AsOfBoundary` = `RunId` + inclusive `ExecutionEventPosition`) |
| **Valid time** | Bitemporal temporal axis (`ValidTimeBasis`) | **WHEN** was this fact actually effective / true in the modeled domain? |
| **System time** | Bitemporal temporal axis (`SystemTimeBasis`) | **WHEN** did Intergrax know / record / accept this version of the fact? |
| **KnowledgeRevisionWatermark** | Authoritative **finalized contiguous** knowledge-order upper bound | Reconstruct using accepted knowledge/revisions **up to** finalized watermark K; not highest allocated |

**Bitemporal state** means **only** valid-time + system-time — **two** temporal axes. It does **not** include execution boundary. **KnowledgeRevisionWatermark** / **KnowledgeRevisionPosition** is **not** a third temporal axis. **Execution AsOfBoundary** is **not** part of bitemporality. Ordering positions and watermarks are deterministic reconstruction/order primitives.

Conceptual structure (TRACE-BITEMP-1 frozen types):

```text
BitemporalKnowledgeBasis
    ├── Valid-Time Basis
    └── System-Time Basis
```

and separately:

```text
Execution AsOfBoundary
```

Higher-level historical reconstruction may combine:

```text
HistoricalExecutionBasis (conceptual)
    ├── Execution AsOfBoundary E
    ├── KnowledgeRevisionWatermark K
    └── BitemporalKnowledgeBasis
        ↓
Historically Reproducible Execution State
```

The combined result is **not** “bitemporal state”. **E** and **K** remain different semantic coordinates/boundaries.

#### Semantic questions (distinct)

| # | Question |
|---|----------|
| 1 | What happened by execution boundary E42? |
| 2 | What was valid at domain time V? |
| 3 | What did Intergrax know at system time S? (wall-clock query input — resolve to watermark K where deterministic knowledge ordering is required) |
| 4 | What did execution E42 operate against, using facts valid at V and known by S (at watermark K)? |

Question 4 is **combined historical execution reconstruction** — not bitemporal state alone.

#### Difference from Execution As-Of (§7)

| Surface | Axis / primitive | Question |
|---------|------------------|----------|
| **Execution As-Of** (`AsOfBoundary`) | Execution history | What did this execution see / do by boundary X? |
| **Valid time** | Domain effectiveness (bitemporal axis) | What was valid / effective at time T? |
| **System time** | Platform knowledge (bitemporal axis) | What did Intergrax know / record at time S? |
| **Knowledge/revision watermark** (`KnowledgeRevisionWatermark`) | Authoritative knowledge-order upper bound | Reconstruct using accepted revisions **up to** K — **not** “all records whose producer timestamp ≤ T” |
| **Bitemporal state** | Valid time + System time | What was valid, according to knowledge recorded by system time S? |
| **Historically reproducible execution state** | Execution boundary + knowledge watermark + bitemporal knowledge basis | What did execution E42 operate against, using facts valid at V and known by S at watermark K? |

```text
RuntimeEvent history
        ↓
Execution AsOfBoundary
        ↓
"What did this execution see / do by boundary X?"

Bitemporal fact history
        ↓
authoritative knowledge/revision ordering (K1 → K2 → K3)
        ↓
KnowledgeRevisionWatermark K
        ↓
Valid-Time Basis + System-Time Basis
        ↓
"What was valid, according to knowledge recorded by system time S, reconstructed at K?"

Execution AsOfBoundary E + KnowledgeRevisionWatermark K + BitemporalKnowledgeBasis
        ↓
Historically Reproducible Execution State
```

Do **not** merge **Execution AsOfBoundary E** with **KnowledgeRevisionWatermark K**. Do **not** merge these into one generic timestamp. Do **not** call the combined result “bitemporal state”.

#### Knowledge / revision ordering (distinct from execution ordering)

For **bitemporal-capable immutable fact/revision history**, every accepted correction/revision that participates in bitemporal historical state **MUST** have a deterministic position in an **authoritative knowledge/revision ordering**.

The purpose of this ordering is to make the **history of corrections itself auditable**. It must support deterministic answers when:

- two services have clock skew;
- several corrections arrive close together;
- corrections are ingested concurrently;
- an old domain fact is corrected after its effective date;
- several corrections supersede or refine the same prior fact;
- system-time timestamps are equal, ambiguous, or not trustworthy for total ordering.

**Critical semantic rule:** **System time is a temporal axis, not sufficient by itself as authoritative correction ordering.** Architecture **MUST NOT** define correction ordering as timestamp-only semantics (e.g. `ORDER BY system_time` or equivalent). A stable ordering position / cursor / sequence / revision position is required; the exact typed contract belongs to **TRACE-BITEMP-1**.

Knowledge/revision ordering is **not** a third bitemporal time axis. Bitemporal state remains **valid time + system time** only (§8.2–§8.3).

Conceptually, execution ordering, knowledge/revision ordering, and the two bitemporal temporal axes are **independent**:

```text
Execution history:   E1 → E2 → E3 → E4
Knowledge history:   K1 → K2 → K3 → K4
Valid time:          V
System time:         S
```

A correction accepted at knowledge position **K20** after execution **E42**:

- **MUST NOT** be retroactively inserted into E42's original execution sequence;
- **MUST NOT** rewrite what execution E42 actually knew at that boundary;
- **MUST** receive its own deterministic position in knowledge/revision history;
- **MAY** alter what the platform **now knows** was valid at an earlier valid time;
- **MUST** preserve the previous system-time belief.

Higher-level historically reproducible execution reconstruction may therefore conceptually combine:

```text
Execution AsOfBoundary E
+ KnowledgeRevisionWatermark K
+ Valid-Time Basis
+ System-Time Basis
```

TRACE-BITEMP-1 freezes the exact runtime types in `intergrax.contracts.bitemporal_knowledge`. Do **not** merge **E** with **K**.

#### Semantic questions (extended)

| # | Question |
|---|----------|
| 5 | Which correction/revision was accepted before/after knowledge/revision position K? |
| 6 | In what authoritative order were corrections K1 → K2 → K3 accepted? |
| 7 | What do we now know was valid at the time of execution E42? |
| 8 | What did the system believe was valid when E42 executed? |
| 9 | What was the authoritative knowledge watermark at system time S? |
| 10 | What revisions were accepted up to watermark K, and what was known at K? |
| 11 | What did execution E operate against using knowledge watermark K? |

Questions 5–6 and 9–10 require knowledge/revision ordering — **not** timestamp replay alone. Questions 7–8 and 11 require combined reconstruction (execution boundary + watermark + bitemporal knowledge basis) without mutating E42's execution history.

#### Knowledge / revision watermark

**KnowledgeRevisionWatermark** is the frozen TRACE-BITEMP-1 type for a **stable authoritative finalized contiguous upper boundary** in knowledge/revision ordering (see §8 revision position lifecycle and §8.11).

Conceptually:

```text
K1 → K2 → K3 → K4 → K5
```

Watermark **K3** means: reconstruct using knowledge/revisions accepted **up to the authoritative knowledge position K3**.

It **MUST NOT** mean: all records whose producer timestamp `<=` some timestamp.

The watermark is based on **authoritative accepted revision ordering**, not producer/service wall-clock timestamps.

#### Revision position lifecycle and watermark finality (TRACE-BITEMP-ARCH-SYNC-R5)

Revision-order positions have a **provider-independent lifecycle**. Frozen type: `KnowledgeRevisionPositionLifecycle` (`ALLOCATED`, `ACCEPTED`, `UNRESOLVED`, `TERMINAL_NON_COMMITTED`).

| Conceptual state | Meaning |
|------------------|---------|
| **ALLOCATED** | An authoritative position has been reserved/assigned but is **not** yet known to be safely visible as accepted knowledge |
| **COMMITTED / ACCEPTED** | The revision has reached the canonical accepted state and is durably associated with its authoritative position |
| **UNRESOLVED / IN_FLIGHT** | The position may still become committed/accepted; readers **cannot** safely advance a stable knowledge watermark past it |
| **TERMINAL_NON_COMMITTED** | The position can **never** become an accepted revision and has reached a durable terminal outcome (e.g. explicit VOID / ABORTED / RETIRED semantics) |

**Allocated position ≠ accepted revision.** Revision position allocation and revision acceptance **may** be separate internal provider steps. Canonical semantics expose acceptance only after the contract's atomic acceptance requirements are satisfied. An allocated-but-unaccepted position **MUST NOT** appear as accepted knowledge — regardless of whether the canonical implementation later uses one DB transaction, sequencer + durable acceptance, CAS, or another mechanism.

**KnowledgeRevisionWatermark MUST NOT mean highest allocated position.** A provider may allocate **K** before the corresponding revision is durably accepted. Examples: transaction allocates a sequence value then rolls back; external sequencer allocates **K** but the acceptance write fails; process crashes between allocation and durable acceptance; concurrent acceptance remains unresolved. Highest-allocated may therefore expose a boundary that contains unresolved knowledge history. Canonical readers **MUST NOT** infer completeness from allocation alone.

**Finalized contiguous watermark semantics.** **KnowledgeRevisionWatermark K** represents the highest authoritative position such that **every position ≤ K** within the applicable ordering scope has reached a **durable terminal outcome** and **no unresolved/in-flight allocation remains below K**.

A terminal outcome may be:

- **accepted/committed** revision, **or**
- **explicit durable terminal non-committed** outcome

Do **not** define the watermark as "highest contiguous committed" if that would make a permanent rollback gap block advancement forever. Instead:

```text
FINALIZED = COMMITTED/ACCEPTED  OR  DURABLY TERMINAL-NON-COMMITTED
```

Example — watermark **may** advance across a permanent terminal gap:

```text
K1 COMMITTED
K2 TERMINAL_NON_COMMITTED
K3 COMMITTED
K4 COMMITTED
→ watermark MAY advance to K4
```

Example — unresolved gap **blocks** advancement:

```text
K1 COMMITTED
K2 UNRESOLVED
K3 COMMITTED
K4 COMMITTED
→ watermark MUST NOT advance beyond K1
```

**No invisible gaps below watermark.** There **MUST NOT** be an unresolved or semantically unknown gap below a published **KnowledgeRevisionWatermark**. Every position ≤ watermark **MUST** be deterministically classifiable. Readers **MUST** be able to distinguish accepted revision positions from terminal non-committed positions without reconstructing provider-specific allocation behavior. A provider-specific "missing row" is **not** sufficient canonical semantics — absence alone **MUST NOT** ambiguously mean rolled back, still pending, never allocated, lost write, or provider bug. **TRACE-BITEMP-1** **MUST** define how terminal non-committed positions are represented semantically; **TRACE-BITEMP-2** chooses physical persistence representation. Architecture freezes terminal-non-committed **semantics**, not necessarily a physical "void record" implementation — future providers may use tombstone/void revision state, allocator ledger, transactional status row, sequencer finalization metadata, or another provider-specific representation behind the canonical contract.

**Idempotent acceptance / dedup identity.** Every logical revision acceptance **MUST** have a stable idempotency/dedup identity. Conceptually:

```text
accept(revision_id R, acceptance_key A) → position K

retry accept(same revision_id R, same acceptance_key A)
  → same accepted semantic result
  → same authoritative position K

accept(same acceptance_key A, different revision_id R2)
  → RevisionAcceptanceConflictError
  → MUST NOT return K for R1
```

`KnowledgeRevisionId` (`krev_` + 32 hex) identifies **what** immutable logical revision is being accepted. It is minted by the knowledge revision lifecycle **before** `accept_revision` — the ordering authority **consumes** it and **MUST NOT** mint revision identity during acceptance. `RevisionAcceptanceKey` identifies **which** logical acceptance operation / retry identity. `KnowledgeRevisionPosition` is **where** acceptance sits in authoritative tenant-scoped knowledge order. These roles are distinct from `EventId`, `RunId`, and `supersedes` lineage.

A retry **MUST NOT** create a second accepted revision merely because the original caller did not receive the response. A retry **MUST NOT** consume a semantically different authoritative position for the same already-accepted logical operation. Exact typed key/name and ownership/scope belong to **TRACE-BITEMP-1** — architecture does **not** assume the key is generated by the client.

**Failure / crash semantics (contract requirements).** **TRACE-BITEMP-1** **MUST** define behavior for at least:

| Scenario | Required eventual outcome |
|----------|---------------------------|
| **A** Position allocated; acceptance succeeds | Accepted/committed at **K** |
| **B** Position allocated; acceptance rolls back | Terminal non-committed or explicit unresolved until resolved |
| **C** Position allocated; process crashes before acceptance outcome is known | Remains explicitly unresolved until classified, or becomes accepted/terminal non-committed |
| **D** Acceptance durably commits; caller times out before response | Accepted at **K**; retry returns same semantic result |
| **E** Retry occurs after **D** | Same semantic **K** — no duplicate accepted revision |
| **F** Sequencer/provider issued **K** but durable revision write never commits | Terminal non-committed or explicitly unresolved — watermark cannot advance past unresolved **K** |
| **G** Provider recovers after restart with unresolved positions | Each position eventually becomes accepted/committed **or** terminal non-committed, or remains explicitly unresolved such that watermark cannot advance past it |

No silent ambiguous state.

#### Unresolved position resolution, lease/fencing, and auditable terminalization (TRACE-BITEMP-ARCH-SYNC-R6)

R5 freezes lifecycle states and finalized-contiguous watermark semantics. R6 freezes **who may resolve** `UNRESOLVED` positions, **how stale writers are fenced**, **how terminalization is audited**, and **how watermark liveness is preserved** without sacrificing safety.

**Resolution semantic ownership.** The transition:

```text
UNRESOLVED → TERMINAL_NON_COMMITTED
```

is a **governed lifecycle resolution** owned semantically by the Observability / Bitemporal domain through **`RevisionOrderingAuthority`**. Applications, agents, arbitrary business logic, and generic Platform Plugin wrappers **MUST NOT** independently declare knowledge revision positions void. Resolution is a **sub-capability** of `RevisionOrderingAuthority` — not a second unrelated authority and not application-owned semantics. Exact runtime method/type names belong to **TRACE-BITEMP-2** unless already frozen in TRACE-BITEMP-1.

**Semantic authority vs resolution trigger.** Architecture distinguishes:

| Role | Meaning |
|------|---------|
| **Semantic authority** | The canonical contract deciding whether a lifecycle transition is valid |
| **Trigger / source** | What initiated a resolution attempt |

Possible triggers **MAY** include: recovery after restart; lease-expiry reaper; failed-transaction recovery; provider reconciliation; explicit operator/governance action. Triggers **MUST NOT** invent their own lifecycle semantics. Every resolution **MUST** pass through one canonical resolution rule set on `RevisionOrderingAuthority`.

```text
Recovery / Reaper / Operator
        |
        v
RevisionOrderingAuthority (canonical resolution rules)
        |
        v
validate ownership + fencing + durable state
        |
        +--> ACCEPTED
        |
        +--> TERMINAL_NON_COMMITTED
        |
        +--> remain UNRESOLVED
```

**Bounded resolution / liveness invariant.** An `UNRESOLVED` knowledge revision position **MUST NOT** be allowed to pin a tenant `KnowledgeRevisionWatermark` indefinitely without an active bounded resolution path. Every unresolved position **MUST** eventually:

- become `ACCEPTED`, **or**
- become `TERMINAL_NON_COMMITTED`, **or**
- remain explicitly `UNRESOLVED` while an active bounded recovery/resolution process continues.

The system **MUST NOT** rely on indefinite manual intervention as the default production mechanism. Manual/operator/governance action **MAY** exist as exceptional fallback. Exact timeout/SLA duration is **not** frozen here — **TRACE-BITEMP-2** owns concrete timing/configuration.

**Watermark safety and liveness (both required).**

| Property | Requirement |
|----------|-------------|
| **Safety** | Watermark **MUST NOT** pass an unresolved `K` (R5) |
| **Liveness** | Stale unresolved positions are actively driven toward a terminal outcome under bounded resolution |

Do **not** sacrifice one for the other.

**Lease semantics (conceptual).** When in-flight acceptance requires bounded ownership, architecture requires a **lease / ownership mechanism**:

- a writer/resolver temporarily owns authority to complete a particular acceptance/resolution operation;
- ownership is bounded;
- stale ownership can be superseded under canonical rules.

**Lease expiry alone MUST NOT automatically prove that a revision is safe to void.** Architecture **explicitly rejects**:

```text
lease expired → blind TERMINAL_NON_COMMITTED
```

An old writer might still be alive or might later resume. Lease expiry is a **trigger for recovery/resolution**, not sufficient proof of terminal non-commitment.

**Fencing (required).** Once recovery/resolution authority supersedes an old writer, the old writer **MUST NOT** be able to later commit or mutate the lifecycle outcome for that position. Conceptually:

```text
Writer A owns generation/fence F1
        |
        | lease/recovery superseded
        v
Recovery authority owns F2
        |
        +--> finalizes/recovers K
        |
        v
late Writer A using F1 attempts commit → MUST be rejected
        or, if already in-flight and cannot be physically cancelled,
        → durable outcome MUST NOT become canonical ACCEPTED (R7)
```

Architecture **MUST** guarantee newer authority supersedes older authority. The platform **SHOULD** prevent stale physical commit where the provider/storage allows it. But even where an already-in-flight storage transaction cannot be physically cancelled, fencing **MUST** prevent that durable outcome from becoming canonical **ACCEPTED** knowledge. Provider qualification **MUST** test both: (1) prevention where possible, and (2) safe semantic isolation where prevention is impossible. Do **not** claim every storage engine can cancel an in-flight transaction. Exact representation (fencing token, generation, epoch, version, or equivalent) belongs to **TRACE-BITEMP-2**.

**Void is not a new knowledge revision position.** Resolving existing position **K** from `UNRESOLVED` → `TERMINAL_NON_COMMITTED` **MUST NOT** allocate a new `KnowledgeRevisionPosition` merely to express that lifecycle transition.

**Forbidden:**

```text
K17 UNRESOLVED
K18 = "void K17"
```

**Correct:**

```text
K17 lifecycle: UNRESOLVED → TERMINAL_NON_COMMITTED
+ separate immutable resolution/audit record
```

The terminalization decision finalizes the **existing K**. It is **not** a new accepted knowledge revision.

**Immutable resolution record (conceptual).** Every transition to `TERMINAL_NON_COMMITTED` **MUST** be auditable via an immutable resolution record. Exact runtime type/name belongs to **TRACE-BITEMP-2**. The record **SHOULD** capture canonical safe metadata such as:

- target ordering scope / tenant
- target `KnowledgeRevisionPosition` **K**
- prior lifecycle state
- resulting lifecycle state
- resolution reason code
- resolution source (recovery; lease expiry/reaper; provider reconciliation; operator/governance; other canonical source)
- authority/fencing generation or equivalent reference
- system-time of the resolution decision
- actor/service/operator identity where applicable
- provenance/evidence reference supporting the decision
- correlation/idempotency identity where applicable

Raw payload/content is **not** required in the resolution record. The record **MUST** be immutable/audit-preserving.

**Resolution record ≠ knowledge revision.**

| Artifact | Role |
|----------|------|
| **Knowledge revision** | Changes accepted knowledge / domain fact history |
| **Resolution record** | Records how/why an existing revision position lifecycle was finalized |

The resolution record:

- **MUST NOT** receive a new knowledge revision **K** merely because it exists;
- **MUST NOT** change valid-time semantics of the underlying domain fact;
- **MUST NOT** become a new bitemporal knowledge revision by default;
- **MAY** carry system-time/audit metadata describing when platform resolution occurred;
- **MUST** remain queryable for audit/provenance.

Do **not** collapse resolution history into revision lineage.

**Late commit after terminalization (fail-closed).** Once **K** is durably `TERMINAL_NON_COMMITTED` under a newer valid resolution/fencing authority, a stale writer **MUST NOT** later transition **K** to `ACCEPTED`. `TERMINAL_NON_COMMITTED` is **terminal**. A late write using stale ownership/fence **MUST** fail canonical acceptance — and if it nevertheless becomes physically durable, it **MUST** be treated as a fenced-out/orphaned durable write (R7), not as resurrection of **K**. **`TERMINAL_NON_COMMITTED → ACCEPTED` is forbidden.** Reconciliation **MUST NOT** rewrite canonical historical meaning toward a stale durable outcome. If product/domain semantics require a later correction, it **MUST** be a new logical acceptance with a new `RevisionAcceptanceKey` and new **K** — do **not** reuse terminal **K**.

**Race: original writer vs recovery.** **TRACE-BITEMP-2** **MUST** handle writer/recovery races on the same **K**. Canonical rule: exactly one valid lifecycle outcome wins under current authoritative fencing/ownership at the authoritative linearization point (R7). **No timestamp-based winner selection.**

| Case | Outcome |
|------|---------|
| **A** Canonical acceptance linearizes before recovery obtains newer authority | `ACCEPTED`; recovery **MUST** observe `ACCEPTED` and **cannot** terminalize/void **K** |
| **B** Recovery terminalization linearizes before stale writer | `TERMINAL_NON_COMMITTED`; late writer rejected or orphaned if physically durable |
| **C** State remains ambiguous | `UNRESOLVED`; watermark remains pinned; bounded resolution continues |

**Recovery / reaper role (conceptual).** Architecture **SHOULD** define a production path such as an unresolved scanner / recovery worker / reaper responsible for:

- finding stale `UNRESOLVED` positions;
- obtaining current resolution authority/fence;
- verifying durable acceptance state;
- attempting safe recovery;
- classifying terminal state;
- writing immutable resolution record;
- enabling watermark advancement when finalized.

Process topology (daemon vs background task, scheduler, queue, cron, DB implementation) belongs to **TRACE-BITEMP-2** / operational design.

**Governance / manual action (exception path).** Explicit governance/operator terminalization **MAY** exist only as a controlled exception. It **MUST**:

- use the same canonical resolution authority on `RevisionOrderingAuthority`;
- obey the same fencing rules;
- produce the same immutable resolution record;
- never bypass unresolved-state validation;
- be fully auditable.

Manual action **MUST NOT** be a magic override that ignores current authoritative writer ownership. Force-resolution semantics, if ever allowed, require **TRACE-BITEMP-2** or a later ADR with authorization and evidence requirements. RBAC details are **not** frozen here.

**In-doubt / 2PC positions (provider-independent).** If a provider uses 2PC or another protocol capable of producing in-doubt operations:

- in-doubt **K** is `UNRESOLVED`;
- watermark **MUST NOT** pass it;
- recovery **MUST** use provider-specific evidence behind canonical resolution semantics;
- lease expiry alone is insufficient;
- resolution **MUST** eventually classify the position or keep it explicitly unresolved;
- provider-specific 2PC terminology **MUST NOT** leak into canonical reader semantics.

Architecture does **not** select or require 2PC for the canonical provider merely because this scenario is documented.

#### Acceptance linearization and fenced-out/orphaned durable commits (TRACE-BITEMP-ARCH-SYNC-R7)

R6 freezes resolution ownership, lease/fencing, and auditable terminalization. R7 freezes **authoritative acceptance/finalization linearization**, the distinction between **physical durability** and **canonical acceptance**, and production-safe semantics when a stale/fenced-out writer's in-flight persistence transaction becomes physically durable **after** a newer fencing generation has already authoritatively finalized the corresponding position **K** as `TERMINAL_NON_COMMITTED`.

**Physical durability ≠ canonical acceptance.** A physical/durable write existing in storage is **not**, by itself, sufficient to make a knowledge revision canonically `ACCEPTED`. Canonical acceptance requires:

- valid current `RevisionOrderingAuthority` ownership/fencing
- successful canonical acceptance transition
- authoritative `KnowledgeRevisionPosition` association
- lifecycle state `ACCEPTED` under the winning authority

```text
PHYSICAL DURABILITY  !=  CANONICAL ACCEPTANCE
```

A provider **MUST NOT** infer `ACCEPTED` solely because bytes/rows/documents exist in the underlying persistence layer.

**Authoritative linearization point.** Architecture **MUST** guarantee exactly one authoritative linearization point for each position lifecycle outcome. Linearization is the single canonical concurrency point that determines which outcome won: `ACCEPTED` **or** `TERMINAL_NON_COMMITTED`. **No timestamp ordering.**

| Case | Canonical rule |
|------|----------------|
| **A — acceptance-first** | Valid writer acceptance linearizes first → **K** = `ACCEPTED` → later recovery **MUST** observe `ACCEPTED` and **cannot** void **K** |
| **B — terminalization-first** | Newer fencing authority terminalization linearizes first → **K** = `TERMINAL_NON_COMMITTED` → late stale writer **cannot** canonically accept **K** → any later physical write from stale writer is fenced-out/orphaned |

The exact transactional/CAS/storage primitive belongs to **TRACE-BITEMP-2**. Do **not** freeze vendor-specific mechanics here.

**Terminalization remains authoritative.** Once **K** = `TERMINAL_NON_COMMITTED` has authoritatively linearized under the winning/newer fencing generation, that result **MUST** remain canonical. A later durable write produced under an older/stale authority **MUST NOT** cause `TERMINAL_NON_COMMITTED → ACCEPTED`. Architecture **MUST NOT** reconcile canonical lifecycle state toward a stale durable outcome. Otherwise historical watermarks could change meaning after publication.

**Historical watermark immutability.** Example:

```text
K16 ACCEPTED
K17 TERMINAL_NON_COMMITTED
K18 ACCEPTED
K19 ACCEPTED
→ watermark = K19
```

A reader reconstructing at **K19** **MUST** permanently observe **K17** as containing no accepted knowledge revision. If an old F1 persistence transaction later becomes physically durable, that **MUST NOT** retroactively alter the meaning of watermark **K19**. Canonical historical reconstruction at the same **K** **MUST** remain deterministic before and after orphan discovery.

**Fenced-out / orphaned durable write (conceptual).** Exact runtime type/name belongs to **TRACE-BITEMP-2**. Meaning: data physically reached durable persistence, but did so without valid canonical acceptance authority because its writer had already been superseded/fenced out.

```text
Writer A owns fence F1
        ↓
starts persistence transaction

Recovery obtains F2
        ↓
authoritatively finalizes K17 as TERMINAL_NON_COMMITTED
        ↓
watermark may later advance

old F1 transaction nevertheless reaches durable storage
        ↓
physical data exists
        ↓
canonical K17 remains TERMINAL_NON_COMMITTED
        ↓
late write = FENCED_OUT_DURABLE_WRITE / ORPHANED_DURABLE_WRITE
```

Such a write:

- **MUST NOT** become accepted knowledge
- **MUST NOT** resurrect `TERMINAL_NON_COMMITTED` **K**
- **MUST NOT** affect `KnowledgeRevisionWatermark`
- **MUST NOT** participate in canonical reconstruction
- **MUST** be detectable/auditable
- **MUST** enter a controlled reconciliation/quarantine path

Prefer keeping canonical **K** lifecycle unchanged and representing the storage anomaly separately. Do **not** define it as a new lifecycle value for `KnowledgeRevisionPosition` unless implementation design later proves this is necessary.

**Quarantine / reconciliation.** An orphaned/fenced-out durable write **MUST** be isolated from canonical knowledge reads:

```text
physical durable write
        +
stale fencing authority
        ↓
orphan detection
        ↓
quarantine / reconciliation
        ↓
audit + operator/recovery visibility
```

It **MUST NOT** silently enter canonical projections. Possible implementation actions **MAY** include quarantine marker, storage reconciliation record, provider-specific isolation, cleanup/garbage collection, or another safe mechanism. Physical implementation is **not** frozen here.

**Orphan / integrity evidence record (conceptual).** A detected orphaned/fenced-out durable write **MUST** produce immutable audit/integrity evidence distinct from knowledge revision lineage. Exact runtime type/name belongs to **TRACE-BITEMP-2**. The record **SHOULD** capture:

- tenant/ordering scope
- target **K**
- stale fencing generation
- winning fencing generation
- canonical lifecycle outcome
- provider/storage reference
- detection source
- reason classification
- system-time detected
- evidence/provenance reference
- reconciliation disposition/status where applicable

Raw knowledge payload is **not** required. The orphan record **MUST NOT** become a knowledge revision. Architecture **MUST NOT** allocate a new **K** merely to record the anomaly.

**Commit-before-finalization vs finalization-before-commit.** Both cases are frozen explicitly. Timestamp ordering **MUST NOT** decide which case occurred — canonical concurrency/fencing/transaction authority decides.

| Case | Sequence | Canonical outcome |
|------|----------|-------------------|
| **A** | Writer F1 canonical acceptance succeeds → **K** = `ACCEPTED` → Recovery F2 starts later | Recovery **MUST** observe `ACCEPTED`; **cannot** terminalize/void **K** |
| **B** | Recovery F2 terminalizes first → **K** = `TERMINAL_NON_COMMITTED` → old F1 transaction later becomes physically durable | **K** remains `TERMINAL_NON_COMMITTED`; late write = fenced-out/orphaned durable write → quarantine/reconciliation |

**No reconciliation by resurrection.** Architecture **explicitly rejects**: "storage contains durable revision, therefore change **K** back to `ACCEPTED`" after terminalization has linearized. If the orphaned content is still logically valid and should be accepted, create a **new** logical acceptance: new `RevisionAcceptanceKey` → new **K_new** → normal acceptance flow. Do **not** reuse terminal **K**. Do **not** mutate historical terminal outcome.

**Atomicity requirement refinement.** The canonical provider **SHOULD** coordinate acceptance key, position allocation, lifecycle transition, durable accepted revision, and fencing/generation validation inside the strongest available atomic boundary (§8.11). However architecture **MUST** still define orphan behavior because:

- qualified alternative providers may have different persistence topology
- crashes/network partitions may produce ambiguous client outcomes
- external or distributed persistence may permit physical writes after authority loss
- future provider implementations must preserve canonical semantics

Transactional default does **not** eliminate the need for explicit orphan semantics.

**Provider observational equivalence.** Every qualified provider **MUST** preserve the same winning canonical lifecycle outcome, watermark semantics, and historical reconstruction regardless of whether its persistence layer can physically prevent a late stale write.

| Provider behavior | Canonical reader observation |
|-------------------|---------------------------|
| **A** Late stale write physically rejected | Same as B |
| **B** Late stale write physically lands but is quarantined as orphaned | Same as A |

**Audit vs knowledge history.** Keep separate:

| Stream | Role |
|--------|------|
| **A** Canonical knowledge history | Accepted domain facts |
| **B** Position lifecycle history | `ALLOCATED` / `ACCEPTED` / `UNRESOLVED` / `TERMINAL_NON_COMMITTED` |
| **C** Resolution audit history | `ResolutionRecord` per terminalization |
| **D** Storage-integrity / orphan evidence | Fenced-out/orphaned durable write detection |

An orphaned durable write is a storage/integrity event. It is **not** accepted knowledge, a knowledge revision, a new **K**, revision lineage, or a valid-time correction. It **MAY** be linked to `ResolutionRecord`, provider diagnostics, integrity incident/problem signal, or operator audit surfaces.

**Production-derived decision input (now selected in §8.11).** Transactional allocation is the canonical default because revision position allocation and durable acceptance are coordinated within the same transactional boundary. Alternatives remain valid as qualified providers behind `RevisionOrderingAuthority`.

#### Wall-clock query vs reconstruction boundary

Architecture distinguishes:

| | Surface | Role |
|---|----------|------|
| **A** | Auditor/user wall-clock question | Query input / temporal basis. Example: "What did the platform know at 2026-08-10T14:00?" |
| **B** | Canonical reconstruction boundary | Authoritative `KnowledgeRevisionWatermark` K in accepted knowledge/revision order |

```text
Wall-clock system-time query T
        ↓
resolve authoritative KnowledgeRevisionWatermark K
        ↓
reconstruct knowledge state at K
        ↓
optionally combine with Execution AsOfBoundary E
        ↓
historically reproducible execution state
```

Wall-clock time is a **query input / temporal basis**. It **MUST NOT** replace deterministic revision ordering. Where deterministic knowledge ordering is required, historical reads **SHOULD** resolve time-oriented questions onto an authoritative revision boundary **before** combining with execution reconstruction.

A wall-clock system-time query ("What did the system know at time **T**?") **MAY** resolve to a **KnowledgeRevisionWatermark K** only if **K** satisfies the **finalized contiguous boundary** semantics (§8 revision position lifecycle). The resolver **MUST NOT** return a highest-allocated position containing unresolved gaps. If knowledge positions beyond the safe watermark were already allocated or even partially processed, they remain outside the canonical stable boundary until their lower gaps are finalized.

This resolution is **semantic**. Architecture does **not** claim that materialization, indexes, or a runtime resolver already exist.

#### Bounded resolution vs unbounded full-history replay

Historical audit queries **SHOULD NOT** require replaying an unbounded complete event/revision history merely because the user supplied wall-clock time.

The query model **MUST** allow bounded, indexable, or materializable resolution strategies **without changing canonical semantics**. Logical reconstruction remains **authoritative and rebuildable**. Materialization, indexes, and checkpoints remain **implementation/performance** concerns — not a competing source of truth, and **not** claimed to exist yet.

Architecture does **not** promise O(1), O(log n), database-index complexity, or any other specific performance bound before implementation design exists.

#### Revision ordering authority — domain-owned semantic contract (TRACE-BITEMP-ARCH-SYNC-R4)

Revision-ordering **semantics** are canonical platform/domain invariants. They are **not** configurable per application and **MUST NOT** be delegated to application business logic or to Platform Plugin runtime wrappers.

The Observability / Bitemporal domain **MUST** own the authoritative revision-ordering semantic contract. Frozen public type: **`RevisionOrderingAuthority`** (`intergrax.contracts.bitemporal_knowledge`).

The contract owns semantic guarantees such as:

- allocate / accept an authoritative revision position
- classify position lifecycle state (allocated, accepted, unresolved, terminal non-committed)
- **resolve** `UNRESOLVED` positions to `ACCEPTED` or `TERMINAL_NON_COMMITTED` under canonical rules (resolution sub-capability — applications **MUST NOT** void positions independently)
- enforce bounded ownership/lease and fencing so stale writers cannot commit after supersession; where physical prevention is impossible, stale durable outcomes **MUST NOT** become canonical `ACCEPTED` (R7)
- emit immutable resolution/audit records for every `TERMINAL_NON_COMMITTED` transition
- preserve monotonic ordering within the declared scope
- expose / reconstruct **KnowledgeRevisionWatermark** using finalized contiguous boundary semantics
- deterministic concurrent acceptance
- atomic association of acceptance and position
- idempotent retry semantics keyed by stable acceptance/dedup identity
- failure-safe acceptance semantics
- auditability
- deterministic historical reads
- provider-independent observational equivalence for all qualified providers

The contract **MUST NOT** delegate semantic ownership to an application, agent/model, or plugin runtime layer.

Concrete serialization **implementation** is provided behind this domain-owned typed provider contract. Provider variation is **implementation** variation — **not** semantic variation.

```text
RevisionOrderingAuthority (domain-owned semantic contract)
        |
        +-- CanonicalRevisionOrderingProvider      <-- Intergrax first-party default
        |
        +-- QualifiedAlternativeProvider
        |
        +-- QualifiedAlternativeProvider
```

#### Canonical production default provider

Intergrax **MUST** ship one canonical first-party production-grade default provider (conceptually **CanonicalRevisionOrderingProvider** — TRACE-BITEMP-2 implements it). Canonical strategy is frozen in §8.11: tenant-scoped transactional allocation + acceptance.

Architecture **selects the canonical strategy** in §8.11. **TRACE-BITEMP-2** implements the first-party provider behind `RevisionOrderingAuthority` without exposing vendor types on the public contract.

The canonical default **MUST**:

- give operators a safe out-of-the-box baseline
- avoid requiring applications to design distributed revision serialization
- serve as the reference implementation of **RevisionOrderingAuthority**
- be the recommended baseline for documentation and proof gates

Canonical default **≠** hardcoded implementation lock-in. Intergrax remains **opinionated enough to work out of the box** while preserving a stable extension boundary for environments with different scale, availability, persistence, or infrastructure characteristics:

```text
OPINIONATED DEFAULT
+
CONTRACT-DRIVEN EXTENSIBILITY
+
SEMANTIC INVARIANCE
```

#### Qualified alternative providers

A host/deployment **MAY** select a qualified alternative provider when deployment requirements differ. Every alternative **MUST** implement the **same** **RevisionOrderingAuthority** contract and preserve exactly the same ordering, watermark, concurrency, audit, failure, and reconstruction semantics.

**Provider-independent observational equivalence.** Provider A may allocate transactionally; Provider B may use a dedicated sequencer; Provider C may use another production-grade allocator. Canonical readers **MUST** see the same semantics for revision acceptance, authoritative position, position finality, gaps, watermark, historical reconstruction, retry/idempotency, and failure visibility. Provider swap **MUST NOT** change the meaning of **KnowledgeRevisionWatermark**. Implementation-specific allocation gaps **MUST** remain below the semantic abstraction. **R7:** providers that physically reject late stale writes and providers that quarantine landed late writes as orphaned **MUST** yield identical canonical reader state.

Examples of future implementation strategies **MAY** include (unselected here):

- transactional / storage-native sequencing
- dedicated sequencer
- distributed sequencer
- scoped sequencer
- optimistic concurrency / CAS-backed allocator
- another equivalent production-grade mechanism

Provider extensibility **MUST NOT** allow:

- timestamp-based ordering instead of authoritative ordering
- disabling monotonicity
- weakening concurrency guarantees
- changing **KnowledgeRevisionWatermark** semantics
- weakening failure atomicity
- destructive historical overwrite
- changing bitemporal valid-time / system-time semantics
- application-specific interpretation of acceptance order

Do **not** treat the list above as a selection. Kafka partitions, PostgreSQL sequences, Redis counters, Snowflake-like IDs, Lamport/vector/HLC clocks, a specific transaction model, and a specific database are likewise **unselected**.

#### Host / deployment provider selection

Provider selection is **host/deployment configuration + dependency injection** — **not** per-request behavior and **not** arbitrary application business logic.

```text
Application / Deployment Host
        |
        +-- configuration / profile
        |
        +-- DI / composition
        |
        v
RevisionOrderingAuthority
        |
        v
Selected qualified provider
```

Provider selection **MUST NOT** be:

- chosen dynamically per request
- selected by agents/models
- scattered across business application code
- independently selected by arbitrary features
- changed in a way that changes historical semantics

A specialized application **MAY** cause its deployment/host configuration to select a qualified provider for infrastructural reasons. An application **MUST NOT** define its own revision-ordering semantics.

**Forbidden model:**

```text
App A -> timestamp ordering
App B -> sequencer
App C -> weak custom ordering
```

**Correct model:**

```text
App/host deployment chooses provider P
        ↓
P implements the same RevisionOrderingAuthority contract
        ↓
same KnowledgeRevisionPosition / KnowledgeRevisionWatermark semantics everywhere
```

Applications consume the canonical semantic contract. The variation is infrastructural; the contract remains interoperable.

#### Provider vs ordering scope — independent decisions

Ordering **scope** and provider **implementation** are separate architecture decisions. TRACE-BITEMP-1 freezes them separately in §8.11: scope = **TENANT**; canonical strategy = transactional allocation + acceptance.

Do **not** encode scope into provider identity. Do **not** assume one provider supports only one scope unless future implementation evidence requires it.

| Ordering scope (example) | Provider (example) | Decision |
|--------------------------|-------------------|----------|
| TENANT | CanonicalTransactionalProvider | scope decision A + provider decision X |
| TENANT | DistributedSequencerProvider | scope decision A + provider decision Y |
| GLOBAL | DistributedSequencerProvider | scope decision B + provider decision Y |

#### Serialization contract (provider qualification criteria)

Every provider used for production-capable bitemporal ordering **MUST** be qualified against canonical invariant tests/proofs. **Qualified** **MUST NOT** mean merely loadable/discoverable.

TRACE-BITEMP-1 freezes the canonical default mechanism in §8.11 against these invariants. Canonical and alternative providers **MUST** pass the same semantic suite:

1. **Uniqueness** — every accepted bitemporal correction/revision gets one unambiguous authoritative position within its ordering scope.
2. **Monotonicity** — later accepted revisions cannot appear before earlier accepted revisions within that scope.
3. **Concurrency determinism** — concurrent accepted corrections resolve to deterministic distinct positions.
4. **Clock independence** — producer/service wall-clock timestamps cannot define authoritative ordering.
5. **Atomic acceptance** — a revision must not become "accepted" without its authoritative position being durably associated with that acceptance; allocated-but-unaccepted positions must not appear as accepted knowledge.
6. **Retry / idempotency** — retrying the same logical acceptance (same stable acceptance/dedup identity) must not create duplicate accepted revisions or consume semantically different positions incorrectly; retry must return the same semantic accepted result and authoritative position.
7. **Failure semantics** — partial failure between persistence and ordering allocation must not create ambiguous accepted history; no half-accepted revision; each position must eventually become accepted/committed, terminal non-committed, or remain explicitly unresolved blocking watermark advancement.
8. **Auditability** — auditors can determine the acceptance order without reconstructing it from timestamps.
9. **Lineage independence** — `supersedes` remains causal lineage and does **not** substitute for total/order position.
10. **Deterministic watermark resolution** — wall-clock system-time queries resolve deterministically to the correct **finalized contiguous** knowledge boundary; watermark must not mean highest allocated; no unresolved gaps below published watermark; permanent terminal gaps may be crossed.
11. **Deterministic repeated reconstruction** — same E/K/temporal basis returns deterministic equivalent state.
12. **Scope definition** — the exact ordering scope is **TENANT** (`KnowledgeOrderingScope`). Cross-scope composition is a `KnowledgeRevisionWatermarkSet`, not a global `K`.
13. **Selected ordering-scope correctness** — proof matches the scope chosen in TRACE-BITEMP-1.
14. **Cross-scope semantics** — where ordering is partitioned, composition semantics are deterministic and documented.
15. **Historical immutability** — accepted corrections are never destructively overwritten.
16. **Stale writer fencing** — once recovery/resolution authority supersedes an old writer, late commits using stale ownership/fence **MUST** be rejected; where physical prevention is impossible, late durable writes **MUST** be quarantined as orphaned and **MUST NOT** become canonical `ACCEPTED` (R7).
17. **Bounded unresolved-position resolution** — every `UNRESOLVED` position has an active bounded resolution path; indefinite manual intervention is **not** the default production mechanism.
18. **Deterministic race resolution** — writer vs recovery races resolve to exactly one valid lifecycle outcome under current authoritative fencing/ownership at the authoritative linearization point; no timestamp-based winner.
19. **No late commit after terminalization** — `TERMINAL_NON_COMMITTED` is terminal; `TERMINAL_NON_COMMITTED → ACCEPTED` is forbidden; orphaned durable writes **MUST NOT** resurrect terminal **K**.
20. **Immutable resolution audit trail** — every `TERMINAL_NON_COMMITTED` transition produces an immutable, queryable resolution record distinct from knowledge revision lineage.
21. **Watermark unpins after safe terminalization** — terminalization of a blocking gap **MAY** allow watermark advancement per finalized-contiguous rules; unresolved positions remain visible until legitimately resolved.
22. **Lease expiry is not void proof** — lease expiry may trigger recovery but **MUST NOT** alone justify blind `TERMINAL_NON_COMMITTED`.
23. **Lifecycle voiding does not allocate new K** — resolving `UNRESOLVED → TERMINAL_NON_COMMITTED` finalizes the existing position; it does **not** mint a new knowledge revision position for void semantics.
24. **Physical durability ≠ canonical acceptance** — storage presence alone **MUST NOT** imply `ACCEPTED`; canonical acceptance requires valid authority, successful transition, position association, and winning lifecycle state (R7).
25. **Authoritative linearization** — exactly one lifecycle outcome (`ACCEPTED` or `TERMINAL_NON_COMMITTED`) wins per **K**; acceptance-first blocks later void; terminalization-first blocks later canonical acceptance (R7).
26. **Historical watermark immutability** — orphan discovery **MUST NOT** retroactively change reconstruction at the same finalized watermark **K** (R7).
27. **Orphan quarantine** — fenced-out/orphaned durable writes **MUST** be isolated from canonical reads and produce immutable integrity evidence; **MUST NOT** affect watermark or allocate new **K** (R7).
28. **No reconciliation by resurrection** — legitimate later acceptance of orphaned content **MUST** use new `RevisionAcceptanceKey` + new **K** (R7).

#### Ordering scope / scalability decision boundary

A **globally** monotonic revision position gives stronger/simpler global watermark semantics but may introduce unnecessary coordination.

A **narrower** ordering scope may scale better but affects the semantics of:

- wall-clock → watermark resolution
- cross-domain reconstruction
- cross-tenant isolation
- global audit questions

TRACE-BITEMP-1 freezes these separately from provider selection (§8.11):

- ordering scope = **TENANT**
- authority owner = Observability / bitemporal domain (`RevisionOrderingAuthority`)
- consistency = unique monotonic positions per tenant; finalized-contiguous watermark
- one watermark represents one tenant, not the whole platform
- cross-tenant queries return `KnowledgeRevisionWatermarkSet` — no canonical cross-tenant total order

#### Relationship to Platform Plugins

This follows canonical **COMMON PLATFORM COORDINATION + DOMAIN-OWNED CAPABILITY CONTRACTS** (see [`PLATFORM_PLUGINS.md`](PLATFORM_PLUGINS.md)).

Platform Plugin infrastructure **MAY** eventually coordinate for externally packaged **RevisionOrderingAuthority** implementations:

- package identity
- discovery
- compatibility metadata
- trust
- qualification metadata

Platform Plugin **MUST NOT** own:

- revision ordering semantics
- acceptance semantics
- watermark semantics
- temporal semantics
- provider runtime contract

There is **no** `PlatformPlugin.execute()` or universal plugin runtime abstraction for revision ordering.

```text
Platform package / discovery coordination
        ↓
domain-owned RevisionOrderingAuthority provider
        ↓
host composition / DI
        ↓
governed bitemporal runtime
```

Runtime execution flows through **domain contracts and host composition** — not through a Platform Plugin runtime wrapper.

### 8.5 Correction semantics

Corrections are **additive** and **immutable-history-preserving**:

- correction history is **immutable** — accepted corrections are never destructively overwritten;
- corrections do **not** destructively overwrite previous belief;
- every accepted correction is **independently addressable**;
- every accepted correction has **deterministic authoritative ordering** relative to other accepted revisions/corrections;
- a revision **MUST NOT** become accepted without its authoritative position being durably associated with that acceptance;
- ordering does **not** depend solely on wall-clock timestamps;
- causal lineage (`revision_id`, `supersedes`) and authoritative ordering are **complementary** — `supersedes` alone does **not** define total correction ordering;
- valid time, system time, and ordering position / watermark are **distinct semantics** — position and watermark are **not** temporal axes.

A later revision that changes valid-time applicability **must preserve** prior system-time belief. Operators and auditors must be able to reconstruct:

- what Intergrax believed at an earlier system time (resolved to an authoritative knowledge/revision watermark where deterministic ordering is required);
- what is now known to have been valid at an earlier valid time;
- what Intergrax believed was valid at an earlier system time;
- in what authoritative order corrections were accepted — without reconstructing that order from timestamps.

Destructive overwrite of historical belief is **forbidden** for bitemporal-capable facts.

### 8.6 Relationship to `revision_id` / `supersedes` / ordering position (§7.6)

Revision lineage, temporal axes, execution boundary, knowledge/revision ordering, and watermark are **complementary, not identical**:

| Mechanism | Responsibility |
|-----------|----------------|
| **`revision_id`** | Immutable revision identity |
| **`supersedes`** | Causal/version lineage between revisions — **not** total/order position |
| **Knowledge/revision position** | Deterministic authoritative ordering of accepted revisions/corrections |
| **KnowledgeRevisionWatermark** | Stable authoritative **finalized contiguous** upper bound in that ordering; not highest allocated; type frozen in TRACE-BITEMP-1 |
| **Valid time** | Domain effectiveness (bitemporal axis) |
| **System time** | When the platform knew/recorded the revision (bitemporal axis) |
| **Execution AsOfBoundary** | Position inside execution history — independent of knowledge ordering |

A revision **may** carry temporal semantics where appropriate. `supersedes` alone is **not** sufficient for bitemporal queries or total correction ordering. Do **not** add `supersedes` to every `RuntimeEvent`.

### 8.7 Relationship to provenance / evidence / proof

| Artifact | Role |
|----------|------|
| **Provenance** | Where a fact/revision came from |
| **Evidence** | Supporting persisted evidence |
| **Proof / Receipt** | Attested / verifiable claim |
| **Bitemporal state** | Selected historical truth along valid-time and system-time — **not** proof, **not** evidence |

### 8.8 Opt-in scope — not every `RuntimeEvent`

**Critical:** bitemporality does **not** require every `RuntimeEvent` to carry `valid_from` / `valid_to`.

`RuntimeEvent` remains the canonical fact that an **execution transition** happened. System/event ordering of `RuntimeEvent` is separate from whether the **domain fact** referenced by that event has valid-time semantics.

Bitemporality **should** apply — with explicit opt-in ownership — to facts/revisions where both axes are meaningful, for example potentially:

- policy revisions
- configuration revisions
- context / knowledge facts
- external integration state
- business-domain facts
- effective permissions / rules
- versioned projections where corrections or backdating matter

This list is **not exhaustive**. Do **not** convert every Intergrax persistence model into a temporal table. Do **not** turn `RuntimeEvent` into a bitemporal or revision-sequenced universal row. The capability is reusable with explicit opt-in — not universal.

### 8.9 Persistence vendor neutrality

Architecture defines semantics and capability. TRACE-BITEMP-1 **does** freeze ordering scope (**TENANT**) and canonical provider **strategy** (single durable transactional boundary — §8.11). **No** database vendor (XTDB, PostgreSQL temporal extensions, SQL Server temporal tables, Datomic, Redis, Cassandra, etc.) is selected on the public contract. Qualified alternative providers remain allowed behind `RevisionOrderingAuthority`. Physical store implementation belongs to TRACE-BITEMP-2.

### 8.10 Implementation status

Accepted architecture · **TRACE-BITEMP-1** typed contracts **Done / Closed** in `intergrax.contracts.bitemporal_knowledge` · **TRACE-BITEMP-2** canonical first-party provider **Planned / In Review** (`CanonicalRevisionOrderingProvider` over durable SQLite via `open_revision_ordering_authority`) · acceptance linearization + fenced-out/orphaned durable commit semantics canon **TRACE-BITEMP-ARCH-SYNC-R7** · unresolved position resolution / lease / fencing / auditable terminalization canon **TRACE-BITEMP-ARCH-SYNC-R6**. Delivery: [`plan/OBSERVABILITY.md`](../maintainers/plans/OBSERVABILITY.md) TRACE-BITEMP-1–TRACE-BITEMP-5.

### 8.11 TRACE-BITEMP-1 frozen contracts

Module: `intergrax.contracts.bitemporal_knowledge`. Opt-in capability — **not** added to `RuntimeEvent`.

| Decision | Frozen type / value |
|----------|---------------------|
| Valid time | `ValidTimeBasis` (`ValidTimeBoundKind.INSTANT` \| `INTERVAL`); half-open `[start, end)`; `end is None` = open-ended; timezone-aware only; no sentinel datetime |
| System time | `SystemTimeBasis` (same instant/interval shape); **not** ordering authority |
| Bitemporal state | `BitemporalKnowledgeBasis(valid_time, system_time)` only — no `AsOfBoundary`, no `KnowledgeRevisionPosition`, no `KnowledgeRevisionWatermark` |
| Revision identity | `KnowledgeRevisionId` (`krev_` + 32 hex); owner = knowledge revision lifecycle; minted **before** `accept_revision`; authority **consumes** only — **MUST NOT** mint revision identity during acceptance; distinct from `RevisionAcceptanceKey`, `KnowledgeRevisionPosition`, `EventId`, `RunId`, `supersedes` |
| Acceptance identity | `RevisionAcceptanceKey` (`rack_` + 32 hex); owner = logical revision-acceptance operation; unique within `KnowledgeOrderingScope`; same `revision_id` + same key → same `K`; same key + different `revision_id` → `RevisionAcceptanceConflictError` |
| Position | `KnowledgeRevisionPosition(scope, value)` — `value >= 1`, clock-independent; **not** `ExecutionEventPosition` |
| Lifecycle | `KnowledgeRevisionPositionLifecycle`: `ALLOCATED`, `ACCEPTED`, `UNRESOLVED`, `TERMINAL_NON_COMMITTED`. Allocated ≠ accepted. Missing row is **not** a state |
| Watermark | `KnowledgeRevisionWatermark(scope, finalized_through_value)`; `0` = empty prefix; finalized = `ACCEPTED` **or** `TERMINAL_NON_COMMITTED`; unresolved/allocated below `K` **blocks**; terminal gap does **not** block |
| Ordering scope | **TENANT** via `KnowledgeOrderingScope.tenant_id` |
| Cross-scope | No total order across tenants. Cross-scope queries return `KnowledgeRevisionWatermarkSet`. Comparing tenant K12 with tenant K20 as one sequence is forbidden (`CrossScopeKnowledgeOrderError`) |
| Authority | `RevisionOrderingAuthority` (ABC): `accept_revision`, `position_lifecycle`, `watermark`, `records_through`, `unresolved_positions`, `acquire_resolution_authority`, `resolve_unresolved_position`. Host/DI selects provider via `open_revision_ordering_authority`. Applications **MUST NOT** invent ordering semantics or independently void positions |
| Resolution audit | `KnowledgeRevisionResolutionRecord` — immutable audit per `TERMINAL_NON_COMMITTED` transition; **not** a knowledge revision |
| Orphan / integrity evidence | `OrphanedDurableRevisionRecord` — immutable integrity evidence per fenced-out/orphaned durable write; **not** a knowledge revision; **MUST NOT** allocate new **K** (R7) |
| Canonical provider strategy | **Transactional / storage-native allocation + acceptance**: one durable transactional boundary atomically coordinating `KnowledgeRevisionId`, acceptance key, position allocation, durable acceptance/reference, lifecycle/finality, and fencing/generation validation where feasible. Explicit orphan semantics still required when physical prevention is impossible (R7). Public type remains `RevisionOrderingAuthority` only |

**Scope rationale (TENANT selected).** Intergrax persistence, isolation, deletion, and execution reconstruction are already tenant-partitioned. A tenant-scoped `K` is the natural unit of “what did Intergrax know for this tenant?” Global reconstruction is compositional: a `KnowledgeRevisionWatermarkSet`, **not** one invented global `K`.

| Alternative | Decision |
|-------------|----------|
| **GLOBAL** | Rejected — invents a cross-tenant total order the product does not require; couples tenant deletion/isolation; extra coordination without stronger reconstructability than a watermark set |
| **DOMAIN** | Rejected — splits one tenant’s knowledge history so a tenant-wide system-time question cannot resolve to a single `K` |
| **AGGREGATE / FACT STREAM** | Rejected — same fragmentation; `supersedes` already covers per-fact causal lineage |

**Provider strategy rationale (transactional selected).** Strongest production architecture that actually protects required invariants (atomic accept, stable idempotent retry, no half-accepted revision, crash-safe classification, contiguous finalized watermark) **without** an extra sequencer that creates allocation-without-acceptance gaps by default. Vendor-neutral: strategy ≠ PostgreSQL/SQLite type in the public contract.

| Alternative | Decision |
|-------------|----------|
| Dedicated sequencer | Rejected as canonical default — extra failure modes (F) without stronger uniqueness/monotonicity than a transactional boundary |
| Distributed / scoped sequencer | Rejected as default — extra coordination; valid later as a **qualified alternative** behind the same ABC |
| CAS / optimistic allocation | Rejected as default — weaker atomic accept+allocate; contention and crash windows harder to classify |
| Reuse `ExecutionEventPosition` / `IdempotencyStore` / `SystemTimeProvider` / context CAS | Rejected — different semantic roles |

**Failure matrix (canonical semantic outcomes):**

| | Expected state | Watermark | Retry | Audit |
|---|----------------|-----------|-------|-------|
| **A** Position allocated; acceptance succeeds | `ACCEPTED` at `K` | May include `K` once all `<= K` are finalized | n/a | Accepted revision at `K` |
| **B** Position allocated; acceptance rolls back | `TERMINAL_NON_COMMITTED` at that `K` (or never-visible if the allocator rolled back without consuming `K`) | Terminal gap does not block | New logical op gets a new key / new `K` | Classifiable terminal (not missing) if `K` was consumed |
| **C** Crash before acceptance outcome known | `UNRESOLVED` until classified | Must not pass this `K` | Recovery classifies to `ACCEPTED` or `TERMINAL_NON_COMMITTED` | Explicit unresolved |
| **D** Commit succeeds; caller times out | `ACCEPTED` at `K` | May include `K` | See **E** | Accepted; caller timeout is not a second revision |
| **E** Retry after **D** | Same `ACCEPTED` `K` | Unchanged | Same `revision_id` + same `RevisionAcceptanceKey` → same `K` | No duplicate accepted revision |
| **F** Sequencer/allocator issued `K`; durable write fails | `TERMINAL_NON_COMMITTED` or `UNRESOLVED` | Unresolved blocks; terminal does not | Same key must not accept a different `revision_id` or different `K` | No accepted revision at failed `K` |
| **G** Restart with unresolved positions | Each remains `UNRESOLVED` until classified | Cannot advance past lowest unresolved | Recovery/classification required | Unresolved list is queryable |
| **H** Duplicate acceptance concurrent | One accepted revision; same `K` | Same as single accept | Same key collapses to one `K` | One audit row |
| **I** Two distinct revisions concurrent | Distinct `K` values, deterministic tenant order | Advances only through finalized prefix | Independent keys | Auditable K1 → K2 order without timestamps |
| **J** Terminal non-committed gap below later accepted revisions | Lower `K` stays `TERMINAL_NON_COMMITTED`; later `ACCEPTED` | Watermark **may** advance across the gap | n/a | Gap is classifiable, not invisible |

**TRACE-BITEMP-2 boundary:** **Planned / In Review** — implemented slice: `CanonicalRevisionOrderingProvider` + `RevisionOrderingSQLiteStore` + `UnresolvedRevisionRecovery` + `open_revision_ordering_authority`. Atomic linearization via SQLite `BEGIN IMMEDIATE` transactions coordinating acceptance bindings, position lifecycle, and per-tenant `RevisionFencingGeneration`. Canonical acceptance requires `canonical_accepted=1` on `knowledge_position_states` — physical payload rows in `knowledge_physical_payloads` are quarantined and never promoted to `ACCEPTED` by presence alone. Known limitations: alternate providers not qualified (TRACE-BITEMP-5); K-only historical knowledge reconstruction **Done** (TRACE-BITEMP-3); temporal query/audit surface not implemented (TRACE-BITEMP-4); execution-as-of query surface not implemented (TRACE-ASOF-4).

### 8.12 TRACE-BITEMP-2 delivered implementation mapping

| Area | Delivered type / path |
|------|------------------------|
| Canonical provider | `intergrax.runtime.observability.canonical_revision_ordering_provider.CanonicalRevisionOrderingProvider` |
| Durable store | `intergrax.runtime.observability.revision_ordering_store.RevisionOrderingSQLiteStore` |
| Recovery | `intergrax.runtime.observability.unresolved_revision_recovery.UnresolvedRevisionRecovery` |
| Host DI | `intergrax.runtime.observability.composition.open_revision_ordering_authority` (`INTERGRAX_REVISION_ORDERING_DB`) |
| Fencing | `RevisionFencingGeneration` per tenant scope; `ResolutionAuthority` from `acquire_resolution_authority`; `writer_fencing_generation` preserves original writer authority; `canonical_fencing_generation` records winning canonical lifecycle authority (acceptance or terminalization) |
| Resolution API | `resolve_unresolved_position` → `KnowledgeRevisionResolutionRecord`; terminalization persists `canonical_fencing_generation` = recovery authority generation |
| Orphan evidence | `OrphanedDurableRevisionRecord` + `knowledge_orphan_records` / quarantined `knowledge_physical_payloads`; requires modeled physical durability — stale canonical acceptance rejection alone does **not** create orphan evidence |
| Watermark | `compute_finalized_watermark` over durable `knowledge_position_states` |
| Revision reference | `KnowledgeRevisionId` bound in `knowledge_acceptance_bindings` — no untyped knowledge payload bucket |

**TRACE-BITEMP-2 boundary (requirements):** implement `RevisionOrderingAuthority` with the selected strategy; persist lifecycle; authoritative resolution path (`UNRESOLVED → ACCEPTED` / `TERMINAL_NON_COMMITTED`); lease/ownership and fencing where required; bounded unresolved scanner/recovery; immutable resolution records; authoritative acceptance/finalization linearization primitive; validate current fencing generation at canonical acceptance; detect/quarantine/isolate orphaned durable writes where physical stale-commit prevention is impossible; persist immutable orphan/integrity evidence; advance watermark; recovery of unresolved positions; stale-writer rejection; distinguish committed-and-canonically-accepted vs unresolved vs terminal vs orphaned physical residue; idempotent recovery; manual/governance fallback through same authority; host DI. **MUST NOT** infer `ACCEPTED` from physical storage presence alone; **MUST NOT** resurrect `TERMINAL_NON_COMMITTED`; **MUST NOT** allocate new **K** for lifecycle voiding or orphan detection; **MUST NOT** invent types, change TENANT scope, weaken finalized-contiguous semantics, add valid/system time onto `RuntimeEvent`, or select a vendor type as the public contract.

### 8.13 TRACE-BITEMP-3 K-only reconstruction (delivered) and downstream ownership

**TRACE-BITEMP-3** — **Done / Closed** (`5c2eedca75fc32101ea7a35e332c2abb3af24985`). Provider-independent deterministic reconstruction of canonical accepted knowledge at finalized **`KnowledgeRevisionWatermark K`**.

```text
finalized KnowledgeRevisionWatermark K
        ↓
complete finalized prefix 1..K (RevisionOrderingAuthority.records_through)
        ↓
canonical ACCEPTED K → KnowledgeRevisionId
        ↓
typed KnowledgeRevisionReader
        ↓
pure deterministic reducer (knowledge_reconstruction.py)
        ↓
immutable HistoricalKnowledgeProjection + typed K → revision provenance
```

Question answered: **What canonical knowledge state resulted from accepted revisions exactly at K?**

Closure does **not** yet deliver — downstream ownership; **not** unresolved TRACE-BITEMP-3 gaps:

| Capability | Downstream owner |
|------------|------------------|
| `ValidTimeBasis` filtering/selection | TRACE-BITEMP-4 |
| `SystemTimeBasis` filtering/selection | TRACE-BITEMP-4 |
| wall-clock **T → finalized K** resolution | TRACE-BITEMP-4 |
| combined **E + K** projection | TRACE-BITEMP-4 |
| combined **E + K + Valid Time + System Time** query | TRACE-BITEMP-4 |
| public temporal/audit API | TRACE-BITEMP-4 |
| execution-as-of **query contract** at boundary **E** | TRACE-ASOF-4 |

**TRACE-ASOF-4** (planned) owns the historical **execution-as-of query contract** at boundary **E** — **What was execution state at E?** plus provenance to execution events. It does **not** own full **E + K + Valid Time + System Time** semantics.

**TRACE-BITEMP-4** (planned) owns temporal knowledge query/audit:

- **ValidTimeBasis** selection/filtering — when a fact was effective in the modeled domain
- **SystemTimeBasis** selection/filtering — when Intergrax knew/recorded a version
- bitemporal selection — **Valid Time + System Time** only (**K** is **not** a third temporal axis)
- wall-clock **T → finalized K** before reconstruction/query when deterministic ordering is required (timestamp does **not** replace **K**)
- combined historical query/audit composition — **E + K + ValidTimeBasis + SystemTimeBasis** → **Historically Reproducible Execution State** where the question requires it

Read-side delivery ownership:

```text
TRACE-BITEMP-3  → stable K-only reconstruction                    → CLOSED
TRACE-ASOF-4    → execution-as-of query at E
TRACE-BITEMP-4  → temporal knowledge query/audit
                  Valid Time + System Time · T→K · optional E+K composition
```

Execution ordering **E** ≠ knowledge ordering **K** ≠ Valid Time ≠ System Time. Do **not** name the combined result “bitemporal execution state”.

---

## 9. Semantic separation of observability artifacts

| Artifact | Role |
|----------|------|
| **`RuntimeEvent`** | Canonical fact that something happened |
| **Unified Run Journal** | Chronological derived execution timeline |
| **As-Of Projection** | Derived execution state at a deterministic journal boundary |
| **Valid time** (`ValidTimeBasis`) | When a fact is effective in the modeled domain |
| **System time** (`SystemTimeBasis`) | When Intergrax recorded / knew a fact version |
| **Bitemporal state** (`BitemporalKnowledgeBasis`) | State selected using valid-time + system-time basis only |
| **Knowledge/revision ordering** | Deterministic authoritative ordering of accepted corrections/revisions — **not** a bitemporal axis; **not** execution ordering |
| **RevisionOrderingAuthority** | Domain-owned semantic contract for authoritative revision ordering; host/DI selects provider; semantics **not** per-application configurable |
| **CanonicalRevisionOrderingProvider** | Intergrax first-party default implementing `RevisionOrderingAuthority`; strategy = tenant-scoped transactional allocation + acceptance (TRACE-BITEMP-2 implements) |
| **KnowledgeRevisionWatermark** | Stable authoritative **finalized contiguous** upper bound in knowledge/revision ordering; not highest allocated; **not** a temporal axis |
| **`HistoricalKnowledgeProjection`** | Immutable K-only knowledge reconstruction output (`reconstruct_knowledge_at_watermark`) — TRACE-BITEMP-3 **Done / Closed** |
| **Historically reproducible execution state** | Combined reconstruction: execution boundary E + knowledge watermark K + bitemporal knowledge basis — **not** “bitemporal state”; owned by TRACE-BITEMP-4 query/audit composition |
| **Provenance** | Origin / lineage of relevant inputs and references |
| **Evidence** | Persisted supporting evidence |
| **Proof / Receipt** | Attested / verifiable claim |

Projection and bitemporal state are read-side historical reconstruction — not proof, not evidence, not a substitute for the event store.

---

## 10. Execution Story relationship

As-of projections and bitemporal historical state are part of the **read side** of Execution Story — not new runtime domains. No new Execution Story domain or event store is introduced by TRACE-ARCH-SYNC-1 or TRACE-BITEMP-ARCH-SYNC.

```text
RuntimeEvent history
       ↓
Execution AsOfBoundary E               Bitemporal fact / revision history
       │                                          ↓
       │                              Knowledge/revision ordering (K1 → K2 → K3)
       │                                          ↓
       │                              KnowledgeRevisionWatermark K
       │                                          ↓
       │                              Valid-Time Basis + System-Time Basis
       │                                          ↓
       ├── as-of execution reconstruction (§7)    └── bitemporal knowledge reconstruction (§8)
       │
       └── Unified Run Journal → Execution Story (chronological narrative)
```

Execution ordering and knowledge/revision ordering are **independent**. A correction accepted at **K20** after **E42** does **not** rewrite E42's execution sequence and is **not** retroactively inserted into E42 execution history.

Wall-clock audit questions that require deterministic knowledge ordering resolve **T → K** first, then optionally combine with **E**. Wall-clock time does **not** replace revision ordering.

Combined historical execution reconstruction (not “bitemporal state”):

```text
Execution AsOfBoundary E
+ KnowledgeRevisionWatermark K
+ BitemporalKnowledgeBasis (valid time + system time)
       ↓
Historically Reproducible Execution State
```

**Roadmap delivery ownership (temporal capabilities preserved downstream):**

```text
TRACE-BITEMP-3  → K-only reconstruction at finalized K           → CLOSED (5c2eedca...)
TRACE-ASOF-4    → execution-as-of query at E
TRACE-BITEMP-4  → temporal knowledge query/audit
                  Valid Time + System Time · T→K · optional E+K composition
```

**E** ordering ≠ **K** ordering ≠ Valid Time ≠ System Time. Combined reconstruction is **Historically Reproducible Execution State** — not “bitemporal state” and not “bitemporal execution state”.

---
