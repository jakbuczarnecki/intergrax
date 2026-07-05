# Observability

**Status:** Canonical architecture (domain pair 1:1)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Plan (1:1):** [`plan/OBSERVABILITY.md`](../plan/OBSERVABILITY.md)  
**Target:** [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../guides/IDEAL_HARNESS_AI_ARCHITECTURE.md)  
**Audit layers:** 21, 30  
**Audit instruction:** [`audit/OBSERVABILITY.md`](../audit/OBSERVABILITY.md)  
**Last updated:** 2026-07-05 — **OBS-PROBLEM-3** problem signal emission boundary canon

---

## Cursor read scope (token budget)

**Do not read this entire file in one session** (OBSERVABILITY canon).

- **Implement / audit default:** trace spine + HOS + signal planes (§1–§4). Extended §5+: [`satellites/OBSERVABILITY_extended_depth.md`](satellites/OBSERVABILITY_extended_depth.md).
- **Use** table of contents below — `Read` with offset/limit per §.
- **Plan hub:** [`plan/OBSERVABILITY.md`](../plan/OBSERVABILITY.md) (scoped §6 only).
- **Audit slice:** [`guides/audit_slices/OBSERVABILITY.md`](../guides/audit_slices/OBSERVABILITY.md).
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
| **Correlation by construction** | `task_id`, `run_id`, `correlation_id`, `parent_event_id` are set by the spine — not passed manually in business code. |
| **Redact before persist** | `DiagnosticPayload.redact()` + `production_mode` run before any store append. |
| **Pluggable persistence** | SQLite default; Cassandra/Elasticsearch/OTLP as integration profiles — same API, different backend. |
| **Read-model unification** | Operators consume **one chronological journal** per run (`build_unified_run_journal`). |
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

**Implementation detail:** Plane A/B/C breakdown, field catalog, and bridge mechanics — §4. Correlation identifiers — §6 and [Required correlation fields](#required-correlation-fields) below. Layered `event_type` / `event_kind` governance — §4.4 and [Event type governance](#event-type-governance) below.

**Cross-layer canon:** [`SYSTEM_INVARIANTS.md`](../guides/SYSTEM_INVARIANTS.md) §7 · [`UNIFIED_EXECUTION_RUNTIME.md`](UNIFIED_EXECUTION_RUNTIME.md) §42.1 · [`NEXUS_EXECUTION_FLOW.md`](NEXUS_EXECUTION_FLOW.md) §12.2 · [`RELIABILITY_FAILURE_AND_HITL.md`](RELIABILITY_FAILURE_AND_HITL.md#attempt-ledger) · [`TOOLS.md`](TOOLS.md) · [`INTEGRATIONS.md`](INTEGRATIONS.md) · [`AGENT_CONTRACTS_AND_ASSEMBLY.md`](AGENT_CONTRACTS_AND_ASSEMBLY.md) §31 · [`CRITIC_VERIFICATION.md`](CRITIC_VERIFICATION.md#boundary-with-observability--evaluation-control-plane-oecp) · [`ADAPTIVE_HARNESS_INTELLIGENCE.md`](ADAPTIVE_HARNESS_INTELLIGENCE.md#governance-boundary) · [`ELASTIC_CAPACITY_AND_SCALING.md`](ELASTIC_CAPACITY_AND_SCALING.md#scaling-action-governance) · [`CODE_CRAFT.md`](CODE_CRAFT.md#codecraft-safety-boundary)

---

## Observability & Evaluation Control Plane

Intergrax observability is **not** limited to traces and metrics. The **Harness Observability Spine (HOS)** remains the **only** canonical observability spine. **Observability & Evaluation Control Plane (OECP)** operates **above** HOS — it consumes `RuntimeEvent`, `TraceEvent`, unified journal, and evidence refs; it **must not** create a parallel trace system.

OECP transforms spine data into eval-grade artifacts: **evidence ledger** records, **eval snapshots**, **metric results**, **regression gates**, and **perturbation suites**. External workbenches (Langfuse, LangSmith, OTLP, Sentry, Phoenix, Braintrust, Datadog, …) are optional sinks — not semantic owners.

**Target architecture:** [`satellites/OBSERVABILITY_extended_depth.md`](satellites/OBSERVABILITY_extended_depth.md) (OECP sections). **Plan:** [`plan/satellites/OBSERVABILITY_eval_control_plane.md`](../plan/satellites/OBSERVABILITY_eval_control_plane.md). **Audit source:** [`audit/OBSERVABILITY_EVALUATION_CONTROL_PLANE_AUDIT.md`](../audit/OBSERVABILITY_EVALUATION_CONTROL_PLANE_AUDIT.md).

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

**Code references:** `problem_signal.py` · `problem_export.py` · `problem_reporter.py` · `export_boundary.py` · `export_policy.py`. **Plan:** OBS-PROBLEM-3 in [`plan/OBSERVABILITY.md`](../plan/OBSERVABILITY.md).

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

Platform contract: `FanoutObservabilityExporter` + `ObservabilityExportRoute` (`export_routing.py`).

### E. Vendor boundary

Sentry, Elastic, OTLP, Langfuse, and similar backends are **future provider projections** that receive policy-safe envelopes from configured route exporters. Runtime, application, agent, tool, and LKW code **MUST NOT** choose vendor destinations directly.

### F. Out of scope (this boundary)

- No Sentry, Elastic, or OTLP provider implementation.
- No runtime automatic problem emission.
- No `ObservabilityEmitter.emit_problem`.
- No `RuntimeEventBus` subscriber for problems.
- No LKW endpoint wiring or operator bootstrap config.

**Code references:** `export_routing.py` · `export_boundary.py` · `export_policy.py`. **Plan:** OBS-ROUTING-0 in [`plan/OBSERVABILITY.md`](../plan/OBSERVABILITY.md).

---

## Required correlation fields

Meaningful runtime events **SHOULD** preserve all correlation identifiers available at the emission boundary:

| Field | Purpose |
|-------|---------|
| `task_id` | User-facing work unit |
| `run_id` | Single execution attempt / trace timeline |
| `node_id` | Graph node placement |
| `agent_id` | Responsible agent |
| `step_id` | UAEP / pipeline step |
| `tool_call_id` | Tool invocation chain (when applicable) |
| `correlation_id` | Cross-agent/tool chain (default: `task_id`) |
| `parent_event_id` | Causal parent in the spine tree |
| `event_id` | Unique event identity |
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
| `task_id` | Logical work unit (user request scope) |
| `run_id` | Single execution attempt (retries → new run or branch per policy) |
| `correlation_id` | Cross-agent/tool chain (default: `task_id`) |
| `parent_event_id` | Causal parent in the spine tree (**target:** populated by `TraceScope`) |
| `node_id` / `agent_id` / `step_id` | Graph and UAEP placement |
| `payload` | Structured facts (**today:** `dict`; **target:** typed `RuntimeEventPayload`) |
| `schema_version` | Envelope version (`runtime_event.v1`) |

**Code:** `intergrax/runtime/events/runtime_event.py`, `phase_coverage.py`, `event_bus.py`

**Catalog:** **56** `RuntimeEventType` spine members (publication budget; OBS-EVOL-9.7). Platform adaptive/capacity/hook/recovery signals emit on `DOMAIN_SIGNAL` + `platform.*` `event_kind` — see §4.4.13.

### 4.4 Layered event identity (P1-ARCH-02 · OBS-EVOL-9)

**Status:** Architecture **accepted** (2026-06-17) · implementation **Done** (OBS-EVOL-9 register) · **ADR:** [`ADR-OBS-003`](../adr/entries/2026-06-17/ADR-OBS-003.md) · **SAR:** accepted 2026-06-17 (§4.4.7–4.4.13)

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

All public emit APIs accept a typed **`EmitContext`** carrying `task_id`, `run_id`, `tenant_id`, `correlation_id`, and active `TraceScope` — correlation by construction (SAR-01).

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
