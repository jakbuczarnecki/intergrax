<!--
© Artur Czarnecki. All rights reserved.
Intergrax framework – proprietary and confidential.
Use, modification, or distribution without written permission is prohibited.
-->

# Observability & Evaluation Control Plane — Architecture Audit

**Status:** Audit source document for architecture and implementation-plan updates  
**Domain:** `OBSERVABILITY` with cross-domain impact on `CRITIC_VERIFICATION`, `ADAPTIVE_HARNESS_INTELLIGENCE`, `EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE`, and `TIER3_APPLICATION_ENVIRONMENT`  
**Target architecture docs:** `docs/project/architecture/OBSERVABILITY.md`, `docs/project/architecture/satellites/OBSERVABILITY_extended_depth.md`
**Target plan docs:** `docs/project/maintainers/plans/OBSERVABILITY.md`, proposed `docs/project/maintainers/plans/satellites/OBSERVABILITY_eval_control_plane.md`
**Date:** 2026-06-24  

---

## 1. Purpose

This document captures the audit result for extending Intergrax observability from a strong Harness Observability Spine into a full evaluation-grade observability platform.

The document is intentionally not the final architecture canon. It is a source document that Cursor or another implementation agent MUST use to update architecture and implementation plans according to the existing Intergrax documentation governance model.

The goal is to prevent the audit from remaining chat-only knowledge and to ensure the resulting architectural changes are deliberate, scoped, traceable, and implementable.

---

## 2. Executive verdict

Intergrax already has a strong observability foundation. The current architecture includes a canonical Harness Observability Spine, typed runtime and diagnostic payloads, correlation and causal tracing, extension SDK support, persistence/export surfaces, and early evaluation primitives.

The main gap is no longer basic observability. The main gap is the absence of a full Observability & Evaluation Control Plane that turns trace data into continuous, automated, evidence-backed quality measurement.

The recommended direction is:

```text
run -> trace -> evidence ledger -> eval snapshot -> metric results -> regression gates -> perturbation suites -> controlled adaptation
```

Intergrax should become capable of answering not only:

```text
What happened during this run?
```

but also:

```text
Was the run correct?
Why was it correct or incorrect?
Which evidence supports that judgment?
Did the result regress compared to the previous version?
Would it fail under a small counterfactual prompt change?
Does it fail more often for a specific device, tenant, tool catalog, RAG collection, or runtime environment?
Can the developer add custom telemetry and custom evals without bypassing the canonical spine?
```

---

## 3. Current foundation — confirmed strengths

### 3.1 Canonical Harness Observability Spine

The existing architecture correctly treats the Harness Observability Spine as the single canonical observability layer. Runtime events, diagnostic trace events, metrics, persistence, unified journal, and external sinks are separated by responsibility. This must remain unchanged.

Preserve these rules:

- HOS is the canonical execution record.
- External tools are sinks, dashboards, or workbenches, not semantic owners.
- Agents and applications must not create private trace pipelines.
- Tool, RAG, LLM, memory, policy, critic, and graph events must be reconstructable from the spine.
- Redaction must happen before persistence and export.
- Runtime and diagnostic payloads must be typed and versioned.

### 3.2 Typed diagnostic and runtime payloads

The existing `DiagnosticPayload` and `RuntimeEventPayload` direction is correct. It should be extended, not replaced.

The current extension SDK already supports:

- agent diagnostic schema IDs;
- application diagnostic schema IDs;
- runtime extension schema registration;
- optional event kind registration;
- namespace enforcement for custom diagnostics and signals.

This is the correct basis for custom telemetry and custom evaluation evidence.

### 3.3 Correlation and causal reconstruction

The existing TraceScope, run/task/tenant correlation, parent event linkage, and unified journal direction are strategically correct. The next layer should build on these fields rather than introduce new execution identifiers.

Every new evaluation or telemetry object should carry enough references to connect back to:

- tenant_id;
- task_id;
- run_id;
- agent_id;
- step_id;
- tool_call_id where applicable;
- correlation_id;
- parent_event_id or evidence_refs where applicable.

### 3.4 Evaluation primitives already exist

Intergrax already has early evaluation primitives such as judge-based evaluation, trajectory evaluation, evaluation profiles, critic profiles, and online evaluation registry concepts.

These should be treated as the seed of a larger control plane, not as the final form.

---

## 4. Main architectural gap

The system has strong tracing, but it does not yet have a complete eval-first observability lifecycle.

The missing layer is:

```text
Observability & Evaluation Control Plane
```

This layer should sit above HOS and consume HOS data. It must not create a parallel observability stack.

The control plane should provide:

- trace completeness verification;
- eval-grade evidence ledger;
- evaluation registry v2;
- metric and eval plugin SDK;
- custom telemetry extension plane;
- counterfactual mutation engine;
- interpolation test generation;
- regression gates;
- external workbench synchronization;
- developer-facing run/eval/debug views.

---

## 5. Non-negotiable architectural principles

### 5.1 HOS remains the canonical source

The new control plane MUST consume HOS events and unified journal records. It MUST NOT introduce a second source of truth for execution traces.

Correct:

```text
HOS -> EvidenceLedger -> EvalRegistry -> RegressionGate
```

Incorrect:

```text
Agent -> custom logger -> private eval database -> separate dashboard
```

### 5.2 Evidence before judgment

No eval result should exist without evidence references. Every metric result, judge score, regression decision, or critic verdict should be traceable to concrete evidence.

Examples of evidence:

- final assembled prompt;
- final model input;
- model output;
- tool catalog snapshot;
- tool selection candidates;
- tool selection decision;
- tool invocation result;
- RAG query;
- retrieved chunks;
- context pack;
- memory reads/writes;
- policy decision;
- critic verdict;
- custom telemetry payloads.

### 5.3 Typed and versioned extension, not raw dictionaries

Custom developer data must enter the system through typed contracts, not raw ad-hoc dictionaries.

Required properties:

- schema_id;
- versioning convention;
- namespace ownership;
- redaction behavior;
- tenant awareness;
- correlation with run/task;
- export policy;
- retention class;
- sampling policy where needed.

### 5.4 External tools are optional sinks/workbenches

Langfuse, LangSmith, OpenTelemetry, Sentry, Phoenix, Braintrust, Datadog, ClickHouse, or any similar tool must be treated as optional integrations.

They may store, visualize, compare, or enrich observability data, but they must not define Intergrax semantics.

### 5.5 Evaluation is continuous, not only post-mortem

Intergrax should support evaluation at multiple moments:

- during a run;
- immediately after a run;
- during production sampling;
- during nightly regression;
- before release;
- during canary rollout;
- after an incident;
- when a prompt, model, tool, RAG collection, or agent profile changes.

---

## 6. Target architecture

Recommended conceptual structure:

```text
Developer UX / Debug Workbench
        |
Observability & Evaluation Control Plane
        |
        |-- Trace Completeness Contract
        |-- Evidence Ledger
        |-- Eval Registry v2
        |-- Metric and Eval Plugin SDK
        |-- Custom Telemetry Extension Plane
        |-- Counterfactual and Interpolation Engine
        |-- Regression Gate
        |-- External Workbench Sync
        |
Harness Observability Spine
        |
        |-- RuntimeEvent
        |-- TraceEvent
        |-- DiagnosticPayload
        |-- RuntimeEventPayload
        |-- Unified Journal
        |-- Metrics Export
        |
Nexus / Agents / Tools / RAG / Memory / Policy / Critic
```

---

## 7. Evidence Ledger

### 7.1 Purpose

The Evidence Ledger is the eval-ready layer derived from HOS. It stores structured evidence required to understand and evaluate a run.

It should not duplicate the whole trace. It should create normalized evidence records that point back to canonical trace or journal entries.

### 7.2 Proposed evidence kinds

```text
intergrax.prompt.assembly.built.v1
intergrax.model.input.finalized.v1
intergrax.model.output.received.v1
intergrax.tool.catalog.snapshot.v1
intergrax.tool.selection.candidates.v1
intergrax.tool.selection.decision.v1
intergrax.tool.invocation.result.v1
intergrax.rag.query.built.v1
intergrax.rag.retrieval.completed.v1
intergrax.context.pack.built.v1
intergrax.memory.context.used.v1
intergrax.policy.decision.recorded.v1
intergrax.critic.verdict.recorded.v1
intergrax.claim.evidence.linked.v1
intergrax.iteration.completed.v1
intergrax.custom.telemetry.recorded.v1
```

### 7.3 Evidence record requirements

Every evidence record should include:

```text
evidence_id
evidence_kind
schema_id
schema_version
tenant_id
task_id
run_id
agent_id optional
step_id optional
tool_call_id optional
correlation_id optional
source_event_id optional
source_trace_event_id optional
payload
redaction_state
retention_class
created_at
```

---

## 8. Trace Completeness Contract

### 8.1 Purpose

The Trace Completeness Contract defines what a run must contain to be considered eval-grade.

A run may be operationally successful but not eval-grade if it lacks key evidence.

### 8.2 Example completeness dimensions

```text
input captured
prompt assembly captured
model input captured or safely redacted
model output captured or safely redacted
tool catalog snapshot captured
tool selection decision captured
RAG query and retrieval captured when RAG is used
context pack captured or summarized
policy decision captured when policy gates are active
critic verdict captured when critic is enabled
custom telemetry providers executed when configured
cost/latency captured
stop reason captured
failure/retry path captured
```

### 8.3 Proposed components

```text
TraceCompletenessProfile
TraceCompletenessChecker
TraceCompletenessReport
TraceCompletenessFinding
TraceCompletenessGate
```

### 8.4 Gate behavior

Trace completeness should support multiple modes:

```text
observe
warn
block_release
block_canary_promotion
fail_ci
```

---

## 9. Eval Registry v2

### 9.1 Purpose

The current evaluation registry should evolve into a versioned registry that can represent datasets, cases, runs, metric results, evidence references, and regression decisions.

### 9.2 Proposed core models

```text
EvalCase
EvalDataset
EvalRun
EvalRunSnapshot
EvalMetricSpec
EvalMetricResult
EvalObservationV2
EvalRegressionResult
EvalPerturbationSpec
EvalEvidenceRef
```

### 9.3 EvalObservationV2 fields

```text
observation_id
tenant_id
task_id
run_id
agent_id
application_id
scenario_id
dataset_id
case_id
case_origin
metric_id
metric_family
metric_version
score
passed
threshold
severity
baseline_score
delta
prompt_version
agent_profile_version
model_profile_version
tool_catalog_version
rag_collection_version
evidence_refs
trace_refs
failure_taxonomy
recommended_action
recorded_at
```

### 9.4 Case origins

```text
manual
production_sample
incident_harvested
critic_failure
regression_failure
counterfactual_generated
interpolation_generated
external_benchmark
```

---

## 10. Metric and Eval Plugin SDK

### 10.1 Purpose

Intergrax should allow developers to add custom metrics without changing the core runtime.

### 10.2 Proposed contract

```python
class EvalMetricPlugin(Protocol):
    metric_id: str
    family: str
    version: str

    def supports(self, case: EvalCase, run: EvalRunSnapshot) -> bool:
        ...

    def score(self, case: EvalCase, run: EvalRunSnapshot) -> EvalMetricResult:
        ...
```

### 10.3 Metric families

```text
deterministic
lexical
embedding
llm_judge
rag
agentic
ops
custom_business
custom_telemetry
```

### 10.4 Baseline built-in metrics

Recommended first built-ins:

```text
exact_match
contains_required_terms
json_schema_validity
trajectory_tool_error_count
trajectory_duplicate_tool_call_count
cost_total
latency_total
rag_context_precision_stub
rag_answer_groundedness_stub
critic_score
```

BLEU, BERTScore, embedding distance, semantic similarity, and judge-based metrics should be plugin families. They should not be hardcoded as core runtime semantics.

---

## 11. Custom Telemetry Extension Plane

### 11.1 Purpose

Developers must be able to plug custom observability data into Intergrax through controlled contracts.

This is strategically important. A production harness cannot know every business-specific or environment-specific signal in advance.

Examples:

```text
IP address hash
device type
browser family
frontend app version
memory usage
CPU usage
GPU usage
container ID
tenant plan
customer segment
business object ID
document class
workflow ID
risk class
custom domain score
```

### 11.2 Key rule

Custom telemetry must extend HOS. It must not bypass HOS.

Correct:

```text
CustomTelemetryProvider -> DiagnosticPayload/RuntimeEventPayload -> HOS -> Journal -> EvidenceLedger -> EvalRegistry
```

Incorrect:

```text
CustomTelemetryProvider -> private logger -> private database -> private dashboard
```

### 11.3 Proposed TelemetryProvider contract

```python
class TelemetryProvider(Protocol):
    provider_id: str
    schema_id: str

    def collect(self, context: TelemetryCollectionContext) -> DiagnosticPayload | RuntimeEventPayload | None:
        ...
```

### 11.4 Proposed TelemetryCollectionContext

```text
tenant_id
task_id
run_id
agent_id optional
application_id optional
step_id optional
request_metadata optional
runtime_metadata optional
profile_metadata optional
trace_scope
production_mode
redaction_policy
```

### 11.5 Proposed TelemetryEnricher contract

```python
class TelemetryEnricher(Protocol):
    enricher_id: str

    def enrich(self, event: RuntimeEvent, context: TelemetryEnrichmentContext) -> RuntimeEvent:
        ...
```

### 11.6 Proposed EventSubscriptionHandler contract

```python
class EventSubscriptionHandler(Protocol):
    handler_id: str

    def handle(self, event: RuntimeEvent, context: EventHandlingContext) -> None:
        ...
```

### 11.7 Example reactions

```text
When TOOL_FAILED -> collect memory snapshot.
When LLM_CALL exceeds cost threshold -> record cost anomaly.
When RAG score is low -> record retrieval quality signal.
When TASK_COMPLETED has low critic score -> create eval candidate.
When GUARDRAIL_BLOCKED -> export event to external sink.
```

### 11.8 Required safeguards

Custom telemetry must enforce:

```text
redaction
PII classification
production-mode masking
tenant isolation
retention class
export allow/deny policy
sampling policy
high-cardinality guard
schema versioning
```

Raw IP addresses, full user-agent strings, device fingerprints, emails, customer names, or business secrets should not be emitted by default. Prefer hashed, bucketed, or classified representations.

Examples:

```text
ip_address_hash instead of raw IP
geo_region instead of precise location
device_type instead of fingerprint
memory_bucket or journal-only memory value instead of Prometheus labels
```

---

## 12. Counterfactual and Interpolation Engine

### 12.1 Purpose

Intergrax should automatically generate robustness tests from existing evaluation cases and production traces.

The goal is to detect fragile agents that pass the original prompt but fail under small semantic changes, ambiguity, entity swaps, or constraint negation.

### 12.2 Counterfactual operations

```text
replace_word
swap_entity
negate_constraint
remove_constraint
add_constraint
change_date
change_location
change_user_role
change_tool_availability
change_rag_collection_version
```

### 12.3 Interpolation operations

```text
merge_two_cases_as_ambiguous_request
combine_success_case_with_failure_case
combine_two_user_intents
blend_two_domain constraints
```

### 12.4 Required lineage

Every generated case must preserve:

```text
parent_case_id
mutation_id
mutation_type
mutation_description
original_input_ref
expected_behavior_delta
created_by
created_at
```

---

## 13. External Observability Workbench Sync

### 13.1 Purpose

Intergrax should integrate deeply with external tools while preserving its own semantics.

External systems should receive exported traces, scores, events, and links, but Intergrax should remain the canonical owner of:

- run lineage;
- evidence meaning;
- eval results;
- regression decisions;
- policy and critic semantics.

### 13.2 Proposed exporters

```text
HOSJournalToOTLPExporter
HOSJournalToLangfuseExporter
HOSJournalToLangSmithExporter
EvalResultsToLangfuseScores
EvalResultsToLangSmithFeedback
VendorDeepLinkStore
ExportRetryQueue
ExportDeadLetterQueue
```

### 13.3 Export requirements

```text
non-blocking export
retry support
dead-letter queue
redaction before export
vendor deep links stored back in Intergrax
export policy per tenant/application/profile
clear distinction between canonical record and external copy
```

---

## 14. Developer Workbench requirements

The future debug/workbench layer should expose:

```text
Run Timeline
Prompt Assembly View
Model Input/Output View
Tool Catalog Snapshot View
Tool Selection Decision View
RAG Evidence View
Context Pack View
Policy Decision View
Critic Verdict View
Custom Telemetry View
Eval Score View
Regression Diff View
Counterfactual Fragility View
Cost/Latency Trend View
External Vendor Links View
```

This does not need to be implemented immediately, but architecture and plan should reserve the concept.

---

## 15. Cross-domain ownership

### 15.1 OBSERVABILITY

Owns:

```text
HOS
Evidence Ledger
Trace Completeness Contract
Custom Telemetry Extension Plane
External observability export
observability workbench concepts
```

### 15.2 CRITIC_VERIFICATION

Owns:

```text
per-run correctness verification
critic orchestration
L0/L1/L2 verification gates
critic verdict semantics
judge usage for verification
```

Relationship:

```text
CRITIC emits verdicts and observations.
OBSERVABILITY stores evidence and exposes continuous measurement infrastructure.
```

### 15.3 ADAPTIVE_HARNESS_INTELLIGENCE

Owns:

```text
controlled adaptation
profile promotion
canary/apply modes
learning from eval and telemetry results
```

Relationship:

```text
Adaptive Harness Intelligence may consume Eval Registry results, regression decisions, and telemetry trends.
It must not invent private evaluation records.
```

### 15.4 EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE

Owns:

```text
experiments UX
developer-facing comparison flows
experiment registry/workflows
bench and lab ergonomics
```

Relationship:

```text
Experiments may use Eval Registry datasets and metric plugins.
They should not duplicate Eval Registry or Evidence Ledger.
```

### 15.5 TIER3_APPLICATION_ENVIRONMENT

Owns:

```text
profile-level opt-in
application wiring
custom telemetry provider selection
custom metric plugin selection
vendor export profiles
application-level eval gates
```

Relationship:

```text
Tier-3 declares what is enabled.
Tier-0/Tier-1 provide the canonical mechanisms.
```

---

## 16. Recommended architecture document updates

### 16.1 `docs/project/architecture/OBSERVABILITY.md`

Add a short canonical summary section:

```text
Observability & Evaluation Control Plane
```

This section should state:

- Intergrax observability is not limited to traces and metrics.
- HOS is the canonical spine.
- The control plane derives eval-grade evidence from HOS.
- Evidence Ledger, Trace Completeness Contract, Eval Registry v2, Custom Telemetry Extension Plane, Counterfactual Engine, and External Workbench Sync are layered above HOS.
- No parallel trace system is allowed.

Keep this section concise. The hub should not become the full audit document.

### 16.2 `docs/project/architecture/satellites/OBSERVABILITY_extended_depth.md`

Add the full details as canonical target architecture sections:

```text
Observability & Evaluation Control Plane
Evidence Ledger
Trace Completeness Contract
Eval Registry v2
Metric and Eval Plugin SDK
Custom Telemetry Extension Plane
TelemetryProvider and TelemetryEnricher contracts
EventSubscriptionHandler reactions
Counterfactual and Interpolation Engine
External Observability Workbench Sync
Eval-grade maturity model L5-L7
```

### 16.3 `docs/project/architecture/CRITIC_VERIFICATION.md`

Add a cross-reference section that clarifies:

```text
CVL verifies individual outputs and trajectories.
OECP manages continuous eval datasets, metrics, regression gates, perturbations, and long-term measurement.
CVL emits verdicts; OECP stores, compares, evolves, and gates them.
```

### 16.4 `docs/project/architecture/TIER3_APPLICATION_ENVIRONMENT.md`

Add profile-level architecture notes for:

```text
custom_telemetry_providers
custom_telemetry_enrichers
custom_event_handlers
custom_eval_metric_plugins
eval_dataset_refs
eval_gate_profiles
counterfactual_profiles
vendor_export_profiles
```

Do not implement these fields unless a later implementation phase explicitly asks for code.

---

## 17. Recommended plan document updates

### 17.1 `docs/project/maintainers/plans/OBSERVABILITY.md`

Add a compact entry pointing to a new satellite:

```text
docs/project/maintainers/plans/satellites/OBSERVABILITY_eval_control_plane.md
```

The hub should remain compact.

### 17.2 New plan satellite

Create:

```text
docs/project/maintainers/plans/satellites/OBSERVABILITY_eval_control_plane.md
```

The satellite should contain the implementation register for the new work.

Recommended phases:

```text
OBS-ECP-0 Architecture canon
OBS-ECP-1 Trace Completeness Contract
OBS-ECP-2 Evidence Ledger
OBS-ECP-3 Eval Registry v2
OBS-ECP-4 Metric/Eval Plugin SDK
OBS-CTP-1 Custom Telemetry Extension Plane
OBS-CTP-2 Telemetry Provider / Enricher contracts
OBS-CTP-3 Event reaction handlers
OBS-PERT-1 Counterfactual Engine
OBS-PERT-2 Interpolation Engine
OBS-EXT-1 External Export Sync
OBS-GATE-1 CI and release gates
OBS-UX-1 Debug/workbench surfaces
```

---

## 18. Proposed first implementation register

### Phase OBS-ECP-0 — Architecture canon

| ID | Type | Deliverable | Acceptance |
|----|------|-------------|------------|
| OBS-ECP-0.1 | Docs | Update `OBSERVABILITY.md` with OECP summary | Hub states scope and no-parallel-trace rule |
| OBS-ECP-0.2 | Docs | Add extended OECP sections | Satellite contains canonical target architecture |
| OBS-ECP-0.3 | Docs | Add `CRITIC_VERIFICATION` cross-reference | Boundary between CVL and OECP is explicit |
| OBS-ECP-0.4 | Docs | Add Tier-3 profile notes | App-level opt-in surfaces are described |
| OBS-ECP-0.5 | Docs | Add plan satellite | New implementation register exists |

### Phase OBS-ECP-1 — Trace Completeness Contract

| ID | Type | Deliverable | Acceptance |
|----|------|-------------|------------|
| OBS-ECP-1.1 | Code | `TraceCompletenessProfile` | Supports observe/warn/block modes |
| OBS-ECP-1.2 | Code | `TraceCompletenessChecker` | Validates required run evidence |
| OBS-ECP-1.3 | Code | `TraceCompletenessReport` | Produces findings with evidence refs |
| OBS-ECP-1.4 | Tests | Completeness tests | Missing tool/RAG/prompt evidence detected |

### Phase OBS-ECP-2 — Evidence Ledger

| ID | Type | Deliverable | Acceptance |
|----|------|-------------|------------|
| OBS-ECP-2.1 | Code | Evidence record model | Includes source event refs and redaction metadata |
| OBS-ECP-2.2 | Code | Prompt/model evidence payloads | Prompt/model evidence can be recorded safely |
| OBS-ECP-2.3 | Code | Tool evidence payloads | Tool catalog, selection, invocation evidence captured |
| OBS-ECP-2.4 | Code | RAG/context evidence payloads | Retrieval and context pack evidence captured |
| OBS-ECP-2.5 | Code | Critic/custom telemetry evidence | Critic verdicts and custom telemetry link to evals |
| OBS-ECP-2.6 | Tests | Evidence extraction tests | Evidence can be derived from HOS/unified journal |

### Phase OBS-ECP-3 — Eval Registry v2

| ID | Type | Deliverable | Acceptance |
|----|------|-------------|------------|
| OBS-ECP-3.1 | Code | `EvalCase` and `EvalDataset` | Cases support origins and lineage |
| OBS-ECP-3.2 | Code | `EvalRun` and `EvalRunSnapshot` | Run snapshots reference evidence |
| OBS-ECP-3.3 | Code | `EvalMetricResult` and `EvalObservationV2` | Observations carry score, threshold, evidence refs |
| OBS-ECP-3.4 | Code | Regression result model | Baseline comparison is represented |
| OBS-ECP-3.5 | Tests | Registry v2 persistence tests | Observations persist and reload |

### Phase OBS-ECP-4 — Metric and Eval Plugin SDK

| ID | Type | Deliverable | Acceptance |
|----|------|-------------|------------|
| OBS-ECP-4.1 | Code | `EvalMetricPlugin` protocol | Plugins support case/run scoring |
| OBS-ECP-4.2 | Code | Plugin registry | Plugins registered by metric_id/family/version |
| OBS-ECP-4.3 | Code | Built-in deterministic metrics | exact match/schema/contains metrics available |
| OBS-ECP-4.4 | Code | Built-in trajectory/ops metrics | cost/latency/tool error metrics available |
| OBS-ECP-4.5 | Tests | Plugin SDK tests | Custom plugin can score and emit observation |

### Phase OBS-CTP-1 — Custom Telemetry Extension Plane

| ID | Type | Deliverable | Acceptance |
|----|------|-------------|------------|
| OBS-CTP-1.1 | Code | `TelemetryProvider` protocol | Provider returns typed payload only |
| OBS-CTP-1.2 | Code | `TelemetryCollectionContext` | Context includes tenant/run/task/profile/redaction |
| OBS-CTP-1.3 | Code | Provider registry | Providers register by provider_id/schema_id |
| OBS-CTP-1.4 | Code | Custom telemetry profile concept | Tier-3 can select providers |
| OBS-CTP-1.5 | Tests | Redaction and namespace tests | Raw unsafe payloads rejected or redacted |

### Phase OBS-CTP-2 — Telemetry enrichment and reaction

| ID | Type | Deliverable | Acceptance |
|----|------|-------------|------------|
| OBS-CTP-2.1 | Code | `TelemetryEnricher` protocol | Existing events can be enriched safely |
| OBS-CTP-2.2 | Code | `EventSubscriptionHandler` protocol | Handlers react to selected event kinds/categories |
| OBS-CTP-2.3 | Code | Memory/cost/anomaly sample handlers | Handlers emit typed follow-up events |
| OBS-CTP-2.4 | Tests | Handler tests | Handler cannot bypass HOS |

### Phase OBS-PERT-1 — Counterfactual Engine

| ID | Type | Deliverable | Acceptance |
|----|------|-------------|------------|
| OBS-PERT-1.1 | Code | `CounterfactualMutation` contract | Mutations preserve lineage |
| OBS-PERT-1.2 | Code | replace/swap/negate/remove mutations | Generated cases contain parent refs |
| OBS-PERT-1.3 | Tests | Mutation tests | Mutations are deterministic and traceable |

### Phase OBS-PERT-2 — Interpolation Engine

| ID | Type | Deliverable | Acceptance |
|----|------|-------------|------------|
| OBS-PERT-2.1 | Code | Interpolation generator | Combines cases into ambiguous requests |
| OBS-PERT-2.2 | Code | Expected behavior delta model | Generated case states expected difference |
| OBS-PERT-2.3 | Tests | Interpolation tests | Parent case lineage preserved |

### Phase OBS-EXT-1 — External export sync

| ID | Type | Deliverable | Acceptance |
|----|------|-------------|------------|
| OBS-EXT-1.1 | Code | OTLP journal exporter mapping | HOS journal exports without semantic loss |
| OBS-EXT-1.2 | Code | Langfuse export adapter | Trace/eval links exported safely |
| OBS-EXT-1.3 | Code | LangSmith export adapter | Trace/eval links exported safely |
| OBS-EXT-1.4 | Code | Export retry and DLQ | Export failure does not block runtime |
| OBS-EXT-1.5 | Tests | Export tests | Redaction and retry behavior verified |

### Phase OBS-GATE-1 — CI and release gates

| ID | Type | Deliverable | Acceptance |
|----|------|-------------|------------|
| OBS-GATE-1.1 | Code/CI | Trace completeness CI gate | Required evidence missing -> gate failure |
| OBS-GATE-1.2 | Code/CI | Eval regression gate | Baseline regression detected |
| OBS-GATE-1.3 | Code/CI | Plugin registry gate | Invalid custom schema/plugin rejected |

### Phase OBS-UX-1 — Debug/workbench surfaces

| ID | Type | Deliverable | Acceptance |
|----|------|-------------|------------|
| OBS-UX-1.1 | CLI/API | Run evidence view | Evidence records visible from run |
| OBS-UX-1.2 | CLI/API | Eval score view | Eval observations visible from run/case |
| OBS-UX-1.3 | CLI/API | Regression diff view | Baseline vs candidate shown |
| OBS-UX-1.4 | CLI/API | Custom telemetry view | Custom payloads queryable safely |

---

## 19. Anti-patterns Cursor must avoid

Do not:

- paste this audit directly into `OBSERVABILITY.md`;
- make the architecture hub too large;
- create a second trace database as the new source of truth;
- let evals read private logs instead of HOS/unified journal/evidence refs;
- add raw dictionary telemetry as an endorsed extension pattern;
- make Langfuse or LangSmith the canonical owner of Intergrax eval semantics;
- mix CRITIC_VERIFICATION ownership with OECP ownership;
- silently implement code while the task is documentation-only;
- mark the whole platform complete;
- start unrelated product dashboard work;
- add business-agent logic to Tier-0;
- bypass redaction or tenant isolation for custom telemetry.

---

## 20. Cursor update instruction

When using this audit as implementation input, Cursor should be instructed as follows:

```text
You are working in repository jakbuczarnecki/intergrax.
Branch: development.

Task type: documentation architecture and implementation-plan update only.
Do not implement runtime code.

Primary audit source:
docs/project/maintainers/audit/OBSERVABILITY_EVALUATION_CONTROL_PLANE_AUDIT.md

Goal:
Update the Intergrax architecture and implementation plan so the audit becomes canonical documentation and an actionable delivery plan.

Read first:
1. docs/project/maintainers/audit/OBSERVABILITY_EVALUATION_CONTROL_PLANE_AUDIT.md
2. docs/project/architecture/OBSERVABILITY.md
3. docs/project/architecture/satellites/OBSERVABILITY_extended_depth.md
4. docs/project/architecture/CRITIC_VERIFICATION.md
5. docs/project/architecture/TIER3_APPLICATION_ENVIRONMENT.md or the relevant satellite if the hub points there
6. docs/project/maintainers/plans/OBSERVABILITY.md
7. docs/project/maintainers/plans/satellites/OBSERVABILITY_audit_history.md
8. docs/project/maintainers/audit/README.md

Required documentation changes:
1. Add a concise canonical summary of Observability & Evaluation Control Plane to docs/project/architecture/OBSERVABILITY.md.
2. Add full target architecture sections to docs/project/architecture/satellites/OBSERVABILITY_extended_depth.md.
3. Add a cross-reference in docs/project/architecture/CRITIC_VERIFICATION.md explaining the boundary between CVL and OECP.
4. Add Tier-3 profile-level notes for custom telemetry, eval plugins, eval gates, perturbation profiles, and vendor export profiles.
5. Update docs/project/maintainers/plans/OBSERVABILITY.md with a compact pointer to a new plan satellite.
6. Create docs/project/maintainers/plans/satellites/OBSERVABILITY_eval_control_plane.md with phased implementation register.
7. If the repository uses a documentation map or satellite index, update it only if required by existing conventions.

Architecture rules:
1. HOS remains the only canonical observability spine.
2. OECP consumes HOS/unified journal/evidence records. It must not create a parallel trace system.
3. Evidence Ledger stores eval-ready evidence with references back to canonical trace/journal entries.
4. Eval Registry v2 stores cases, datasets, runs, metric results, observations, regression results, and perturbation lineage.
5. Custom telemetry must enter through typed contracts: DiagnosticPayload, RuntimeEventPayload, TelemetryProvider, TelemetryEnricher, EventSubscriptionHandler.
6. Custom telemetry must define schema_id, redaction behavior, namespace, tenant isolation, retention/export policy, and high-cardinality safeguards.
7. Langfuse, LangSmith, OTLP, Sentry and similar systems are optional sinks/workbenches, not semantic owners.
8. CRITIC_VERIFICATION owns per-run/per-step correctness verification. OBSERVABILITY/OECP owns continuous measurement, evidence, datasets, metrics, regression gates, and perturbation suites.
9. TIER3_APPLICATION_ENVIRONMENT owns application-level opt-in and wiring profiles, not separate observability semantics.
10. Do not implement code in this task.

Expected new architecture sections in OBSERVABILITY_extended_depth.md:
- Observability & Evaluation Control Plane
- Evidence Ledger
- Trace Completeness Contract
- Eval Registry v2
- Metric and Eval Plugin SDK
- Custom Telemetry Extension Plane
- TelemetryProvider and TelemetryEnricher contracts
- EventSubscriptionHandler reactions
- Counterfactual and Interpolation Engine
- External Observability Workbench Sync
- Eval-grade maturity model L5-L7

Expected new plan phases:
- OBS-ECP-0 Architecture canon
- OBS-ECP-1 Trace Completeness Contract
- OBS-ECP-2 Evidence Ledger
- OBS-ECP-3 Eval Registry v2
- OBS-ECP-4 Metric/Eval Plugin SDK
- OBS-CTP-1 Custom Telemetry Extension Plane
- OBS-CTP-2 Telemetry Provider / Enricher contracts
- OBS-CTP-3 Event reaction handlers
- OBS-PERT-1 Counterfactual Engine
- OBS-PERT-2 Interpolation Engine
- OBS-EXT-1 External Export Sync
- OBS-GATE-1 CI and release gates
- OBS-UX-1 Debug/workbench surfaces

Acceptance:
- The architecture contains decisions, not raw audit prose.
- The full audit remains in docs/project/maintainers/audit/OBSERVABILITY_EVALUATION_CONTROL_PLANE_AUDIT.md.
- The plan contains actionable IDs, deliverables, and acceptance criteria.
- No code changes.
- Links and relative paths are valid.
- Existing documentation governance is respected.

Commit message:
docs: add observability evaluation control plane architecture plan
```

---

## 21. Final transformation rule

Use this document as the source of truth for the transition:

```text
Audit finding -> architecture decision -> capability contract -> plan phase -> task ID -> implementation sprint
```

Do not skip directly from audit findings to code.
