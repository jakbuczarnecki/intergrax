# Harness Architecture Evolution Roadmap

## Status

This document is the canonical cross-domain roadmap for the next stage of Intergrax harness architecture evolution.

It does **not** replace domain architecture documents. Domain documents remain authoritative for the semantics and contracts owned by their respective areas. This roadmap defines:

- what must change across domains,
- what must remain unchanged,
- what must be added,
- what must be consolidated or removed,
- which architectural invariants must become enforceable,
- the implementation order,
- dependency relationships,
- acceptance criteria,
- and the target platform state after completion.

The roadmap is intentionally organized around a small number of major initiatives rather than hundreds of disconnected micro-tasks. Each initiative contains the concrete work items required to make the target architecture real.

---

## 1. Core architectural position

Intergrax should evolve toward a single coherent governed execution operating layer without flattening domain semantics.

The following boundaries remain intentional and must be preserved:

```text
Execution != Agent
Memory != RAG != Context Engineering
Tool != Skill != Integration
Orchestration != Nexus != Execution Runtime
Governance != Tool Runtime
Observability != Diagnostics
Proposal != Permission != Execution
Configured State != Effective State
Package/Plugin != Domain Capability
```

The objective is therefore not to create another universal abstraction that absorbs all domains. The objective is to make the existing domain architecture compose, execute, inspect, explain, and evolve as one coherent harness.

The target mental model is:

```text
ApplicationEnvironmentProfile
        ↓
Profile Resolution
        ↓
Effective Runtime Composition
        ↓
Execution Boundary
        ↓
Capabilities + Governance + Context
        ↓
Evidence + Diagnostics + Evaluation
        ↓
Adaptive, Governed Evolution
```

---

## 2. Canonical ownership decisions

### 2.1 ApplicationEnvironmentProfile remains the composition authority

`ApplicationEnvironmentProfile` remains the canonical Tier-3 environment composition contract.

Do **not** introduce a second peer authority such as a separate `HarnessProfile`.

Allowed additions around it:

- `ProfileResolution` as resolution evidence,
- an immutable effective profile snapshot,
- a read-only effective harness/environment projection,
- inspection and explanation surfaces,
- semantic profile diffing,
- profile lineage and provenance.

These additions must not become competing configuration authorities.

### 2.2 Domain semantic ownership remains distributed

Platform-level composition must not absorb domain semantics.

Examples:

- Tools continue to own tool execution semantics.
- Skills continue to own capability-bundle semantics.
- Context Engineering continues to own model-context assembly.
- Memory continues to own persistent memory semantics.
- RAG continues to own governed retrieval semantics.
- Governance continues to own authority and policy semantics.
- Nexus continues to own accepted orchestration topology execution decisions.
- Unified Execution Runtime continues to own execution lifecycle semantics.
- Observability/HOS continue to record platform facts rather than create them.

### 2.3 No new subsystem without new semantic ownership

Before introducing a new architecture domain, answer:

> Is there a genuinely new responsibility that no existing domain should own?

If the answer is no, implement the behavior as one of:

- provider,
- plugin,
- policy,
- guard,
- strategy,
- hook,
- adapter,
- projection,
- or domain-owned extension.

---

# 3. Mandatory architectural invariants

The following invariants must become explicit in architecture, implementation, tests, and runtime diagnostics where applicable.

## INV-1 — One environment composition authority

```text
ApplicationEnvironmentProfile is the only Tier-3 environment composition authority.
```

## INV-2 — Configured state is not effective state

```text
CONFIGURED != EFFECTIVE
```

The runtime must be able to expose both.

## INV-3 — Every effective decision is explainable

Any effective value, permission, capability, context inclusion, execution strategy, or rejection must have attributable provenance.

## INV-4 — Model-visible means reconstructable

```text
MODEL_VISIBLE => RECONSTRUCTABLE
```

Anything visible to a model request must be reconstructable from durable platform evidence according to the configured evidence policy.

## INV-5 — Child authority cannot exceed parent authority

```text
child_authority ⊆ parent_authority
```

## INV-6 — Proposal is not permission

```text
PROPOSAL != PERMISSION
```

## INV-7 — Permission is not execution

```text
PERMISSION != EXECUTION
```

## INV-8 — Topology identity is not runtime identity

```text
NodeId != ExecutionId
```

## INV-9 — Agent is not execution

```text
Agent != Execution
```

## INV-10 — Package/plugin is not domain capability

```text
PluginPackage != Tool != Skill != Memory != RAG != Context
```

## INV-11 — Context compaction is not evidence deletion

```text
MODEL_CONTEXT_COMPACTION != EVIDENCE_DELETION
```

## INV-12 — Observability records truth; it does not define business truth

Persisted platform facts remain the source of truth. External telemetry and projections are derived representations.

## INV-13 — Runtime extensions cannot self-expand authority

Any temporary or adaptive extension operates within pre-existing authority ceilings.

## INV-14 — Temporary capability expires with its owner scope

Execution-scoped or agent-scoped dynamic registrations must not leak beyond their owner scope.

## INV-15 — Dynamic registration is reversible

Every dynamic registration must have a deterministic unregistration/deactivation path.

## INV-16 — Meaningful side effects cross a governed execution boundary

No meaningful external mutation may bypass canonical authorization and side-effect enforcement.

## INV-17 — Every model call crosses Context Engineering

No host, agent, SDK, or adapter may build an independent model request path that bypasses canonical Context Engineering rules.

## INV-18 — Every tool call crosses ToolRuntime

No agent or host may execute tools through a private runtime path.

## INV-19 — Independent child work is admitted by Unified Execution Runtime

Independent work units must receive canonical execution identity and lifecycle semantics.

## INV-20 — Canonical runtime boundaries are transport-independent

CLI, SDK, REST, ACP, MCP, Slack, web, or any future host are adapters over the same runtime semantics.

---

# 4. Initiative A — Unified Execution Runtime convergence

## Goal

Turn the target execution model into the actual universal runtime spine for all meaningful AI work.

## A1. Canonical execution identity

Implement and enforce:

```text
TaskId
  ↓
RunId
  ↓
AttemptId
  ↓
ExecutionId
  ↓
EventId
```

Required work:

- canonical `ExecutionId` type and creation rules,
- `ExecutionId` available on canonical runtime events,
- explicit root Execution per Attempt,
- explicit `parent_execution_id` for child work,
- elimination of substitute execution identities on canonical paths,
- migration of agent-centric paths to execution-centric identity.

## A2. Neutral Execution Boundary

Establish a runtime boundary that is independent of the selected execution strategy.

Supported strategies should include at minimum:

- direct inference,
- agentic execution,
- orchestration.

The strategy may change the executor but must not change execution identity, authority, lifecycle, evidence, cancellation, budget, or result semantics.

## A3. Execution Tree

Make the execution tree a runtime fact, not only a conceptual diagram.

Required properties:

- root Execution,
- child Execution relationships,
- lineage queries,
- subtree status,
- subtree cancellation,
- subtree evidence aggregation,
- authority inheritance,
- budget inheritance.

## A4. Child Execution admission

Define when child work becomes a separate Execution.

Do not treat every function call, graph node, or tool call as a new Execution by default.

Admission must be explicit and consistent across:

- delegation,
- orchestration,
- remote agents,
- background work,
- long-running work,
- adaptive runtime extensions.

## A5. Hierarchical authority

Implement monotonic authority propagation:

```text
child effective authority <= parent effective authority
```

Any additional restriction is allowed; widening is not.

## A6. Hierarchical budget

Introduce parent-to-child budget allocation/reservation semantics.

The parent must retain authoritative control over total resource envelopes.

Cover:

- token budget,
- monetary budget,
- tool-call budget,
- execution count,
- concurrency,
- wall-clock limits where applicable.

## A7. Cancellation

Implement canonical cancellation semantics for:

- single Execution,
- execution subtree,
- remote child,
- background child,
- tool calls,
- model requests.

The runtime must define cooperative versus forceful cancellation boundaries.

## A8. Pause/resume/HITL

Pause and resume must be representable at Execution lifecycle level.

Human approval waits must not exist only as ad hoc callbacks.

## A9. Retry ownership

Separate and document ownership for:

- Run retry,
- Execution retry,
- tool retry,
- model-provider retry,
- transport retry.

Avoid generic retry layers with ambiguous ownership.

## A10. Execution Environment

Introduce or consolidate a canonical Execution Environment contract that exposes the effective runtime environment for one Execution:

```text
workspace
sandbox
credentials
network
filesystem
process capabilities
runtime capabilities
host capabilities
```

## A11. Neutral Execution Result ABI

Create a result boundary that does not leak agent-specific result types into orchestration or shared runtime contracts.

## Acceptance criteria

- all major execution paths carry canonical `ExecutionId`,
- Execution Tree can be reconstructed from runtime evidence,
- child authority cannot exceed parent authority,
- cancellation propagates deterministically,
- background and delegated work use the same Execution lifecycle,
- orchestration no longer requires agent-specific runtime identity/result semantics.

---

# 5. Initiative B — Profile resolution and effective composition

## Goal

Make the existing environment composition model simple, layered, explainable, inspectable, and immutable at runtime.

## B1. Formal profile layering

Canonical resolution order:

```text
platform defaults
        ↓
distribution/product defaults
        ↓
application environment
        ↓
agent overlay
        ↓
run/task overlay
        ↓
execution-local overlay
```

The exact available layers may vary by host, but precedence must be deterministic.

## B2. ProfileResolution record

Introduce a resolution object/evidence record containing:

- input layers,
- layer revisions,
- effective profile,
- overrides,
- rejected overrides,
- authority clamps,
- compatibility warnings,
- missing dependencies,
- degraded capabilities,
- resolution timestamp/revision.

## B3. Configured versus effective views

Expose both:

- what was configured/requested,
- what the runtime actually resolved.

## B4. Provenance per effective value

For important fields, answer:

```text
requested value
source layer
overridden value
overriding layer
policy clamp
effective value
```

## B5. Immutable effective snapshot

Persist or durably reference the exact effective environment/profile used by each Run/Execution.

Later configuration changes must not rewrite historical meaning.

## B6. Profile revision/fingerprint

Introduce stable revision/fingerprint semantics for resolved environments.

## B7. Semantic profile diff

Provide semantic diff between profiles/revisions, including changes such as:

- tools added/removed,
- policy changes,
- authority changes,
- context-budget changes,
- sandbox changes,
- model/provider changes,
- orchestration changes.

## B8. Capability dependency validation

Resolve and validate chains such as:

```text
Skill
  → Tool
  → Integration
  → Credential
  → Provider
```

## B9. Required versus optional capability

Explicitly classify dependencies as required or optional.

Required dependency missing:

```text
BOOT/RESOLUTION FAILURE
```

Optional dependency missing:

```text
DEGRADED EFFECTIVE PROFILE
```

## B10. User-facing presets

Add ergonomic presets such as:

- safe-readonly,
- developer,
- workspace-agent,
- governed-enterprise,
- automation,
- research.

Presets must expand to existing canonical policy/profile semantics; they do not create new authority models.

## B11. Legacy profile surface audit

Audit nested canonical profile representation versus legacy flat accessors and wire compatibility.

Remove compatibility surfaces where no real consumer requires them.

## Acceptance criteria

- one composition authority remains,
- every Run has an effective profile identity,
- configuration resolution is deterministic,
- important values are explainable,
- invalid required dependency chains fail before execution,
- deprecated duplicate composition paths are removed or formally contained.

---

# 6. Initiative C — Runtime inspection and explanation

## Goal

Make the actual live Intergrax runtime directly inspectable and explainable.

## C1. Canonical inspection API

Create a programmatic read model that powers all surfaces.

Do not implement separate logic independently in CLI, web, SDK, or diagnostics.

## C2. Environment inspection

Expose:

- configured state,
- effective state,
- profile revision,
- providers,
- capabilities,
- degraded capabilities,
- governance posture,
- sandbox,
- model routes,
- context providers,
- memory/RAG state,
- observability state.

## C3. Agent inspection

Expose effective:

- model,
- tools,
- skills,
- authority,
- memory,
- context,
- budgets,
- orchestration capabilities,
- runtime providers.

## C4. Execution inspection

Expose:

- Execution identity,
- parent/root,
- lifecycle state,
- strategy,
- effective authority,
- effective budget,
- environment,
- active children,
- model calls,
- tool calls,
- approvals,
- evidence references.

## C5. Tool inspection

For each tool show:

```text
registered
available
visible
selected
executable
authorized/denied
provider
risk
reason
```

## C6. Skill inspection

Expose:

- manifest,
- version,
- resolved dependencies,
- resolved tools,
- prompt/instruction references,
- policy fragments,
- provenance.

## C7. Context inspection

Expose active context providers and contribution/cost information.

## C8. Explain tool access

Example output semantics:

```text
tool exists: yes
host enabled: yes
agent declared: yes
skill allows: yes
runtime policy: deny
reason: mutation outside current resource scope
```

## C9. Explain profile values

Explain why a value became effective.

## C10. Explain execution strategy

Explain why direct inference, agentic execution, or orchestration was selected.

## C11. Explain context inclusion and exclusion

Support both questions:

- why was this fragment included?
- why was this fragment excluded?

## C12. Capability dependency graph

Generate or expose a graph linking:

- profiles,
- agents,
- skills,
- tools,
- integrations,
- credentials,
- providers,
- policies.

## C13. Execution Tree view

Provide machine-readable tree and human-readable visualization.

## C14. CLI surface

Target commands may include:

```text
intergrax inspect environment
intergrax inspect agent <id>
intergrax inspect execution <id>
intergrax inspect tools
intergrax inspect skills
intergrax inspect context
intergrax explain tool-access <tool_id>
intergrax explain context <request_id>
intergrax profile diff <a> <b>
intergrax graph execution <id>
```

Exact naming is implementation-defined; semantics are mandatory.

## Acceptance criteria

- operators can inspect actual effective runtime state,
- inspection is a projection of canonical platform facts,
- no inspection surface invents separate truth,
- at least tool access, context inclusion/exclusion, and effective profile values are explainable.

---

# 7. Initiative D — Reconstructable model execution

## Goal

Make every important model decision reproducible from platform evidence.

## D1. Canonical ModelRequest identity

Introduce `ModelRequestId` or equivalent canonical identity.

## D2. Execution linkage

Every model request references the owning Execution.

## D3. Provider/model identity

Persist or reference exact:

- provider,
- model,
- route/adapter,
- relevant generation settings.

## D4. Prompt/system revision

Persist or reference exact system/instruction composition revision.

## D5. Tool schema set revision

The request must identify the exact tool schema surface visible to the model.

## D6. Context assembly revision

The request must identify the exact context assembly decision/evidence.

## D7. Profile revision

The request references the effective environment/profile revision.

## D8. Structured model-request evidence

Prefer structured components and provenance over storing only a concatenated prompt string.

Evidence should cover as applicable:

- system instructions,
- conversation/history revision,
- context fragments,
- memory-derived context,
- RAG evidence,
- tool schemas,
- active instruction skills,
- policy overlays visible to the model,
- attachments/artifacts exposed to the model,
- model/provider identity,
- token budget,
- generation settings.

## D9. Deterministic reconstruction

Provide a reconstruction function/test path:

```text
persisted evidence
  ↓
reconstructed model request
  ↓
contract comparison
```

## D10. Tool-result visibility lineage

Every tool result shown to the model must be attributable to a canonical tool execution result.

## D11. Compaction lineage

If model history is compacted, summaries must reference the evidence range/source from which they were derived.

## D12. Model-visible versus user-visible distinction

Represent these separately when outputs differ.

## Acceptance criteria

- representative model calls can be reconstructed from persisted evidence,
- every model-visible tool result and context fragment has provenance,
- history compaction does not destroy audit evidence,
- reconstruction tests run in CI for canonical paths.

---

# 8. Initiative E — Context Engineering convergence and provider seam

## Goal

Preserve the existing Context Engineering architecture while making context sources composable, lazy, attributable, and operationally inspectable.

## E1. Preserve canonical CE pipeline

Maintain the core model:

```text
COLLECT
→ NORMALIZE
→ SCORE
→ FILTER
→ RANK
→ BUDGET
→ COMPRESS/DEGRADE
→ FORMAT
→ VALIDATE
→ EMIT
```

## E2. ContextProvider contract

Formalize a provider seam for sources such as:

- session/history,
- workspace instructions,
- files,
- references,
- Memory,
- RAG,
- runtime state,
- tool results,
- policy overlays,
- time/current-environment context,
- instruction skills.

## E3. Provider lifecycle

Support deterministic registration, activation, deactivation, and inspection.

## E4. Fragment provenance

Each context fragment must identify its source and transformation lineage.

## E5. Fragment lifetime

Support explicit lifetime semantics such as:

- step,
- turn,
- Execution,
- Run,
- Session,
- durable.

## E6. Context replacement semantics

Support replacement/supersession where appropriate rather than treating all context as append-only.

## E7. Workspace instruction provider

Support workspace-owned instruction sources that are visible through CE rather than private host injection.

## E8. Instruction refresh

Allow controlled refresh/revision when workspace instruction sources change.

## E9. File-reference provider

Files/artifacts should enter model context through attributable references/projections.

## E10. Session-reference provider

Support controlled references to other session/run evidence where policy permits.

## E11. Lazy activation

Avoid eagerly injecting full instructions or large resources when catalog/locator semantics are sufficient.

## E12. KV-cache-aware assembly

Where provider/model behavior makes it valuable, preserve stable prompt prefixes and avoid unnecessary high-level schema churn.

This is an optimization, not a correctness requirement.

## E13. Context cost attribution

Expose token cost by source/domain.

## E14. Context diagnostics

Support questions such as:

- which source consumed most context budget?
- which mandatory fragments forced degradation?
- what was dropped and why?

## Acceptance criteria

- all canonical model calls use CE,
- context sources use attributable provider contracts,
- CE can explain inclusions/exclusions,
- context cost by source can be measured,
- lazy instruction/resource loading is supported.

---

# 9. Initiative F — Canonical Tool Runtime pipeline

## Goal

Make tool execution one explicit, inspectable, governed pipeline.

## F1. Canonical pipeline

Define a single official sequence such as:

```text
resolve
→ validate input
→ authorize
→ pre-execute guards/policies
→ schedule
→ execute
→ normalize output
→ post-process
→ persist authoritative result
→ publish model/user projection
```

Exact internal hooks may vary, but ownership and authoritative boundaries must be explicit.

## F2. Authoritative tool result

Define one canonical final tool outcome event/object.

Presentation formatting must not replace the canonical result.

## F3. Typed input/output ABI

Tool contracts must retain stable typed input/output semantics.

## F4. Timeout contract

Tools declare/receive deterministic timeout semantics.

## F5. Cancellation contract

Tools must honor the runtime cancellation model according to declared capabilities.

## F6. Concurrency classification

Support classification such as:

```text
parallel-safe
exclusive
```

and enforce it at scheduling time.

## F7. Nested invocation lineage

Nested tool calls must preserve parent/root call lineage.

## F8. Cost attribution

Capture relevant tool resource/cost data for budgeting and diagnostics.

## F9. Guard extensions

Implement small loop-hygiene behaviors as tools-domain guards/plugins/policies rather than new architecture domains.

Examples:

- repeated identical call warning,
- runaway loop detection,
- execution deadline guard.

## F10. Eliminate private tool runtimes

Audit agents, hosts, integrations, SDK paths, and legacy code for execution paths that bypass canonical ToolRuntime.

## Acceptance criteria

- one canonical tool execution path exists,
- side effects cannot bypass governance,
- canonical result and presentation are separate,
- concurrency/timeout/cancellation behavior is explicit,
- private tool loops are removed or formally routed through ToolRuntime.

---

# 10. Initiative G — Runtime credentials and secret references

## Goal

Make credentials first-class governed runtime resources without exposing secret values to configuration or model context.

## G1. CredentialRef

Configuration should normally refer to credentials by stable reference rather than embedding secret values.

## G2. Credential provider seam

Define provider-neutral credential resolution.

Potential providers include:

- local secure storage,
- environment-backed development provider,
- enterprise secret stores,
- cloud secret managers,
- human authorization flows.

## G3. Per-operation/per-Execution resolution

Resolve credentials as late as practical and within the owning authority scope.

Avoid unrestricted process-global secret injection as the canonical model.

## G4. Rotation

Credential rotation should not require rewriting unrelated profile configuration.

## G5. Execution-scoped exposure

An Execution receives only credentials required and authorized for its effective scope.

## G6. Human authorization flow

Support credentials that must be acquired through interactive authorization rather than static configuration.

## G7. Secrets are not model-visible by default

Secret values must never enter model context unless an explicit specialized contract makes that unavoidable and authorized.

## G8. Credential access evidence

Record credential reference access and authorization decisions without logging secret values.

## Acceptance criteria

- configs can reference secrets without storing secret values,
- credential resolution is provider-neutral,
- runtime access is authority-scoped,
- credential use is auditable without secret leakage.

---

# 11. Initiative H — Unified Execution sandbox and isolation contract

## Goal

Provide a simple user-facing isolation model backed by pluggable execution-environment providers.

## H1. ExecutionSandbox contract

Introduce or consolidate a canonical sandbox/isolation boundary owned by Execution Environment semantics.

## H2. User-facing modes

Provide simple modes such as:

```text
READ_ONLY
WORKSPACE_WRITE
FULL_ACCESS
```

Names may change, but the UX must remain simple.

## H3. Provider abstraction

Allow implementations such as:

- local OS confinement,
- container,
- remote runtime,
- microVM,
- enterprise sandbox provider.

Do not require every backend on every platform.

## H4. Filesystem policy

Explicit readable/writable roots and mount semantics.

## H5. Network policy

Explicit outbound/inbound network rules where supported.

## H6. Process policy

Explicit process spawning/execution rules.

## H7. Credential policy

Define what credential references are exposed to a sandboxed Execution.

## H8. Governed escalation

A denied operation may request narrowly scoped one-shot elevation according to Governance/HITL policy.

## H9. Security honesty

Documentation must distinguish confinement from hard isolation and state provider limitations clearly.

## Acceptance criteria

- sandbox mode is visible in effective profile and Execution inspection,
- tool/code/process capabilities consume the same sandbox contract,
- escalation goes through Governance,
- provider-specific limitations are explicit.

---

# 12. Initiative I — Subagent and external-agent provider model

## Goal

Make delegation provider-neutral while retaining Intergrax execution, authority, budget, and evidence semantics.

## I1. SubagentProvider contract

Create a provider-neutral delegation seam.

## I2. Native Intergrax provider

Support child Intergrax agents in-process and/or remote as implementation permits.

## I3. Forked-context provider

Support child execution seeded from controlled parent history/context where needed.

## I4. Remote-agent providers

Support adapters for external agent protocols/services as providers rather than separate runtime architectures.

## I5. Delegation admission

Independent delegated work must become child Execution when it meets Execution admission rules.

## I6. Continuable child work

Support resuming/following up with appropriate child agents/executions.

## I7. Child control

Support:

- list,
- status,
- follow-up,
- interrupt/cancel,
- completion retrieval.

## I8. Authority/budget inheritance

External providers must not bypass parent authority ceilings or budget envelopes.

## I9. Evidence normalization

Normalize provider outputs and lifecycle evidence into canonical Intergrax Execution evidence.

## Acceptance criteria

- external agents can be invoked without becoming a second execution system,
- delegated work is visible in Execution Tree,
- authority/budget limits propagate,
- provider-specific telemetry remains secondary to canonical evidence.

---

# 13. Initiative J — Background Execution control surface

## Goal

Provide ergonomic background-job behavior without creating a second job runtime beside Unified Execution Runtime.

## J1. Background work as Execution

Background jobs are a projection/control surface over canonical Execution semantics.

Do not introduce an independent lifecycle identity if Execution can own it.

## J2. Ownership fencing

Background work must be scoped to its owning principal/session/run/application according to product semantics.

## J3. Non-blocking control

Support:

- start,
- list,
- status,
- wait,
- cancel,
- completion notification.

## J4. Durable background work

Where product requirements demand durability, background work must survive host/process interruption according to configured recovery semantics.

## J5. Detached semantics

Define explicitly whether an Execution may outlive its parent Session/Run and under what authority.

## J6. Completion routing

Completion notifications must route to the owning interaction surface without creating hidden state authority.

## Acceptance criteria

- no duplicate Job lifecycle system exists,
- background work is inspectable as Execution,
- ownership isolation is enforced,
- durable variants use canonical recovery/evidence mechanisms.

---

# 14. Initiative K — Verified external event intake

## Goal

Provide a generic governed path from authenticated external events to Intergrax work.

## K1. External event intake contract

Define a canonical provider-neutral event envelope.

## K2. Provider authentication

Each adapter verifies source-specific authenticity before normalization.

## K3. Normalization

Normalize provider deliveries into stable platform event types.

## K4. Tenant/workspace/principal resolution

Resolve the correct authority and workspace context before work admission.

## K5. Policy evaluation

External event rules must be subject to Governance.

## K6. Idempotency/deduplication

Provide delivery identity and duplicate handling for relevant providers.

## K7. Durable delivery/retry

Where reliability requires it, support durable intake/retry rather than process-local fire-and-forget behavior.

## K8. Event-to-work admission

External events may create Task/Run/Execution according to canonical intake rules.

## K9. Rule provenance

Record which trusted rule admitted or rejected an event.

## K10. Evidence

Persist normalized event identity, authorization result, admission result, and resulting work identities.

## Acceptance criteria

- provider adapters cannot bypass tenant/policy resolution,
- duplicate delivery is handled deterministically where configured,
- resulting work uses canonical Task/Execution paths,
- event admission is auditable.

---

# 15. Initiative L — Durable artifacts, attachments, and large-output spill

## Goal

Create one coherent artifact model so large or durable resources can exist outside model context while remaining retrievable and attributable.

## L1. Durable artifact identity

Use stable artifact identity independent of prompt/context placement.

## L2. Versioning/content identity

Use versioned and/or content-addressed storage semantics where appropriate.

## L3. Ownership and scope

Artifacts must carry tenant/workspace/principal/execution ownership as required.

## L4. Attachment provenance

Track source, upload/creation event, transformation lineage, and policy state.

## L5. Model-visible projection

Context Engineering decides what projection of an artifact is visible to the model.

## L6. Large-output spill

If tool or runtime output exceeds configured inline limits:

```text
full output → durable artifact
model context → bounded preview + locator
```

## L7. Read/search later

Provide controlled artifact retrieval/search capabilities.

## L8. Retention policy

Artifacts have explicit retention/expiry rules independent of context compaction.

## L9. Artifact lineage

Derived artifacts reference their source artifacts/evidence.

## L10. Evidence attachment

Important evidence records may reference artifacts rather than duplicating large payloads.

## Acceptance criteria

- large tool outputs do not force unbounded context growth,
- full data remains retrievable when policy permits,
- artifact existence is independent of model context,
- provenance/ownership are preserved.

---

# 16. Initiative M — Context compaction and retention

## Goal

Control context pressure without losing durable evidence.

## M1. Automatic compaction trigger

Trigger compaction based on model/provider-aware context pressure.

## M2. Explicit compaction API/command

Allow controlled manual compaction where product surfaces need it.

## M3. Spill/prune before summarizing where appropriate

Prefer moving large recoverable tool outputs to artifacts before compressing semantic conversation history.

## M4. Summary lineage

A compacted summary references the original event/evidence range.

## M5. Protected fragments

Policy, authority, critical task constraints, and required instructions must not be silently removed.

## M6. Quality validation

Evaluate compaction quality for information retention and task continuity.

## M7. Pinning

Allow important facts/events to be pinned against ordinary compaction.

## M8. Evidence independence

Compaction changes model-visible history, not canonical historical evidence.

## Acceptance criteria

- automatic compaction works on canonical model paths,
- compacted context can be traced to original evidence,
- critical policy/task constraints survive compaction,
- audit history is unaffected by model-context reduction.

---

# 17. Initiative N — Runtime invariant service

## Goal

Let domains continuously verify that the live runtime still satisfies critical architecture contracts.

## N1. Domain-owned invariants

Each semantic owner defines invariants for relationships it owns.

Examples:

```text
MODEL_VISIBLE_RECONSTRUCTABLE
CHILD_AUTHORITY_NOT_GREATER_THAN_PARENT
SIDE_EFFECT_HAS_AUTHORIZATION
TOOL_RESULT_HAS_TOOL_CALL
EXECUTION_EVENT_HAS_EXECUTION_ID
PROFILE_REVISION_EXISTS
BACKGROUND_CHILD_HAS_OWNER
```

## N2. Central invariant runner

Provide a common registry/execution/reporting service.

## N3. Runtime selection

Support enabling/disabling categories for development, qualification, and production as policy permits.

## N4. Attribution

Violations must identify the owning domain/package/component.

## N5. Severity

Support severity/impact classification.

## N6. Evidence output

Invariant failures produce canonical diagnostic/evidence records.

## N7. CI reuse

Where practical, the same invariant logic should be reusable in tests/qualification and live runtime checks.

## N8. Invariant catalog

Generate documentation/catalog from registered invariants.

## Acceptance criteria

- critical architecture invariants have executable checks,
- violations are attributable,
- checks can run in CI and selected runtime profiles,
- invariant failures integrate with diagnostics/evidence.

---

# 18. Initiative O — Dynamic orchestration proposals

## Goal

Allow models/planners to propose flexible work topology without allowing generated topology to become runtime authority by itself.

## O1. WorkflowProposal

Introduce a typed proposal artifact for dynamically authored orchestration.

## O2. Proposal validation

Validate:

- schema,
- capabilities,
- authority,
- budget,
- cycles,
- max depth,
- fan-out,
- concurrency,
- provider availability.

## O3. Proposal is not accepted topology

The model/planner proposal remains untrusted until accepted by platform rules.

## O4. Accepted OrchestrationDefinition

Only accepted topology enters Nexus execution semantics.

## O5. Alternate proposal providers

Model-authored scripts or external planners may be topology proposal providers, not independent execution authorities.

## Acceptance criteria

- generated workflows cannot bypass Governance or Nexus,
- accepted topology has explicit provenance,
- invalid/budget-violating topology is rejected before execution,
- dynamic topology is inspectable and reconstructable.

---

# 19. Initiative P — Capability Skills and Instruction Skills

## Goal

Preserve strong capability-composition Skills while adding lightweight lazy-load instructional capabilities.

## P1. Preserve CapabilitySkill semantics

Current Skills remain declarative capability packages that may include:

- tool references,
- prompt references,
- policy fragments,
- dependencies,
- risk metadata.

## P2. Version identity hardening

Ensure manifest version, registry identity, resolved identity, and persisted provenance cannot diverge ambiguously.

## P3. Preserve resolved provenance

Do not reduce resolved Skills to only a final allowed-tool list.

Retain a resolved skill snapshot/package for evidence and inspection.

## P4. InstructionSkill

Add a distinct lightweight instructional skill class for reusable task methods/instructions.

Suggested properties:

- skill id,
- version,
- description,
- instruction source/artifact,
- activation policy,
- token estimate,
- provenance.

## P5. Lazy loading

Expose compact catalogs/descriptions and load full instruction content only when required.

## P6. No authority grant

Instruction Skills cannot independently grant tool access or increase authority.

## P7. Development refresh

Support controlled development-time refresh of instruction content without weakening production versioning.

## Acceptance criteria

- CapabilitySkill semantics remain unchanged,
- InstructionSkill is clearly separate,
- Skill provenance survives resolution,
- lazy instruction loading reduces unnecessary context usage.

---

# 20. Initiative Q — Reversible capability/plugin lifecycle

## Goal

Create one consistent lifecycle vocabulary across extension surfaces without introducing a universal runtime semantics engine.

## Q1. Common lifecycle vocabulary

Where applicable, align on:

```text
discover
→ validate
→ register
→ activate
→ inspect
→ deactivate
→ unregister
```

## Q2. Reversible registration handles

Dynamic registrations return explicit handles/tokens that support deterministic teardown.

## Q3. Owner scoping

Support scopes such as:

- platform/global,
- application,
- agent,
- Execution.

## Q4. Automatic scoped cleanup

Execution-scoped registrations are removed when the owning Execution ends.

## Q5. Replace/version semantics

Define deterministic behavior for:

- replacement,
- version coexistence,
- upgrade,
- rollback,
- missing dependency.

## Q6. CapabilityDescriptor

Introduce shared control-plane metadata without flattening domain semantics.

Suggested fields:

```text
capability_id
kind
version
provider
config_schema
dependencies
provided_services
risk_class
trust_level
lifecycle_support
inspection_metadata
qualification_state
provenance
```

## Q7. Platform Plugins remain package/control-plane infrastructure

Platform Plugins may own packaging, discovery, compatibility, trust, qualification, and lifecycle vocabulary, but not replace Tool/Skill/RAG/Memory/etc. semantics.

## Acceptance criteria

- dynamic registrations are reversible,
- temporary registrations are scope-safe,
- common lifecycle terminology exists,
- domain semantics remain owned by domains.

---

# 21. Initiative R — Governance UX and permission presets

## Goal

Keep rich policy semantics while making permissions understandable to users and operators.

## R1. Preserve Meaningful Side Effect Authorization

Fresh side-effect evaluation remains authoritative immediately before the mutation.

## R2. Approval cannot override fresh deny

Preserve fail-closed semantics.

## R3. Consumable one-shot grants

Approval grants must be identity-bound and consumable according to operation semantics.

## R4. Permission presets

Expose understandable presets such as:

```text
observe
workspace
controlled
trusted
full
```

Presets expand to canonical sandbox, ToolRuntime, Governance, and authority configuration.

## R5. Explain permission

Support operator explanations for:

- why approval was required,
- why an action was denied,
- which rule/authority boundary determined the result.

## R6. Side-effect preview

Before human approval, show the intended resource/action/change in a stable normalized format.

## R7. Effective authority snapshot

Each Execution exposes the effective authority envelope under which it operates.

## Acceptance criteria

- permission UX becomes simpler without simplifying underlying policy semantics,
- approval decisions are identity-bound and auditable,
- operators can explain deny/approval outcomes.

---

# 22. Initiative S — Workspace, settings, and credentials separation

## Goal

Prevent unrelated concepts from collapsing into a single environment/configuration object.

## S1. Workspace is runtime/product context

Workspace owns identity/scope/resources/policy context appropriate to the product.

## S2. Settings are preferences/configuration

User/operator settings are not secret storage and not workspace identity.

## S3. Credentials are secret resources

Credentials use references/providers and separate authority rules.

## S4. Environment composes references

`ApplicationEnvironmentProfile` may reference/configure these systems, but must not become the direct storage authority for their runtime data.

## Acceptance criteria

- workspace identity, settings, and credentials have distinct contracts,
- environment composition references them without conflating their semantics.

---

# 23. Initiative T — Feedback, goals, plans, and work-state capabilities

## Goal

Support collaboration state only where it creates product/evaluation value, without creating unnecessary platform domains.

## T1. Feedback bridge

Create a canonical bridge:

```text
human feedback
→ evaluation evidence
→ outcome signal
→ optional adaptive input
```

This is the highest-value capability in this group.

## T2. Goals

If required by scenarios, represent persistent same-session/run goals as structured work state.

## T3. Plans

Allow reviewed planning state where applications benefit from explicit plans.

## T4. Todo/checklist

Treat todos primarily as a collaboration-state/tool surface rather than a new architecture owner unless future requirements justify otherwise.

## T5. No silent authority

Plan/goal/todo state never grants execution permission.

## Acceptance criteria

- feedback is available to Evaluation/AHI as attributable evidence,
- optional work-state features do not become duplicate execution/orchestration authorities.

---

# 24. Initiative U — Scheduling convergence

## Goal

Make scheduled work an admission mechanism into the canonical Task/Execution system rather than a separate agent runtime.

## U1. Scheduled intent

Represent the requested future work and schedule independently from actual Execution.

## U2. Admission at trigger time

At execution time, resolve current:

- authority,
- policy,
- credentials,
- profile,
- workspace,
- capability availability.

## U3. Durable schedules

Support durable recurrence/cancellation where product requirements demand it.

## U4. Evidence

Record creation, modification, trigger, cancellation, and resulting Task/Execution identities.

## Acceptance criteria

- scheduled work cannot bypass current policy at trigger time,
- resulting work uses canonical execution semantics,
- schedule state and runtime execution state remain separate.

---

# 25. Initiative V — Provider-neutral process, filesystem, terminal, and code runtimes

## Goal

Support strong coding/automation capabilities without forcing Intergrax to reimplement every specialized runtime locally.

## V1. Provider-neutral capability contracts

Where required, define seams such as:

```text
ProcessProvider
FilesystemProvider
TerminalProvider
CodeRuntimeProvider
```

## V2. Governance integration

All such capabilities must consume canonical:

- Execution Environment,
- Sandbox,
- CredentialRef,
- ToolRuntime/Governance,
- evidence.

## V3. External providers first where economically sensible

Prefer adapters to mature external/remote execution systems when they satisfy requirements.

## V4. Local implementations only when justified

Build local process/terminal/code/LSP implementations only when required by a product/scenario and when they deliver clear strategic value.

## V5. No feature-parity race

The platform goal is orchestration/governance of capable executors, not maximal duplication of specialized coding-agent functionality.

## Acceptance criteria

- code/process capabilities can be provided externally,
- external execution does not bypass Intergrax authority/evidence,
- local implementation scope is scenario-driven.

---

# 26. Initiative W — SDK, API, ACP, MCP, and host convergence

## Goal

Ensure every transport and host is an adapter over the same canonical runtime.

## W1. Shared runtime entry boundaries

The following must route to the same canonical semantics:

- Python/other SDKs,
- REST/HTTP,
- ACP,
- MCP,
- CLI,
- Slack/chat channels,
- web UI,
- future hosts.

## W2. No private policy model

Hosts must not implement competing authorization semantics.

## W3. No private tool runtime

Hosts and SDKs cannot execute tools outside ToolRuntime.

## W4. No private execution lifecycle

Hosts cannot create substitute run/execution identities.

## W5. Host capability declaration

Each host declares what interaction/runtime capabilities it provides.

## W6. Qualification per host/profile

Qualification evidence should identify which host/profile/runtime combinations are proven.

## Acceptance criteria

- transport choice does not change core execution semantics,
- policy/tool/context boundaries remain canonical across hosts,
- host-specific capability gaps are inspectable.

---

# 27. Initiative X — Generated architecture and capability metadata

## Goal

Reduce documentation drift by generating mechanical catalogs and dependency views from source-of-truth contracts.

## X1. Generated capability catalog

## X2. Generated tool catalog

## X3. Generated skill catalog

## X4. Generated provider catalog

## X5. Generated configuration catalog

## X6. Generated invariant catalog

## X7. Generated module/dependency graph

## X8. Generated extension/capability dependency graph

## X9. CI freshness gates

Fail CI when generated architecture/catalog artifacts are stale relative to canonical source contracts.

## Acceptance criteria

- mechanical documentation is generated rather than hand-duplicated,
- stale generated artifacts are caught in CI,
- architects can inspect dependency graphs without manual reconstruction.

---

# 28. Initiative Y — Documentation architecture simplification

## Goal

Make the documentation reflect the runtime mental model instead of amplifying internal complexity.

## Y1. One glossary

Maintain one canonical runtime terminology source.

## Y2. One identity vocabulary

Task/Run/Attempt/Execution/Event must have consistent definitions across docs.

## Y3. Each domain answers one primary question

Examples:

```text
Memory → what persists?
RAG → what external knowledge is retrieved?
Context Engineering → what enters the model now?
Governance → what is allowed?
Execution Runtime → how does work live and execute?
Nexus → what executes next in accepted topology?
Observability → what platform facts are recorded?
Diagnostics → what do those facts mean operationally?
```

## Y4. CURRENT before TARGET

Architecture documents must clearly separate implemented reality from target architecture.

## Y5. Canonical implementation links

Each major architecture document references canonical runtime implementations and tests/proofs where practical.

## Y6. Public versus deep technical vocabulary

Public/product docs should avoid unnecessary internal acronyms when simpler language works.

## Y7. Stale documentation detection

Automate checks where mechanical freshness can be verified.

## Acceptance criteria

- major terms are unambiguous,
- target architecture is not accidentally presented as shipped reality,
- duplicated mechanical documentation is reduced.

---

# 29. Initiative Z — Security, trust, and supply-chain hardening

## Goal

Make third-party providers, plugins, runtime extensions, credentials, and execution environments explicit trust-boundary objects.

## Z1. Trust metadata

Every external provider/plugin/capability package exposes trust classification.

## Z2. Risk classification

Capabilities and side effects expose risk metadata appropriate to Governance.

## Z3. Credential isolation

Execution receives only authorized credential references.

## Z4. Network destination policy

Support governed network-destination restrictions where relevant.

## Z5. Filesystem mount policy

Explicit access roots/mounts.

## Z6. Process spawning policy

Explicit runtime permission.

## Z7. Third-party activation warning/approval

High-risk third-party extensions require appropriate policy/HITL treatment.

## Z8. Extension sandbox tests

Dynamic or third-party code should be testable in constrained environments before activation.

## Z9. Supply-chain evidence

Record package/provider identity, version, provenance, qualification state, and activation history.

## Z10. Permission diff before activation

Activation tooling should surface the capability/authority delta introduced by a new provider/extension.

## Acceptance criteria

- runtime knows what third-party code/provider is active,
- credential and network access are explicit,
- activation history and permission deltas are inspectable.

---

# 30. Initiative AA — Memory and RAG hardening

## Goal

Preserve current Memory/RAG/CE separation while closing governance, persistence, and evidence gaps.

## AA1. Preserve domain separation

```text
Memory != RAG != Context Engineering
```

## AA2. Memory scope authority

Ensure memory reads/writes honor tenant/user/session/task/organization scope boundaries.

## AA3. Tool-result persistence policy

Tool output must not silently become long-term memory.

Distinguish explicit persistence from ephemeral runtime use.

## AA4. Memory write evidence

Record why/where memory was written.

## AA5. Consolidation provenance

Derived/consolidated memories reference source memories/evidence.

## AA6. Deletion/tombstone evidence

Logical deletion and vector-index tombstones remain attributable.

## AA7. RAG retrieval scope authority

Retrieval must apply tenant/namespace/workspace policy before context inclusion.

## AA8. Canonical RetrievalHit ABI

Normalize retrieval results across providers.

## AA9. Retrieval strategy evidence

Record dense/hybrid/hierarchical/graph/reranker decisions as configured.

## AA10. Query reconstruction

Persist or reconstruct the effective retrieval query and filters for important paths.

## AA11. CE retains inclusion ownership

Retrieval does not decide by itself what enters model context.

## Acceptance criteria

- Memory and RAG maintain distinct semantics,
- persistence and retrieval decisions are evidence-backed,
- unauthorized cross-scope retrieval is impossible on canonical paths,
- tool output cannot silently become LTM.

---

# 31. Initiative AB — Adaptive Harness Intelligence expansion

## Goal

Expand adaptive behavior from profile tuning toward governed evolution of versioned harness artifacts without allowing uncontrolled mutation.

## AB1. Broaden adaptive artifact taxonomy

Potential versioned adaptive artifacts:

```text
RoutingProfile
ExecutionStrategyProfile
ContextProfile
RagProfile
ToolSelectionProfile
PolicyRecommendation
InstructionSkill
WorkflowDefinition
CapabilityProviderRecommendation
RuntimeExtensionCandidate
```

Not every class must support automatic application.

## AB2. Authority level per artifact class

Each artifact class defines the maximum adaptation authority allowed:

- observe only,
- recommend,
- human-gated apply,
- bounded automatic apply.

## AB3. Shadow/canary lifecycle

Retain versioning, shadow, canary, verification, keep/rollback semantics.

## AB4. No envelope expansion

AHI cannot expand its own authority envelope.

## AB5. Reconstructable adaptation evidence

Every recommendation/application must link outcome signals, evidence, candidate artifact, governance decision, verification, and rollback state.

## Acceptance criteria

- adaptation remains versioned and governed,
- new artifact classes cannot bypass class-specific authority limits,
- adaptive decisions are reconstructable.

---

# 32. Initiative AC — Governed Runtime Evolution

## Goal

Allow Intergrax to detect missing capabilities and safely propose, validate, temporarily activate, evaluate, promote, or roll back runtime extensions.

This is a late-stage initiative and must not begin before the required execution, sandbox, invariant, credential, composition, and evidence foundations are complete.

## AC1. CapabilityGap

Define a typed record for a missing capability discovered during execution.

## AC2. RuntimeExtensionProposal

A proposal should declare:

- purpose,
- requested interface,
- implementation/source artifact,
- dependencies,
- required authority,
- risk,
- expected scope/lifetime,
- tests.

## AC3. Static validation

Validate schema, imports/dependencies, forbidden operations, and declared contracts before execution.

## AC4. Sandbox build/test

Build and execute candidate tests in an isolated environment.

## AC5. Contract tests

Verify the candidate against the declared capability ABI.

## AC6. Scenario/golden tests

Evaluate candidate behavior against bounded expected scenarios where available.

## AC7. Shadow execution

Run candidate behavior without granting authoritative production side effects where possible.

## AC8. Governance decision

Activation authority depends on risk, owner scope, artifact class, and current policy.

## AC9. Execution-scoped activation first

The safest default dynamic activation scope is the owning Execution.

## AC10. Automatic expiry

Temporary capability registration expires with its owner scope unless promoted through an explicit separate workflow.

## AC11. Canary activation

Allow broader bounded activation after successful shadow validation where policy permits.

## AC12. Promotion

Persistent plugin/provider installation is a distinct governed promotion workflow.

## AC13. Rollback

Activation and promotion must support deterministic rollback.

## AC14. Evidence

Record the complete lifecycle:

```text
capability gap
→ proposal
→ validation
→ sandbox/tests
→ governance
→ shadow
→ activation
→ use
→ evaluation
→ promotion/expiry/rollback
```

## Acceptance criteria

- dynamic extensions cannot self-expand authority,
- temporary extensions are scope-bounded and reversible,
- activation is evidence-backed,
- persistent promotion requires a separate governed decision.

---

# 33. Initiative AD — Test and qualification infrastructure

## Goal

Create deterministic test support for the new runtime invariants and cross-domain contracts.

## AD1. Harness testkit

Provide reusable deterministic construction of representative runtime compositions.

## AD2. Model mock/fault server

Support scripted model responses, failures, delays, malformed tool calls, and provider errors.

## AD3. Model replay adapter

Replay recorded model interactions where appropriate.

## AD4. Tool replay

Allow deterministic replay/stubbing of tool outcomes.

## AD5. Session/run snapshots

Snapshot durable state for regression testing.

## AD6. Effective profile snapshots

Verify profile resolution and composition changes.

## AD7. Model request reconstruction tests

Prove `MODEL_VISIBLE => RECONSTRUCTABLE` on canonical paths.

## AD8. Registration leak tests

Ensure temporary registrations are removed.

## AD9. Cancellation race tests

Test parent/child/tool/remote cancellation behavior.

## AD10. Parallel tool tests

Verify safe/exclusive concurrency semantics.

## AD11. Sandbox matrix

Test supported providers/modes and escalation boundaries.

## AD12. Subagent provider contract tests

All providers must satisfy the same lifecycle/authority/result semantics.

## AD13. Workflow proposal adversarial tests

Test invalid topology, authority expansion, runaway fan-out, excessive depth, and budget violations.

## AD14. Runtime extension hostile-code tests

Before Governed Runtime Evolution ships, test escape attempts and authority violations.

## AD15. Scenario Proofs remain highest-level falsification

Unit/integration tests do not replace Scenario Proof evidence for platform-level claims.

## Acceptance criteria

- canonical invariants have deterministic automated tests,
- provider contracts have reusable conformance suites,
- Scenario Proofs can demonstrate the integrated architecture under adversarial conditions.

---

# 34. Initiative AE — Developer and operator experience

## Goal

Reduce the cognitive cost of using Intergrax without reducing architectural depth.

## AE1. Minimal canonical quick start

A developer should be able to run a small governed harness without understanding every internal domain first.

## AE2. Sensible defaults

Defaults should produce safe, inspectable behavior.

## AE3. Canonical CLI concepts

Target capabilities include:

- run,
- inspect,
- explain,
- doctor,
- profile diff,
- replay,
- proof run.

## AE4. `doctor`

Diagnose:

- missing providers,
- invalid credentials refs,
- unsupported sandbox mode,
- capability dependency failures,
- stale configuration,
- host capability mismatches.

## AE5. Generated catalogs

Expose current tools, skills, providers, capabilities, profiles, invariants, and configuration surfaces.

## AE6. One boot mental model

Document the runtime boot flow clearly:

```text
load composition
→ resolve dependencies
→ resolve effective profile
→ validate capabilities
→ activate providers
→ expose inspection state
→ admit work
```

## Acceptance criteria

- a new developer can understand active runtime composition without reading all architecture documents,
- common misconfiguration can be diagnosed automatically,
- runtime explanation is available without custom debugging.

---

# 35. Consolidation and removal program

The roadmap includes deliberate removal of redundant paths.

## 35.1 Legacy flat environment-profile compatibility

Audit actual consumers and remove legacy flat compatibility paths where they no longer have justified use.

## 35.2 Legacy ToolBase and legacy tool aliases

Converge on canonical ToolContract/ToolRuntime surfaces.

## 35.3 Private agent tool loops

Remove or route through canonical ToolRuntime.

## 35.4 Competing execution identities

Do not allow AgentRunId/NodeRunId/WorkerRunId or similar identities to substitute for canonical Execution identity.

## 35.5 Duplicate registries

Audit registries for true semantic ownership versus historical duplication.

## 35.6 Duplicate plugin discovery

Consolidate discovery/control-plane behavior where domain separation does not require duplication.

## 35.7 Duplicate config resolution

One canonical environment/profile resolution path.

## 35.8 Duplicate observability paths

All authoritative runtime facts flow through canonical observability/evidence contracts.

## 35.9 Retry engine overlap

Audit retry logic and assign explicit ownership to each retry layer.

## Acceptance criteria

- duplicate runtime authorities are removed,
- compatibility layers are justified by real consumers,
- new code cannot reintroduce private execution/tool/config authorities without architecture review.

---

# 36. Implementation order

The program should be implemented in dependency order. Later waves must not pull foundational target concepts forward before their prerequisites exist.

## P0 — Canonical runtime spine

Complete first:

1. canonical `ExecutionId`, root and parent relationships,
2. Execution Tree,
3. neutral Execution Boundary,
4. RuntimeEvent execution identity,
5. child authority propagation,
6. child budget propagation,
7. subtree cancellation,
8. neutral Execution result ABI,
9. configured versus effective profile distinction,
10. `ProfileResolution`,
11. immutable effective profile revision/snapshot,
12. `MODEL_VISIBLE => RECONSTRUCTABLE`,
13. canonical model-request identity/evidence,
14. canonical Runtime Inspection API foundation,
15. runtime invariant service foundation,
16. elimination of competing composition/execution authorities on canonical paths.

### P0 exit gate

Do not call P0 complete until representative direct-inference, agentic, and orchestration paths all use the same execution/evidence/profile identity spine.

---

## P1 — Composition, inspection, governance UX, and safety foundation

1. formal profile layering,
2. profile provenance and semantic diff,
3. dependency validation,
4. required/optional capability handling,
5. agent/run/execution overlays,
6. environment/agent/execution/tool/skill/context inspection,
7. explain tool access,
8. explain profile value,
9. explain context inclusion/exclusion,
10. capability dependency graph,
11. runtime credential references/providers,
12. unified ExecutionSandbox contract,
13. permission presets,
14. reversible registration handles,
15. CapabilityDescriptor/control-plane metadata,
16. generated capability/config/provider catalogs,
17. runtime invariant catalog and initial critical invariants.

### P1 exit gate

An operator must be able to understand what the runtime resolved, what is active, what is allowed, why something is denied, and which authority/profile/evidence revision governs a given Execution.

---

## P2 — Runtime capability power

1. ContextProvider seam,
2. workspace/file/session reference providers,
3. lazy Instruction Skills,
4. context cost attribution,
5. KV-cache-aware CE optimizations,
6. canonical large-output spill/artifacts,
7. context compaction,
8. canonical tool-result convergence,
9. tool concurrency/timeout/cancellation hardening,
10. SubagentProvider,
11. native/remote/external agent providers,
12. continuable child work,
13. background Execution control surface,
14. verified external event intake,
15. scheduling convergence,
16. SDK/API/ACP/MCP/host convergence,
17. provider-neutral process/filesystem/terminal/code runtime seams where scenarios require them.

### P2 exit gate

Intergrax must be able to compose heterogeneous agents and capabilities under one Execution/Governance/Evidence model without host-specific or provider-specific runtime forks.

---

## P3 — Dynamic orchestration and adaptive expansion

1. WorkflowProposal,
2. topology/capability/authority/budget validation,
3. accepted dynamic OrchestrationDefinition,
4. feedback → Evaluation/AHI bridge,
5. adaptive artifact taxonomy expansion,
6. class-specific adaptive authority levels,
7. shadow/canary/verification support for new artifact classes,
8. richer runtime invariant coverage,
9. Scenario Proofs targeting new dynamic behavior.

### P3 exit gate

Dynamic topology and adaptive changes must remain proposals until accepted through deterministic platform governance and must remain fully reconstructable.

---

## P4 — Governed Runtime Evolution

Only after P0–P3 foundations are proven:

1. CapabilityGap,
2. RuntimeExtensionProposal,
3. static validation,
4. sandbox build/test,
5. contract tests,
6. hostile-code tests,
7. shadow execution,
8. governance decision,
9. Execution-scoped activation,
10. expiry,
11. canary,
12. verification,
13. promotion workflow,
14. rollback,
15. full lifecycle evidence,
16. Scenario Proof demonstrating safe bounded runtime evolution.

### P4 exit gate

A dynamically proposed capability must be unable to escape its authority, scope, lifecycle, sandbox, or evidence requirements even under adversarial testing.

---

# 37. Dependency map

The most important dependencies are:

```text
ExecutionId / Execution Tree
        ↓
Authority + Budget inheritance
        ↓
Background Execution / Delegation / Dynamic Work
```

```text
ProfileResolution
        ↓
Effective Runtime Snapshot
        ↓
Inspection + Explanation + Reproducibility
```

```text
ModelRequest Evidence
        ↓
Reconstructability
        ↓
Diagnostics / Evaluation / Replay
```

```text
Artifacts + Spill
        ↓
Compaction
        ↓
Context Efficiency without Evidence Loss
```

```text
Sandbox + Credentials + Invariants + Reversible Registration
        ↓
Governed Runtime Evolution
```

```text
Feedback + HOS + Evaluation
        ↓
AHI
        ↓
Governed Adaptive Evolution
```

---

# 38. What must explicitly not be built

The following are anti-goals unless future evidence changes the decision:

1. A second composition authority parallel to `ApplicationEnvironmentProfile`.
2. A universal plugin base class that erases Tool/Skill/RAG/Memory/Context semantics.
3. A second Job runtime separate from Unified Execution Runtime.
4. A separate dynamic-workflow execution engine parallel to Nexus.
5. A private session runtime used as a substitute for Memory/Execution/Observability semantics.
6. Automatic global installation of model-generated code.
7. Runtime extensions that can widen their own authority.
8. Full local feature-parity race for terminal/LSP/code runtime capabilities where external providers are sufficient.
9. A new architecture subsystem for small guard/policy behaviors.
10. Context compaction that deletes audit evidence.
11. Host-specific ToolRuntime, Governance, Context Engineering, or Execution semantics.
12. RAG or Memory directly injecting model context outside Context Engineering.
13. Plans, goals, or todos becoming hidden execution authority.
14. Telemetry sinks becoming canonical platform truth.

---

# 39. Scenario Proof requirements

Every major cross-domain capability added by this roadmap should have bounded adversarial proof coverage before strong maturity claims are made.

Priority proof themes:

## Proof A — Execution identity and lineage

Demonstrate:

- root Execution,
- children,
- retries,
- cancellation,
- evidence reconstruction.

## Proof B — Effective composition explainability

Demonstrate:

- layered profile resolution,
- rejected override,
- authority clamp,
- tool-access explanation.

## Proof C — Model-request reconstruction

Demonstrate exact reconstruction of a representative model request from persisted evidence.

## Proof D — Governed external delegation

Delegate to an external provider while preserving parent authority, budget, Execution Tree, and canonical evidence.

## Proof E — Large-output artifact/spill

Demonstrate bounded model context while preserving full output as retrievable durable artifact.

## Proof F — Runtime invariant violation

Intentionally violate a critical invariant and show detection, attribution, and evidence.

## Proof G — External event to governed work

Authenticate, deduplicate, admit, execute, and audit an external event.

## Proof H — Governed runtime evolution

Late-stage proof:

- detect missing capability,
- propose extension,
- sandbox/test,
- govern,
- shadow,
- activate only within bounded scope,
- use capability,
- expire or promote,
- reconstruct the full lifecycle.

---

# 40. Target user/developer experience

After the roadmap is complete, a developer should be able to express a high-level environment without manually wiring every subsystem.

Illustrative mental model:

```yaml
profile: governed-workspace-agent
model: primary
permissions: workspace
memory: standard
knowledge: enabled
sandbox: workspace-write
```

The platform resolves this into the canonical domain contracts.

The developer/operator should then be able to ask:

```text
What is active?
Why is this tool unavailable?
Why did this context fragment enter the model?
Why was this fragment excluded?
Which profile value won?
Which policy required approval?
What children did this Execution spawn?
Which credentials were referenced?
Which side effects occurred?
Can the exact model request be reconstructed?
What changed between two runtime profile revisions?
Which runtime invariant failed?
```

The system must answer from canonical state and evidence rather than from best-effort diagnostics.

---

# 41. Target platform architecture

The intended end state is:

```text
                     USER / APPLICATION / EVENT
                               │
                               ▼
                    canonical intake/admission
                               │
                               ▼
                 ApplicationEnvironmentProfile
                               │
                       ProfileResolution
                               │
                               ▼
                    Effective Runtime View
                               │
                               ▼
                     Execution Boundary
                               │
              ┌────────────────┼────────────────┐
              │                │                │
          inference          agentic       orchestration
                               │                │
                               │              Nexus
                               │                │
                               └───────┬────────┘
                                       ▼
                                   Execution
                                       │
       ┌──────────────┬────────────────┼─────────────────┬──────────────┐
       ▼              ▼                ▼                 ▼              ▼
      CE            Tools            Memory             RAG       Delegation
       │              │                                  │              │
       │         Governance                         retrieval      child Exec
       │              │                                  │              │
       └──────────────┴──────────────────┬───────────────┴──────────────┘
                                         ▼
                                Runtime Events / HOS
                                         │
                                         ▼
                          Evidence / Diagnostics / Eval
                                         │
                                         ▼
                                        AHI
                                         │
                                         ▼
                              Governed Evolution
```

Cross-cutting control plane:

```text
Platform Plugins / Capability Metadata
Profile Resolution
Inspection / Explanation
Runtime Invariants
Credentials
Sandbox / Execution Environment
Artifacts
Generated Catalogs
```

---

# 42. Expected outcome after full completion

The completed architecture should provide the following properties simultaneously:

## 42.1 Simpler composition without weaker semantics

Intergrax retains domain separation while presenting one coherent environment/harness mental model.

## 42.2 One execution spine

Inference, agents, orchestration, delegated work, background work, scheduled work, and dynamic extensions use canonical Execution semantics.

## 42.3 Full runtime inspectability

The platform can expose what is configured, effective, active, allowed, denied, degraded, and why.

## 42.4 Reconstructable AI decisions

Model-facing inputs, tool visibility, context selection, policy overlays, and profile revisions are attributable and reconstructable.

## 42.5 Strong authority boundaries

Parent-child authority, side effects, credentials, sandbox access, external providers, and temporary capabilities all remain governed.

## 42.6 Context efficiency without loss of evidence

Lazy instructions, spill, artifacts, compaction, and CE budgeting reduce context cost while canonical evidence remains durable.

## 42.7 Heterogeneous executor support

Intergrax can govern native agents and external agents/executors under one operating model.

## 42.8 Executable architecture invariants

The platform continuously verifies critical relationships rather than relying exclusively on documentation and tests.

## 42.9 Adaptive but controlled runtime

AHI can evolve versioned harness artifacts through evidence, governance, shadow, canary, verification, and rollback.

## 42.10 Safe runtime evolution

At the most advanced stage, the system can propose and temporarily activate missing capabilities without allowing generated behavior to escape authority or scope.

---

# 43. Strategic end state

The goal of this roadmap is not to maximize the number of agent-framework features.

The target is a governed execution operating layer capable of running and coordinating heterogeneous AI workloads while retaining:

- authority,
- evidence,
- explainability,
- reproducibility,
- isolation,
- recovery,
- context discipline,
- knowledge governance,
- and controlled adaptation.

The platform should be able to govern work performed by:

- direct model inference,
- Intergrax agents,
- external agents,
- orchestration graphs,
- background workers,
- retrieval systems,
- code/process runtimes,
- scheduled automation,
- event-triggered automation,
- and future dynamically proposed capabilities.

All of these must remain subordinate to the same canonical concepts:

```text
Execution
Authority
Budget
Context
Governance
Evidence
Recovery
Evaluation
```

That is the intended architecture convergence point of this roadmap.

---

# 44. Definition of Done for the complete program

The roadmap is considered fully implemented only when all of the following are true:

1. `ExecutionId` is canonical across the runtime.
2. Execution Tree is a real runtime structure.
3. Direct, agentic, orchestration, delegated, background, and scheduled work use the same Execution semantics.
4. `ApplicationEnvironmentProfile` remains the only environment composition authority.
5. Profile resolution is layered, deterministic, and explainable.
6. Every Run/Execution has immutable effective profile identity.
7. Operators can inspect configured and effective runtime state.
8. Important effective decisions can be explained from provenance.
9. Canonical model requests are reconstructable from evidence.
10. All canonical model calls pass through Context Engineering.
11. All canonical tool calls pass through ToolRuntime.
12. Meaningful side effects pass through fresh Governance enforcement.
13. Child authority cannot exceed parent authority.
14. Hierarchical budget propagation is enforced.
15. Cancellation works across Execution subtrees and provider boundaries.
16. Credentials use references/providers and are authority-scoped.
17. Sandbox/Execution Environment is unified across process/code/tool capabilities.
18. External agents are providers under canonical Execution semantics.
19. Background work does not introduce a second lifecycle authority.
20. External events enter through governed admission.
21. Large outputs can spill to durable artifacts without losing retrievability.
22. Context compaction preserves evidence lineage.
23. Capability and plugin registrations are reversible where dynamic.
24. Critical runtime invariants are executable and attributable.
25. Dynamic workflow proposals cannot bypass Nexus/Governance.
26. Capability Skills and Instruction Skills have distinct semantics.
27. Adaptive artifacts are versioned, governed, verified, and reversible.
28. Runtime extensions cannot self-expand authority and expire with their scope unless explicitly promoted.
29. SDK/API/ACP/MCP/host paths do not create private execution/policy/tool/context semantics.
30. Generated catalogs/dependency graphs are freshness-gated.
31. Major new architecture claims have bounded Scenario Proof evidence.
32. Documentation consistently distinguishes CURRENT from TARGET.
33. Redundant legacy/parallel authorities identified by this roadmap are removed or explicitly justified.

When these conditions are satisfied, Intergrax should behave as one coherent governed AI execution operating layer rather than as a collection of individually strong but operationally separate subsystems.
