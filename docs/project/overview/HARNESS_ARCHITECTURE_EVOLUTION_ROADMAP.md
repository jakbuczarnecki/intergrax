# Harness Architecture Evolution Roadmap

## Status and baseline

This document is the canonical cross-domain roadmap for the next stage of Intergrax harness architecture evolution. It defines what must remain unchanged, what must converge, what must be added, what must be removed or consolidated, the dependency order, the proof requirements, and the completion gates for the whole program.

It does **not** replace domain architecture documents. Domain documents remain authoritative for the semantics and contracts they own. This roadmap coordinates work across those domains and must not become a second semantic authority.

**Validated repository baseline:** `development @ 60e95a748d387702bfdd20443d0674caf495cb65` on 2026-09-02.

The status labels in this roadmap are deliberately conservative:

- **CURRENT** — an implementation or canonical contract exists and is a real foundation.
- **PARTIAL** — the foundation exists, but convergence, hardening, or migration remains.
- **GAP** — the capability is materially missing from the canonical runtime path.
- **TARGET** — target architecture is defined, but implementation is incomplete or not yet proven.
- **NEW** — a new capability should be added under an existing semantic owner or a clearly justified new owner.
- **CONSOLIDATE** — multiple surfaces or historical paths should converge without creating a new authority.

Before implementing any initiative, its CURRENT/PARTIAL/GAP classification must be revalidated against the then-current `development` HEAD. This roadmap must never be used to rebuild something that already exists.

---

# 1. Core architectural position

Intergrax should evolve toward one coherent governed execution operating layer without flattening domain semantics.

The following boundaries are intentional and must remain explicit:

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
Workspace != Settings != Credentials
Transport Identity != Runtime Identity
Model Context != Durable Evidence
```

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
Durable Runtime Facts / Checkpoint / Evidence
        ↓
Diagnostics + Evaluation
        ↓
Adaptive, Governed Evolution
```

The goal is not to create a universal god-object. The goal is to make the existing semantic domains compose, execute, inspect, recover, explain, test, and evolve as one platform.

---

# 2. Canonical ownership decisions

## 2.1 `ApplicationEnvironmentProfile` remains the only Tier-3 environment composition authority

Do **not** introduce a peer `HarnessProfile` or any other second environment authority.

Allowed additions around it include:

- `ProfileResolution` as typed resolution evidence,
- immutable effective profile snapshots/revisions,
- read-only effective runtime projections,
- profile provenance and semantic diff,
- inspection and explanation surfaces,
- typed overlays as **inputs** to resolution.

A read model, snapshot, overlay, projection, preset, or inspector must never become a competing source of configuration truth.

## 2.2 Domain semantic ownership remains distributed

- Unified Execution Runtime owns execution lifecycle and identity coordination.
- Nexus owns accepted orchestration topology execution decisions.
- Governance owns policy, authority, approval, and side-effect authorization decisions.
- Budget owns allowance, reservation, consumption, and release semantics.
- Tools own tool contracts and canonical tool execution semantics.
- Skills own capability-bundle semantics; instructional skills remain distinct.
- Context Engineering owns model-context assembly.
- Memory owns persistent memory semantics.
- RAG owns governed retrieval semantics.
- Observability/HOS records canonical platform facts and historical projections.
- DIAG interprets evidence; it does not create execution truth.
- Checkpoint/recovery owns durable restore state, not runtime identity authority.
- Platform Plugins own package/control-plane concerns, not Tool/Skill/RAG/Memory semantics.

## 2.3 No new subsystem without new semantic ownership

Before introducing a new architecture domain, answer:

> Is there a genuinely new responsibility that no existing domain should own?

If not, implement the behavior as a provider, plugin, policy, guard, strategy, hook, adapter, projection, preset, read model, or domain-owned extension.

## 2.4 Consumers depend on contracts, not concrete providers

Canonical dependency direction:

```text
Consumer
  ↓
Domain Contract / Service Definition
  ↓
Provider
```

A consumer must not depend directly on a concrete provider when a provider-neutral domain contract exists or should exist.

---

# 3. Mandatory architectural invariants

These invariants must become explicit in architecture, implementation, conformance tests, and runtime diagnostics where applicable.

1. **INV-1 — One environment composition authority**  
   `ApplicationEnvironmentProfile` is the only Tier-3 environment composition authority.

2. **INV-2 — Configured is not effective**  
   `CONFIGURED != EFFECTIVE`.

3. **INV-3 — Effective decisions are explainable**  
   Important effective values, permissions, capability availability, context decisions, and execution-strategy decisions have attributable provenance.

4. **INV-4 — Model-visible means reconstructable according to evidence policy**  
   `MODEL_VISIBLE => RECONSTRUCTABLE` through one of the defined reconstruction levels in Initiative D.

5. **INV-5 — Child authority cannot exceed parent authority**  
   `child_authority ⊆ parent_authority`.

6. **INV-6 — Proposal is not permission**  
   `PROPOSAL != PERMISSION`.

7. **INV-7 — Permission is not execution**  
   `PERMISSION != EXECUTION`.

8. **INV-8 — Topology identity is not runtime identity**  
   `NodeId != ExecutionId`.

9. **INV-9 — Agent is not execution**  
   `Agent != Execution`.

10. **INV-10 — Package/plugin is not domain capability**  
    `PluginPackage != Tool != Skill != Memory != RAG != Context`.

11. **INV-11 — Context compaction is not evidence deletion**  
    `MODEL_CONTEXT_COMPACTION != EVIDENCE_DELETION`.

12. **INV-12 — Observability records truth; it does not invent execution truth**.

13. **INV-13 — Runtime extensions cannot self-expand authority**.

14. **INV-14 — Temporary capability expires with its owner scope unless explicitly promoted**.

15. **INV-15 — Dynamic registration is reversible**.

16. **INV-16 — Meaningful side effects cross fresh governed authorization immediately before the effect**.

17. **INV-17 — Every canonical model call crosses Context Engineering**.

18. **INV-18 — Every canonical tool call crosses ToolRuntime**.

19. **INV-19 — Independently meaningful child work is admitted by Unified Execution Runtime**.

20. **INV-20 — Canonical runtime boundaries are transport-independent**.

21. **INV-21 — Required causal evidence precedes meaningful work**  
    If causal/audit evidence is required to establish a runtime boundary, recovery relation, or side-effect fence, it must be durably established before meaningful work begins. Failure to persist required evidence fails closed at that admission boundary.

22. **INV-22 — Profile overlays are inputs, not authorities**  
    Agent/Run/Execution overlays are typed deltas submitted to canonical profile resolution; they never become independent effective-profile authorities.

23. **INV-23 — Consumers depend on provider-neutral contracts**  
    Providers may vary; domain semantics do not.

24. **INV-24 — Tool and capability authority is monotonic**  
    Downstream scope may narrow effective permission but may not widen upstream authority.

25. **INV-25 — Activation is atomic from the runtime consumer perspective**  
    A failed multi-capability activation must not leave an accidental half-active composition; staged activation must commit or roll back according to the lifecycle contract.

26. **INV-26 — In-flight Executions are version-pinned**  
    Runtime reconfiguration must not silently change the provider/profile/schema semantics of an already admitted Execution unless an explicit migration/rebind contract allows it.

27. **INV-27 — Human interaction absence never implies approval**  
    If a required human interaction cannot be delivered, the runtime pauses, denies, or fails according to policy; it never auto-allows.

28. **INV-28 — Checkpoint is durable state, not identity authority**  
    Restoring from a checkpoint preserves canonical runtime identity semantics rather than minting a competing tree.

29. **INV-29 — Session/read models are projections where facts already exist canonically**  
    A projection may summarize or index state but must not silently become a second mutable source of truth.

30. **INV-30 — Security claims are bounded by a declared threat model and conformance evidence**  
    Tests demonstrate bounded properties; they do not justify absolute claims of mathematical impossibility.

---

# 4. Program status matrix

This matrix is the execution-control layer for the roadmap. It intentionally distinguishes new work from migration/hardening.

| ID | Initiative | Roadmap status | Canonical owner / primary existing area | Change type |
|---|---|---|---|---|
| A | Unified Execution Runtime convergence | PARTIAL / TARGET | Unified Execution Runtime / UEA | migration + hardening |
| B | Profile resolution and effective composition | PARTIAL | Applications / environment profile | consolidation + DX |
| C | Runtime inspection and explanation | GAP / PARTIAL | cross-domain read model over canonical facts | new read-model + DX |
| D | Reconstructable model execution | PARTIAL | CE + Observability/HOS + execution evidence | hardening |
| E | Context Engineering provider convergence | CURRENT / PARTIAL | Context Engineering | hardening + provider seam |
| F | Canonical ToolRuntime pipeline | CURRENT / PARTIAL | Tools / ToolRuntime / Governance boundary | convergence |
| G | Runtime credentials and secret references | PARTIAL | security/secrets/integration runtime | consolidation + provider seam |
| H | Execution sandbox and isolation | PARTIAL | execution environment / sandbox / security | convergence |
| I | Subagent and external-agent providers | PARTIAL / TARGET | delegation + UER + Nexus where orchestration applies | provider seam |
| J | Background Execution control | CURRENT / PARTIAL | Background Tasks + UER | migration + DX, **not new job runtime** |
| K | Verified external event intake | PARTIAL | application intake/integrations + UER | generalization + hardening |
| L | Artifacts, attachments, spill | PARTIAL | artifacts/storage + CE + tools | consolidation |
| M | Context compaction and retention | PARTIAL / TARGET | CE/UCL/evidence | implementation + hardening |
| N | Runtime invariant service | GAP / PARTIAL | domain-owned checks + diagnostics | new cross-domain registry, domain-owned rules |
| O | Dynamic orchestration proposals | TARGET | Orchestration/Nexus/Governance | new proposal path |
| P | Capability Skills + Instruction Skills | CURRENT / GAP | Skills + CE | preserve + add instructional type |
| Q | Reversible capability/plugin lifecycle | PARTIAL | Platform Plugins + domain registries | convergence |
| R | Governance UX and permission presets | CURRENT / PARTIAL | Governance/HITL/ToolRuntime/sandbox | DX + hardening |
| S | Workspace/settings/credentials separation | PARTIAL | application/workspace/settings/security | semantic cleanup |
| T | Feedback/goals/plans/work-state | PARTIAL / OPTIONAL | collaboration/evaluation/AHI | scenario-driven |
| U | Scheduling convergence | PARTIAL | scheduler/intake + UER | convergence |
| V | Process/filesystem/terminal/code providers | PARTIAL / OPTIONAL | tools/execution environment/providers | provider-first, scenario-driven |
| W | SDK/API/ACP/MCP/host convergence | PARTIAL | hosting/application boundaries + UER | convergence |
| X | Generated architecture/capability metadata | PARTIAL | docs/tooling/control plane | automation |
| Y | Documentation architecture simplification | PARTIAL | documentation canon | consolidation |
| Z | Security/trust/supply-chain hardening | PARTIAL | security/governance/plugins | hardening |
| AA | Memory and RAG hardening | CURRENT / PARTIAL | Memory + RAG + CE | hardening |
| AB | Adaptive Harness Intelligence expansion | CURRENT / PARTIAL | AHI + Evaluation/HOS | controlled expansion |
| AC | Governed Runtime Evolution | GAP / LATE TARGET | Plugins + sandbox + UER + Governance + AHI | strategic new capability |
| AD | Test and qualification infrastructure | PARTIAL | test support + qualification + Scenario Proofs | hardening |
| AE | Developer/operator experience | PARTIAL | CLI/docs/inspection | DX |
| AF | Checkpoint, durability, and recovery convergence | PARTIAL / TARGET | Checkpoint + UER + Observability + Reliability | **P0 migration/hardening** |
| AG | Runtime health, readiness, and atomic activation | GAP / PARTIAL | control-plane projection + provider lifecycle | new operational read-model + lifecycle hardening |
| AH | Controlled live composition reconfiguration | GAP / TARGET | profile resolution + plugin lifecycle + UER | later controlled reconfiguration |
| AI | Canonical Human Interaction seam | PARTIAL | Governance/HITL/host interaction | convergence |

The status matrix is not a substitute for implementation-specific audit. It is the planning baseline that prevents the roadmap from treating existing architecture as absent.

---

# 5. Initiative A — Unified Execution Runtime convergence

**Goal:** make the frozen execution model the actual universal runtime spine.

**CURRENT/PARTIAL:** Task/Run/Attempt/Event identity and agentic execution infrastructure exist; canonical `ExecutionId`, neutral Execution Boundary, full Execution Tree propagation, hierarchical budget, and some authority paths remain incomplete.

**Required work:**

1. Canonical `ExecutionId`, root Execution, `parent_execution_id`, and `RuntimeEvent.execution_id`.
2. Neutral Execution Boundary supporting inference, agentic, and orchestration strategies without changing lifecycle semantics.
3. Real Execution Tree with lineage queries and subtree cancellation.
4. Explicit child-Execution admission rules; LLM calls, tool calls, nodes, and function calls are not automatically Executions.
5. Hierarchical authority propagation and monotonic narrowing.
6. Hierarchical budgets for tokens, money, tool calls, child executions, concurrency, wall clock, and agent-loop/step counts.
7. Canonical cancellation across local, remote, background, model, and tool work.
8. Pause/resume/HITL as lifecycle state rather than hidden callback state.
9. Retry ownership taxonomy: provider/tool retry, Execution retry generation, transport redelivery, whole-Run retry.
10. Canonical Execution Environment and neutral Execution Result ABI.
11. Distributed worker admission must preserve runtime identity; transport redelivery must not create new logical execution identity by itself.

**Acceptance:** representative inference, agentic, orchestration, delegated, and background paths use the same identity/lifecycle/evidence spine.

---

# 6. Initiative B — Profile resolution and effective composition

**Goal:** keep one environment authority while making composition layered, explainable, immutable per Execution, and easy to inspect.

**Required work:**

1. Canonical resolution order: platform defaults → product/distribution → application → agent delta → Run/Task delta → Execution delta.
2. Every overlay is a typed delta input; only `ProfileResolution` produces the effective result.
3. `ProfileResolution` records input layers, revisions, overrides, rejected overrides, authority clamps, dependency failures, warnings, and degraded capabilities.
4. Separate configured/requested from effective state.
5. Persist or durably reference effective profile revision/fingerprint per Run/Execution.
6. Semantic profile diff across tools, models, policies, sandbox, CE budgets, orchestration, and providers.
7. Capability dependency validation such as Skill → Tool → Integration → Credential → Provider.
8. Required dependency missing = fail before execution; optional dependency missing = explicit degraded state.
9. Ergonomic presets expand to canonical policy/profile semantics and grant no new authority.
10. Audit nested canonical profile representation versus legacy flat/wire compatibility and remove unjustified compatibility surfaces.

**Acceptance:** no second composition authority exists; historical Executions retain immutable meaning after configuration changes.

---

# 7. Initiative C — Runtime inspection and explanation

**Goal:** expose actual live runtime state from canonical facts.

Create one programmatic inspection read model used by CLI, web, SDK, diagnostics, and future hosts.

It must support:

- environment configured/effective state and profile revision,
- agent model/tools/skills/context/memory/authority/budgets,
- Execution identity/tree/state/strategy/children/model calls/tool calls/approvals/evidence,
- tool states: registered, available, visible, selectable, executable, authorized/denied, provider, risk, reason,
- skill manifest/version/resolution/provenance,
- context contributions, cost, inclusion/exclusion reasons,
- capability dependency graph,
- runtime health/readiness from Initiative AG.

Explanation surfaces must answer at minimum:

- why is this tool available or unavailable?
- why was approval required or denied?
- why did this profile value win?
- why was context included or excluded?
- why was this execution strategy selected?

Inspection is a projection; it must never mint canonical identities or invent facts.

---

# 8. Initiative D — Reconstructable model execution

**Goal:** make model-facing requests attributable and reproducible within evidence/retention policy.

Add canonical ModelRequest identity linked to Execution, exact provider/model/route settings, prompt/system revision, tool-schema-set revision, profile revision, context assembly revision, attachments/artifacts, and relevant generation settings.

Define three reconstruction levels:

1. **Exact reconstruction** — complete request payload can be reconstructed byte-/contract-equivalently where policy permits retention.
2. **Referential reconstruction** — immutable authorized artifact/references allow reconstruction without duplicating the payload in every evidence record.
3. **Structural provenance** — when privacy/retention rules require deletion or redaction, preserve identity, hash/fingerprint, source/revision, redaction/retention event, and enough structure to prove lineage without falsely claiming exact reconstruction.

Compaction summaries must reference their source evidence range. Model-visible and user-visible projections must be separable. Session/conversation read models should derive from canonical evidence where facts already exist.

**Acceptance:** canonical paths have CI reconstruction tests and never claim exact reconstructability when retention policy makes it impossible.

---

# 9. Initiative E — Context Engineering convergence

Preserve the existing CE pipeline:

```text
COLLECT → NORMALIZE → SCORE → FILTER → RANK → BUDGET
→ COMPRESS/DEGRADE → FORMAT → VALIDATE → EMIT
```

Required hardening:

- formal provider contract for history, workspace instructions, files, references, Memory, RAG, runtime state, tool results, policy context, time, and instructional skills,
- deterministic provider registration/activation/deactivation,
- fragment provenance and lifetime: step/turn/Execution/Run/Session/durable,
- replacement/supersession semantics,
- controlled refresh of workspace instructions,
- file/session/artifact references through CE rather than host-private injection,
- lazy loading of instructions/resources,
- token-cost attribution by source,
- context diagnostics and exclusion reasons,
- KV-cache-aware stable-prefix optimization where provider behavior makes it useful.

No model path may bypass CE.

---

# 10. Initiative F — Canonical ToolRuntime pipeline

Preserve ToolContract/ToolRuntime and formalize one official path:

```text
resolve → validate → effective-permission intersection → authorize
→ pre-execute guards/policies → schedule → execute
→ normalize → post-process → persist authoritative result
→ publish user/model projections
```

Required hardening:

- canonical authoritative tool outcome separate from presentation,
- stable typed input/output ABI,
- timeout/cancellation contracts,
- concurrency classification such as parallel-safe/exclusive,
- nested tool-call lineage,
- resource/cost attribution,
- small guards for repeat-call/runaway/deadline behavior,
- structured error identity (`error_id`, class, origin, retryability, user-safe message, diagnostic reference, cause),
- removal/routing of private tool loops.

Effective permission must remain a monotonic intersection of upstream host/profile/skill/policy/modality/invoker constraints; downstream logic may narrow, never widen.

---

# 11. Initiative G — Runtime credentials and secret references

Introduce/consolidate a provider-neutral `CredentialRef` model so configuration normally names secrets rather than storing values.

Required:

- local development and enterprise/cloud secret providers behind one contract,
- late per-operation/per-Execution resolution,
- rotation without rewriting unrelated config,
- execution-scoped credential exposure,
- interactive authorization flows,
- no secret value in model context by default,
- credential-access evidence that never logs the secret value.

---

# 12. Initiative H — Unified Execution sandbox and isolation

Provide a simple contract with user-facing modes such as `READ_ONLY`, `WORKSPACE_WRITE`, and `FULL_ACCESS`, backed by pluggable local/container/remote/microVM/enterprise providers.

The contract covers filesystem, network, process, and credential exposure. Denied operations may request narrowly scoped one-shot escalation through Governance/Human Interaction. Provider documentation must distinguish confinement from hard isolation and declare limitations honestly.

Do not require every OS backend merely for feature parity; implementations are scenario- and deployment-driven.

---

# 13. Initiative I — Subagent and external-agent providers

Define a provider-neutral delegation seam. Native and external agents remain providers under Intergrax Execution/Governance/Evidence semantics.

Required:

- native local/remote provider,
- controlled forked-context child,
- external protocol/service adapters where useful,
- child Execution admission for independently meaningful delegated work,
- list/status/follow-up/interrupt/cancel/completion controls,
- parent authority/budget inheritance,
- provider output normalization into canonical evidence.

External provider telemetry is secondary; it must not become a second execution lifecycle authority.

---

# 14. Initiative J — Background Execution control

**Important:** do not build a second job runtime. Existing Background Tasks architecture is a real foundation.

The work is to converge it onto canonical UER semantics and add ergonomic control:

- preserve existing TaskRegistry/WorkerRuntime/provider transport architecture,
- migrate transport/redelivery identity to canonical Attempt/Execution semantics,
- expose start/list/status/wait/cancel/completion as a control projection over Execution,
- retain ownership fencing,
- make durable background work use canonical checkpoint/recovery,
- define detached-child lifetime and authority explicitly,
- completion routing must not create hidden state authority.

---

# 15. Initiative K — Verified external event intake

Generalize a provider-neutral event envelope with source authentication, normalization, tenant/workspace/principal resolution, Governance, idempotency/deduplication, durable delivery/retry where required, and event→Task/Run/Execution admission.

Required ordering for reliable flows:

```text
verified delivery
→ durable delivery/idempotency decision
→ authority/workspace resolution
→ required causal evidence
→ work admission
→ meaningful work
```

Recovery must prevent a crash/redelivery from silently duplicating already-committed meaningful side effects.

---

# 16. Initiative L — Durable artifacts, attachments, and spill

Create one coherent artifact model with stable identity, ownership/scope, provenance, version/content identity, retention, lineage, and controlled retrieval.

Large tool/runtime output should support:

```text
full payload → durable artifact
model context → bounded preview + authorized locator
```

Artifact existence is independent of model context. CE decides the model-visible projection. Evidence may reference large artifacts instead of duplicating payloads.

---

# 17. Initiative M — Context compaction and retention

Implement model/provider-aware automatic compaction and explicit compaction controls.

Prefer spill/pruning of large recoverable outputs before semantic summarization where appropriate. Summaries retain source lineage. Critical task/policy/authority fragments are protected. Important facts may be pinned. Compaction quality is evaluated.

Canonical audit evidence is unaffected by context reduction except where independent privacy/retention policy explicitly removes or redacts evidence.

---

# 18. Initiative N — Runtime invariant service

Domains define the invariants they semantically own; a common registry/runner executes, filters, and reports them.

Initial high-value invariants include:

```text
MODEL_VISIBLE_RECONSTRUCTABLE_WITH_DECLARED_LEVEL
CHILD_AUTHORITY_NOT_GREATER_THAN_PARENT
SIDE_EFFECT_HAS_FRESH_AUTHORIZATION
REQUIRED_CAUSAL_EVIDENCE_PRECEDES_MEANINGFUL_WORK
TOOL_RESULT_HAS_TOOL_CALL
EXECUTION_EVENT_HAS_EXECUTION_ID
PROFILE_REVISION_EXISTS
BACKGROUND_CHILD_HAS_OWNER
PROFILE_OVERLAY_IS_NOT_AUTHORITY
DYNAMIC_REGISTRATION_HAS_OWNER_AND_TEARDOWN
```

Failures include owner attribution, severity, evidence references, and may be reused in CI/conformance where practical.

---

# 19. Initiative O — Dynamic orchestration proposals

Models/planners may produce typed `WorkflowProposal` artifacts. Validate schema, capabilities, authority, budget, cycles, depth, fan-out, concurrency, and provider availability.

Proposal is not accepted topology. Only an accepted `OrchestrationDefinition` enters Nexus. Model-authored scripts or external planners are proposal providers, not execution authorities.

---

# 20. Initiative P — Capability Skills and Instruction Skills

Preserve existing Capability Skills and their tool/prompt/policy/dependency/risk semantics.

Harden version identity and preserve resolved skill provenance. Add a distinct lightweight `InstructionSkill` with id/version/description/instruction source/activation policy/token estimate/provenance. Expose compact catalogs and lazy-load full instructions.

Instruction Skills may not grant tool access or widen authority.

---

# 21. Initiative Q — Reversible capability/plugin lifecycle

Align extension surfaces where applicable on:

```text
discover → validate → register → stage → activate → inspect
→ deactivate → unregister
```

Dynamic registration returns deterministic teardown handles. Registrations are owner-scoped (platform/application/agent/Execution). Execution-scoped registrations disappear when the owner ends.

Define replace/version coexistence/upgrade/rollback/dependency-failure semantics. In-flight Executions remain pinned to the admitted provider/version unless explicit migration is allowed.

`CapabilityDescriptor` control-plane metadata should include:

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
stability: experimental|preview|stable|deprecated
compatibility constraints
```

Platform Plugins remain the package/control plane; they do not absorb domain semantics.

---

# 22. Initiative R — Governance UX and permission presets

Preserve Meaningful Side Effect Authorization and fresh evaluation immediately before effects. Fresh DENY cannot be overridden by an earlier approval.

Add understandable presets such as observe/workspace/controlled/trusted/full that only expand into canonical Governance, sandbox, ToolRuntime, and authority configuration.

Provide explainable approval/deny reasons, normalized side-effect previews, one-shot consumable grants, and effective authority snapshots per Execution.

---

# 23. Initiative S — Workspace, settings, and credentials separation

Workspace = runtime/product scope and resources. Settings = user/operator preferences. Credentials = governed secret resources. `ApplicationEnvironmentProfile` may reference/configure these systems but must not become their runtime storage authority.

---

# 24. Initiative T — Feedback, goals, plans, and work state

Highest priority is a canonical feedback bridge:

```text
human feedback → evaluation evidence → outcome signal → optional adaptive input
```

Goals/plans/todos remain optional scenario-driven collaboration state. They never grant execution permission or become a hidden orchestration authority.

---

# 25. Initiative U — Scheduling convergence

Represent scheduled intent separately from actual Execution. At trigger time resolve **current** authority, policy, credentials, profile, workspace, and capability availability. Durable schedules support recurrence/cancellation as required. Creation/change/trigger/cancel and resulting Task/Execution identities are evidence-backed.

---

# 26. Initiative V — Provider-neutral process/filesystem/terminal/code runtimes

Where scenarios require them, define provider-neutral seams such as ProcessProvider, FilesystemProvider, TerminalProvider, and CodeRuntimeProvider. They consume canonical Execution Environment, sandbox, credentials, ToolRuntime/Governance, and evidence.

Prefer external/remote mature providers where economically sensible. Build local implementations only when scenarios establish strategic value. Do not enter a feature-parity race for specialized coding-agent functionality.

---

# 27. Initiative W — SDK/API/ACP/MCP/host convergence

Python SDK, REST/HTTP, ACP, MCP, CLI, Slack/chat, web, and future hosts are adapters over the same canonical runtime boundaries.

No host-specific private policy engine, private tool runtime, private execution identity, or private CE path is allowed. Hosts declare capabilities; qualification evidence is scoped to host/profile/runtime combinations.

---

# 28. Initiative X — Generated architecture and capability metadata

Generate rather than hand-duplicate:

- capability catalog,
- tool catalog,
- skill catalog,
- provider catalog,
- configuration catalog,
- invariant catalog,
- module/dependency graph,
- extension/capability graph,
- transport/API schemas and client types where the canonical technology permits generation.

CI freshness gates should catch stale generated artifacts.

---

# 29. Initiative Y — Documentation architecture simplification

Maintain one glossary and one identity vocabulary. Each domain answers one primary question. CURRENT precedes TARGET. Canonical docs link to implementation and tests/proofs where practical. Public docs avoid unnecessary internal acronyms. Mechanically verifiable documentation receives freshness gates.

This roadmap must be linked from the project documentation navigation and the general roadmap so it is discoverable as the canonical cross-domain architecture-evolution program.

---

# 30. Initiative Z — Security, trust, and supply-chain hardening

Make providers/plugins/extensions explicit trust-boundary objects with risk/trust/stability/qualification/provenance metadata.

Harden credential isolation, network destination policy, filesystem mounts, process spawning, high-risk activation approval, extension sandbox testing, supply-chain evidence, and permission-delta review before activation.

Security acceptance language must be bounded: canonical paths must fail closed under the declared threat model and pass defined adversarial/conformance suites; do not claim universal impossibility from finite testing.

---

# 31. Initiative AA — Memory and RAG hardening

Preserve `Memory != RAG != Context Engineering`.

Required hardening:

- tenant/user/session/task/org scope authority,
- explicit tool-result persistence policy so outputs do not silently become LTM,
- memory write and consolidation provenance,
- deletion/tombstone evidence,
- RAG scope enforcement before retrieval/context inclusion,
- canonical RetrievalHit ABI,
- retrieval strategy evidence,
- effective query/filter reconstruction where required,
- CE remains the only owner of final model-context inclusion.

Acceptance claims should be bounded to canonical qualified paths and threat-model tests rather than absolute language.

---

# 32. Initiative AB — Adaptive Harness Intelligence expansion

Broaden the governed adaptive artifact taxonomy where evidence justifies it, for example routing, execution-strategy, context, RAG, tool-selection, policy recommendations, Instruction Skills, WorkflowDefinitions, provider recommendations, and runtime-extension candidates.

Each artifact class defines its maximum authority: observe, recommend, human-gated apply, or bounded automatic apply. Preserve versioning, shadow, canary, verification, keep/rollback, and no self-expansion of the adaptive authority envelope.

---

# 33. Initiative AC — Governed Runtime Evolution

This remains a **late-stage** initiative and must not begin before Execution, checkpoint/recovery, sandbox, credentials, runtime invariants, reversible registration, effective composition, and evidence foundations are proven.

Lifecycle:

```text
CapabilityGap
→ RuntimeExtensionProposal
→ structural/static validation
→ sandbox build/test
→ contract + adversarial tests
→ governance
→ shadow
→ Execution-scoped activation
→ use/evaluation
→ expiry | canary | promotion | rollback
```

A model-safe `RuntimeCapabilityView` must exist before `CapabilityGap`: the model may see only authorized capability names/descriptions, limitations, and resolvable gaps; it must not receive secrets, hidden providers, cross-tenant state, or security-sensitive control-plane metadata.

Persistent installation/promotion is a distinct governed workflow. Runtime extension cannot widen its own scope, authority, or lifetime.

---

# 34. Initiative AD — Test and qualification infrastructure

Provide reusable deterministic construction of representative runtime compositions, scripted model fault/mocking, model/tool replay, run/session/effective-profile snapshots, reconstruction tests, registration-leak tests, cancellation races, parallel-tool tests, sandbox matrices, provider conformance suites, workflow adversarial tests, and hostile runtime-extension tests.

Scenario Proofs remain the highest-level falsification/evidence layer; unit and integration tests do not substitute for platform claims.

---

# 35. Initiative AE — Developer/operator experience

A developer should be able to start a small safe governed runtime without reading every architecture document.

Provide sensible defaults and canonical user-facing capabilities for run, inspect, explain, doctor, profile diff, replay, and proof execution. `doctor` should detect missing providers, invalid credential refs, unsupported sandbox, capability dependency failures, stale configuration/generated metadata, provider health issues, and host capability mismatches.

Document one boot mental model:

```text
load composition
→ resolve deltas/dependencies
→ validate effective profile
→ stage providers
→ activate atomically
→ expose runtime health/inspection
→ admit work
```

---

# 36. Initiative AF — Checkpoint, durability, and recovery convergence

**Priority:** P0.

**CURRENT/PARTIAL:** UER already defines checkpoint/recovery ownership and target semantics; current checkpoint state does not yet preserve the complete canonical Execution Tree.

**Goal:** make pause/resume, worker crash, host restart, background work, HITL, and distributed execution recover through one Run-scoped durable state model without becoming an identity authority.

Checkpoint/recovery must preserve as applicable:

- current Attempt,
- root Execution and Execution Tree,
- per-Execution lifecycle state,
- Execution retry generation/index,
- Nexus orchestration state,
- agent/UAEP cursors where applicable,
- pending HITL/HumanInteraction state,
- budget reservations and relevant consumption state,
- side-effect fences/idempotency outcome state,
- required causal evidence references,
- transport cursors/leases/delivery relations,
- effective profile/provider version pins,
- artifact/context references required for continuation.

Required semantics:

1. Restore/resume preserves canonical identity where the same logical Attempt/Execution continues.
2. Whole-Run retry mints a new Attempt and new runtime Execution instances according to UER rules.
3. Transport redelivery alone does not redefine runtime identity.
4. Checkpoint persistence is not a second execution tree.
5. Side-effect retry/recovery must re-evaluate the correct authorization/idempotency boundary and must not silently duplicate committed effects.
6. Required durability/evidence write failures fail closed where proceeding would make causal or side-effect correctness unverifiable.

**Acceptance:** a forced-crash Scenario Proof restores the canonical tree and either resumes safely or fails safely without duplicate unauthorized meaningful side effects.

---

# 37. Initiative AG — Runtime health, readiness, and atomic activation

**Goal:** distinguish capability existence from real operational readiness.

Define a read-model state such as:

```text
READY
DEGRADED
UNAVAILABLE
FAILED
DRAINING
```

Health/readiness is a projection over canonical provider/domain facts, not a new business authority.

Required work:

- provider/capability health contributors,
- dependency-aware effective readiness,
- explicit degraded reason and remediation hints,
- `inspect`/`doctor` integration,
- health impact on admission according to required/optional dependency semantics,
- staged multi-capability activation,
- atomic commit of the effective activation set from the consumer perspective,
- deterministic rollback if staging/activation fails,
- draining semantics where a provider is being replaced but in-flight pinned Executions still depend on the old version.

Example:

```text
jira.search
registered: yes
authorized: yes
provider: jira-v2
provider state: UNAVAILABLE
effective executable: no
reason: credential authorization expired
```

---

# 38. Initiative AH — Controlled live composition reconfiguration

**Priority:** P2/P3 after B, Q, AG, and AF foundations.

Support controlled runtime changes without reboot where safe and useful:

```text
patch request
→ validate schema/dependencies
→ authority/governance check
→ build candidate ProfileResolution
→ stage providers/registrations
→ atomic activation
→ mint new effective profile revision
→ new Executions use new revision
→ in-flight Executions remain pinned
→ drain old version
→ rollback if required
```

Eligible changes may include model routes, context providers, instruction sources, selected capability providers, and permission/profile configuration when allowed. Live patching is not an alternate composition authority; it submits a typed delta into canonical resolution.

---

# 39. Initiative AI — Canonical Human Interaction seam

Generalize human collaboration beyond approval callbacks through a provider-neutral interaction request/result contract covering:

- approval,
- question/clarification,
- structured choice,
- credential authorization,
- review/feedback.

Hosts such as CLI, web, Slack, API, or future channels provide interaction adapters. Governance owns approval semantics; UER owns lifecycle consequences such as waiting/resume; the interaction seam owns delivery/response correlation only.

If required interaction has no authorized responder/provider, the runtime follows policy to pause, deny, timeout, or fail — never auto-allow.

---

# 40. Consolidation and removal program

Audit and remove or formally contain:

1. unjustified legacy flat environment-profile compatibility,
2. legacy ToolBase/tool aliases that compete with ToolContract/ToolRuntime,
3. private agent tool loops,
4. competing execution identities used as substitutes for `ExecutionId`,
5. duplicate registries without distinct semantic ownership,
6. duplicate plugin discovery/control-plane paths,
7. duplicate config/profile resolution,
8. duplicate canonical observability/evidence paths,
9. retry engines with ambiguous ownership,
10. session/read-state authorities that duplicate canonical persisted facts,
11. host-specific policy/context/execution semantics,
12. dynamic registration paths without owner/lifetime/teardown.

Every compatibility surface retained after audit must identify a real consumer and removal condition.

---

# 41. Implementation order

## P0 — Canonical runtime spine and durability

Complete first:

1. canonical `ExecutionId`, root and parent relationships,
2. Execution Tree and `RuntimeEvent.execution_id`,
3. neutral Execution Boundary and neutral result ABI,
4. child authority propagation and monotonic permission rules,
5. hierarchical budget including agent-loop/step limits,
6. subtree cancellation and pause/resume semantics,
7. AF checkpoint/durability/recovery convergence,
8. required-causal-evidence-before-meaningful-work invariant,
9. configured vs effective profile distinction,
10. typed profile overlays + `ProfileResolution`,
11. immutable effective profile/provider-version pins,
12. ModelRequest identity and reconstruction levels,
13. canonical Runtime Inspection API foundation,
14. runtime invariant service foundation,
15. canonical Human Interaction contract foundation for pause/HITL paths,
16. removal/containment of competing execution/composition authorities on canonical paths.

### P0 exit gate

Representative inference, agentic, orchestration, background, delegated, pause/resume, and crash/recovery flows use the same identity/profile/evidence/checkpoint semantics. Required causal evidence failures fail closed at their declared boundaries.

## P1 — Composition, safety, inspection, activation

1. profile layering, provenance, diff, dependency validation,
2. required/optional dependency handling,
3. environment/agent/execution/tool/skill/context inspection,
4. explain tool/profile/context decisions,
5. capability dependency graph,
6. runtime credential references/providers,
7. unified ExecutionSandbox,
8. Governance permission presets and interaction UX,
9. reversible registration handles,
10. `CapabilityDescriptor` including stability/compatibility,
11. AG runtime health/readiness and atomic activation,
12. generated capability/config/provider/invariant catalogs,
13. critical runtime invariants and conformance tests.

### P1 exit gate

An operator can determine what was configured, what is effective, what is healthy, what is allowed, why something is denied/degraded, and which exact profile/authority/provider revision governs an Execution.

## P2 — Runtime capability power and efficiency

1. ContextProvider seam and attributable workspace/file/session/artifact references,
2. lazy Instruction Skills,
3. context cost attribution and diagnostics,
4. KV-cache-aware optimizations,
5. artifacts/spill and compaction,
6. ToolRuntime structured errors/concurrency/timeout/cancellation hardening,
7. SubagentProvider and external-agent adapters,
8. continuable child work,
9. background Execution control surface,
10. verified external event intake,
11. scheduling convergence,
12. SDK/API/ACP/MCP/host convergence,
13. provider-neutral process/filesystem/terminal/code seams where scenarios require them,
14. AH controlled live composition reconfiguration for safe eligible changes.

### P2 exit gate

Heterogeneous providers/agents/capabilities can be composed under one Execution/Governance/Evidence model, and controlled reconfiguration does not mutate in-flight semantics.

## P3 — Dynamic orchestration and adaptive expansion

1. WorkflowProposal and validation,
2. accepted dynamic OrchestrationDefinition through Nexus,
3. feedback→Evaluation/AHI bridge,
4. adaptive artifact taxonomy and class-specific authority,
5. shadow/canary/verification for new artifact classes,
6. richer runtime invariants,
7. model-safe RuntimeCapabilityView,
8. Scenario Proofs for dynamic topology, reconfiguration, recovery, and provider health.

## P4 — Governed Runtime Evolution

Only after P0–P3 are proven:

1. CapabilityGap,
2. RuntimeExtensionProposal,
3. static validation,
4. sandbox build/test,
5. contract and hostile-code tests,
6. shadow execution,
7. governance decision,
8. Execution-scoped activation,
9. expiry,
10. canary,
11. verification,
12. promotion workflow,
13. rollback,
14. full lifecycle evidence,
15. bounded adversarial Scenario Proof.

---

# 42. Dependency map

```text
ExecutionId / Execution Tree
        ↓
Authority + Budget + Required Causal Evidence
        ↓
Checkpoint / Recovery
        ↓
Background / Delegation / Distributed / Dynamic Work
```

```text
Typed profile deltas
        ↓
ProfileResolution
        ↓
Effective Snapshot + Version Pin
        ↓
Inspection / Health / Atomic Activation
        ↓
Controlled Live Reconfiguration
```

```text
ModelRequest Evidence
        ↓
Exact | Referential | Structural Reconstruction
        ↓
Diagnostics / Evaluation / Replay
```

```text
Artifacts + Spill
        ↓
Compaction
        ↓
Context Efficiency without Evidence Confusion
```

```text
Sandbox + Credentials + Invariants + Checkpoint
+ Reversible Registration + Human Interaction
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

# 43. Explicit anti-goals

Do not build unless future evidence overturns the decision:

1. a second composition authority beside `ApplicationEnvironmentProfile`,
2. a universal plugin base that erases domain semantics,
3. a second generic Job runtime beside UER,
4. a dynamic-workflow execution engine parallel to Nexus,
5. a private Session runtime substituting for Memory/Execution/Observability facts,
6. automatic global installation of model-generated code,
7. runtime extensions that widen their own authority,
8. a local feature-parity race for terminal/LSP/code runtimes when providers suffice,
9. new architecture domains for simple guards/policies,
10. context compaction that implies audit deletion,
11. host-specific ToolRuntime/Governance/CE/Execution semantics,
12. RAG/Memory directly injecting model context outside CE,
13. plan/goal/todo state granting authority,
14. telemetry sinks becoming canonical truth,
15. profile overlays becoming independent authorities,
16. checkpoints minting a competing execution tree,
17. health/readiness projections deciding business policy,
18. live reconfiguration silently rebinding in-flight Executions.

---

# 44. Scenario Proof program

Each major cross-domain claim must have bounded adversarial proof before strong maturity claims.

1. **Execution identity and lineage** — root/children/retries/cancellation/evidence reconstruction.
2. **Effective composition explainability** — layered resolution, rejected override, authority clamp, dependency degradation, tool-access explanation.
3. **Model-request reconstruction** — demonstrate exact and referential reconstruction plus a privacy-retention structural-provenance case.
4. **Governed external delegation** — external provider under inherited authority/budget and canonical evidence.
5. **Large-output artifact/spill** — bounded model context with retrievable full artifact.
6. **Runtime invariant violation** — intentionally violate a critical invariant and show detection/attribution/evidence.
7. **External event to governed work** — authenticate, deduplicate, establish causal evidence, admit, execute, and recover safely.
8. **Crash / Resume / Side-effect Safety** — checkpoint a nontrivial Execution Tree, force crash, restore, preserve causal lineage, and prove no duplicate unauthorized meaningful effect under the tested fault model.
9. **Credential + Sandbox Escalation** — start read-only with no secret exposure, request a privileged effect, obtain one-shot authorized interaction, consume scoped grant, perform effect, and return to prior authority.
10. **Atomic Activation / Provider Replacement** — fail activation mid-stage and prove rollback; then replace provider version while old Executions remain pinned and drain safely.
11. **Governed Runtime Evolution** — late-stage capability gap → proposal → sandbox/tests → governance → shadow → bounded activation → evaluation → expiry/promotion/rollback.

Proof language must state the tested threat/failure model and never convert bounded evidence into an absolute security claim.

---

# 45. Target user and operator experience

A developer should be able to express a high-level environment without manually wiring every subsystem:

```yaml
profile: governed-workspace-agent
model: primary
permissions: workspace
memory: standard
knowledge: enabled
sandbox: workspace-write
```

The platform resolves this into canonical domain contracts. The operator can then ask:

```text
What is configured?
What is effective?
What is healthy or degraded?
Why is this tool unavailable?
Why did this context fragment enter the model?
Why was it excluded?
Which profile value won?
Which policy required approval?
What children did this Execution spawn?
Which credential references were used?
Which provider/version is this Execution pinned to?
Which meaningful side effects occurred?
Can this model request be reconstructed exactly, referentially, or only structurally?
What changed between profile revisions?
Which runtime invariant failed?
Can this Execution be resumed safely after a crash?
```

Answers come from canonical state/evidence and read-model projections, not best-effort log interpretation.

---

# 46. Target platform architecture

```text
USER / APPLICATION / VERIFIED EVENT / SCHEDULE
                 │
                 ▼
       canonical intake/admission
                 │
                 ▼
   ApplicationEnvironmentProfile
        + typed profile deltas
                 │
          ProfileResolution
                 │
                 ▼
       Effective Runtime View
       + Health / Readiness
                 │
                 ▼
        Execution Boundary
                 │
      ┌──────────┼──────────┐
      │          │          │
  inference   agentic   orchestration
                            │
                           Nexus
                            │
                     child Executions
                 │
                 ▼
             Execution
                 │
   ┌─────────────┼──────────────────────────────┐
   ▼             ▼              ▼              ▼
  CE           Tools          Memory/RAG     Delegation
                 │                              │
             Governance                    child Exec
                 │
       Human Interaction when required
                 │
   └─────────────┴──────────────┬───────────────┘
                                ▼
                  Required causal evidence
                                │
                       Runtime Events / HOS
                                │
                Checkpoint / durable recovery state
                                │
                                ▼
                 Evidence / DIAG / Evaluation
                                │
                                ▼
                               AHI
                                │
                                ▼
                    Governed Runtime Evolution
```

Cross-cutting control plane:

```text
Platform Plugins / Capability Metadata
Profile Resolution / Version Pinning
Inspection / Explanation / Health
Runtime Invariants
Credentials
Sandbox / Execution Environment
Artifacts / Spill
Atomic Activation / Draining
Generated Catalogs / Schemas
```

---

# 47. Expected outcome

After full completion Intergrax should provide simultaneously:

- simpler composition without weaker semantic boundaries,
- one execution spine across direct, agentic, orchestration, delegated, background, scheduled, and event-triggered work,
- durable crash/recovery semantics with canonical identity preservation,
- inspectable configured/effective/healthy runtime state,
- explainable effective decisions,
- declared-level reconstruction of model-facing requests,
- strong hierarchical authority and budget boundaries,
- unified credentials/sandbox/human-interaction safety surfaces,
- context efficiency through lazy instructions, artifacts, spill, and compaction without confusing model context with evidence retention,
- heterogeneous executor/provider support,
- executable runtime invariants,
- atomic activation and version-pinned in-flight work,
- controlled live reconfiguration,
- governed adaptive change and late-stage bounded runtime evolution.

---

# 48. Definition of Done for the complete program

The program is complete only when all of the following are true on qualified canonical paths:

1. `ExecutionId` is canonical across runtime work.
2. Execution Tree is a real runtime structure.
3. Direct, agentic, orchestration, delegated, background, scheduled, and event-admitted work use the same Execution semantics.
4. `ApplicationEnvironmentProfile` remains the only environment composition authority.
5. Agent/Run/Execution overlays are typed deltas, not authorities.
6. Profile resolution is layered, deterministic, attributable, and dependency-aware.
7. Every admitted Run/Execution has immutable effective profile/provider-version identity.
8. Operators can inspect configured, effective, healthy, degraded, and active state.
9. Important effective decisions are explainable from provenance.
10. Canonical model requests expose their declared reconstruction level and satisfy it.
11. All canonical model calls pass through Context Engineering.
12. All canonical tool calls pass through ToolRuntime.
13. Meaningful side effects pass through fresh Governance enforcement.
14. Child authority cannot exceed parent authority.
15. Tool/capability permission intersection is monotonic.
16. Hierarchical budget propagation includes bounded agent-loop/step behavior.
17. Cancellation works across Execution subtrees/provider boundaries.
18. Checkpoint/recovery preserves canonical identity and required state for supported recovery scenarios.
19. Required causal evidence is persisted before meaningful work at declared fail-closed boundaries.
20. Side-effect recovery/retry uses explicit idempotency/fence/authorization semantics.
21. Credentials use references/providers and are authority-scoped.
22. Sandbox/Execution Environment is unified across process/code/tool capabilities.
23. Human interaction uses a canonical provider-neutral seam and never auto-allows on responder absence.
24. External agents remain providers under canonical Execution semantics.
25. Background work does not create a second lifecycle authority.
26. External events enter through governed, idempotent admission where configured.
27. Large outputs can spill to durable artifacts with provenance and controlled retrieval.
28. Context compaction preserves lineage and does not imply evidence deletion.
29. Dynamic registrations are reversible, scoped, and version-aware.
30. Multi-capability activation is staged/atomic and can roll back safely.
31. In-flight Executions remain pinned across provider/profile replacement unless explicitly migrated.
32. Critical runtime invariants are executable and attributable.
33. Dynamic workflow proposals cannot bypass Nexus/Governance.
34. Capability Skills and Instruction Skills remain semantically distinct.
35. Adaptive artifacts are versioned, governed, verified, and reversible.
36. Runtime extensions cannot self-expand authority and expire with scope unless explicitly promoted.
37. Model-safe capability introspection reveals only authorized runtime information.
38. SDK/API/ACP/MCP/host paths do not create private execution/policy/tool/context semantics.
39. Generated catalogs, dependency graphs, and applicable API/client schemas are freshness-gated.
40. Major new architecture claims have bounded Scenario Proof evidence and declared threat/failure models.
41. Documentation consistently separates CURRENT/PARTIAL/GAP/TARGET.
42. Redundant legacy/parallel authorities identified by this roadmap are removed or explicitly justified by real consumers.
43. The roadmap is linked from the documentation navigation and general project roadmap.

When these conditions are satisfied, Intergrax should behave as one coherent governed AI execution operating layer rather than as a collection of individually strong but operationally separate subsystems.
