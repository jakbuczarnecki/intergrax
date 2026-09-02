# Harness Architecture Evolution Roadmap

## Status and baseline

This document is the canonical cross-domain roadmap for the next stage of Intergrax harness architecture evolution.

It coordinates work across existing semantic domains. It does **not** replace domain architecture documents and it must never become a second semantic authority.

**As-built audit baseline:** `development @ 38e5b54726f7b6e3861c59754b99dad7e52caf6f`, validated on 2026-09-02.

The roadmap is deliberately based on repository reality, not only architecture intent. Before any implementation session, the relevant CURRENT/PARTIAL/GAP statements must be revalidated against the then-current `development` HEAD.

Status labels:

- **CURRENT** — real implementation and/or canonical contract exists and is usable as a foundation.
- **PARTIAL** — substantial implementation exists, but convergence, migration, hardening, proof, or adoption remains.
- **GAP** — materially missing from canonical runtime paths.
- **TARGET** — target architecture is defined but implementation/proof is incomplete.
- **OPTIONAL** — implement only when a Scenario Proof or concrete product need justifies it.
- **CONSOLIDATE** — converge existing surfaces instead of creating a new authority.

Hard rule:

> **ALREADY EXISTS => DO NOT REBUILD.**

When this roadmap and an older CURRENT section in a domain document disagree, implementation must pause until the code reality and canonical documentation are reconciled.

---

# 1. Core architectural position

Intergrax evolves toward one coherent governed execution operating layer without flattening domain semantics.

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
Checkpoint != Runtime Identity Authority
Inbound Interaction != Human Continuation Interaction
```

Target mental model:

```text
ApplicationEnvironmentProfile
        ↓
Profile Resolution
        ↓
Effective Runtime Composition
        ↓
Execution Runtime / Execution Boundary
        ↓
Capabilities + Governance + Context
        ↓
Durable Runtime Facts / Checkpoint / Evidence
        ↓
Inspection + Diagnostics + Evaluation
        ↓
Adaptive, Governed Evolution
```

The goal is not to create a universal god-object. Existing semantic domains remain authoritative and must compose through explicit contracts.

---

# 2. Canonical ownership decisions

## 2.1 Application environment authority

`ApplicationEnvironmentProfile` remains the only Tier-3 environment composition authority.

Do not introduce a peer `HarnessProfile` or any other second environment authority.

Allowed additions:

- `ProfileResolution` typed evidence,
- immutable effective snapshots/revisions,
- typed deltas/overlays as inputs,
- read-only runtime projections,
- provenance and semantic diff,
- inspection and explanation APIs.

An overlay, read model, preset, inspector, or effective view is never an independent configuration authority.

## 2.2 Domain ownership remains distributed

- UER owns execution lifecycle, execution identity, admission coordination, strategy routing coordination, and execution-tree runtime semantics.
- Nexus owns accepted orchestration topology decisions and what executes next.
- Governance owns policy, authority, approval, and meaningful-side-effect authorization decisions.
- Budget owns allowance, reservation, consumption, release, and enforcement semantics.
- Tools own executable tool contracts and ToolRuntime semantics.
- Skills own capability composition; instructional skills remain distinct.
- Context Engineering owns model-facing context assembly.
- UCL owns durable context lifecycle coordination where applicable.
- Memory owns persistent memory semantics.
- RAG owns governed retrieval semantics.
- Observability/HOS records canonical platform facts and historical projections.
- DIAG interprets canonical evidence; it does not create execution truth.
- Checkpoint/recovery owns durable restore state, not identity authority.
- Platform Plugins own package/control-plane coordination, not Tool/Skill/RAG/Memory runtime semantics.
- Runtime interaction intake owns inbound interaction normalization; human continuation interaction remains a distinct runtime seam.

## 2.3 No new subsystem without new semantic ownership

Before introducing a new architecture domain, answer:

> Is there a genuinely new responsibility that no existing domain should own?

If not, implement the behavior as a provider, plugin, policy, guard, strategy, hook, adapter, projection, preset, read model, or domain-owned extension.

## 2.4 Provider dependency direction

Canonical dependency direction:

```text
Consumer
  ↓
Domain Contract / Service Definition
  ↓
Provider
```

Consumers must not depend directly on concrete providers when a provider-neutral contract exists or should exist.

---

# 3. Mandatory architectural invariants

These invariants must be reflected in canonical documentation, code, conformance tests, runtime diagnostics, and Scenario Proofs where applicable.

1. **INV-1 — One environment composition authority**  
   `ApplicationEnvironmentProfile` is the only Tier-3 environment composition authority.

2. **INV-2 — Configured is not effective**  
   `CONFIGURED != EFFECTIVE`.

3. **INV-3 — Effective decisions are explainable**  
   Effective values, permissions, capability availability, context decisions, and strategy decisions have attributable provenance.

4. **INV-4 — Model-visible means reconstructable according to evidence policy**  
   `MODEL_VISIBLE => RECONSTRUCTABLE` through an explicit reconstruction level.

5. **INV-5 — Child authority cannot exceed parent authority**  
   `child_authority ⊆ parent_authority`.

6. **INV-6 — Proposal is not permission**.

7. **INV-7 — Permission is not execution**.

8. **INV-8 — Topology identity is not runtime identity**  
   `NodeId != ExecutionId`.

9. **INV-9 — Agent is not Execution**.

10. **INV-10 — Plugin package is not domain capability**.

11. **INV-11 — Model-context compaction is not evidence deletion**.

12. **INV-12 — Observability records truth; it does not invent execution truth**.

13. **INV-13 — Runtime extensions cannot self-expand authority**.

14. **INV-14 — Temporary capability expires with owner scope unless explicitly promoted**.

15. **INV-15 — Dynamic registration is reversible**.

16. **INV-16 — Meaningful side effects cross fresh governed authorization immediately before the effect**.

17. **INV-17 — Every canonical model call crosses Context Engineering**.

18. **INV-18 — Every canonical tool call crosses ToolRuntime**.

19. **INV-19 — Independently meaningful child work is admitted by UER**.

20. **INV-20 — Canonical runtime boundaries are transport-independent**.

21. **INV-21 — Required causal evidence precedes meaningful work**  
   When durable causal/audit evidence is required to establish an admission, recovery, or side-effect boundary, persistence must succeed before meaningful work begins. Failure fails closed at that boundary.

22. **INV-22 — Profile overlays are inputs, not authorities**.

23. **INV-23 — Consumers depend on provider-neutral contracts**.

24. **INV-24 — Tool and capability authority is monotonic**  
   Downstream scope may narrow upstream authority but may never widen it.

25. **INV-25 — Activation is atomic from the runtime-consumer perspective**.

26. **INV-26 — In-flight Executions are version-pinned**  
   Live reconfiguration must not silently rebind provider/profile/schema semantics of admitted Executions.

27. **INV-27 — Human interaction absence never implies approval**.

28. **INV-28 — Checkpoint is durable state, not identity authority**.

29. **INV-29 — Session/read models are projections where facts already exist canonically**.

30. **INV-30 — Security claims are bounded by a declared threat model and evidence**.

31. **INV-31 — Skill requirement is not host capability grant**  
   A Skill may require a Tool; it must not silently expand host `ToolProfile` availability.

32. **INV-32 — Caller-provided tool scope is narrowing only**  
   An explicit per-call/invoker allow-list must intersect stricter upstream authorities; it must never replace them.

33. **INV-33 — Effective profile revisions are immutable**  
   Reconfiguration produces a new revision; it does not mutate an already bound effective profile in place.

34. **INV-34 — Inbound interaction and execution-to-human continuation remain distinct contracts**.

---

# 4. As-built program status matrix

| ID | Initiative | As-built status | Canonical owner / existing area | Change type |
|---|---|---|---|---|
| A | Unified Execution convergence | **CURRENT / PARTIAL** | UER / UEA | finish convergence + proof |
| B | Profile resolution and effective composition | PARTIAL | Applications / environment profile | consolidation + DX |
| C | Runtime inspection and explanation | GAP / PARTIAL | read models over canonical facts | new read-model + DX |
| D | Reconstructable model execution | PARTIAL | CE + HOS + UER evidence | hardening |
| E | Context provider convergence | **CURRENT / PARTIAL** | Context Engineering | lifecycle/provenance hardening |
| F | Canonical ToolRuntime pipeline | CURRENT / PARTIAL | Tools / ToolRuntime | safety + convergence |
| G | Runtime credentials and secret references | PARTIAL | security/secrets/integrations | provider seam + late resolution |
| H | Execution sandbox and isolation | CURRENT / PARTIAL | runtime sandbox + security + execution | convergence |
| I | Subagent and external-agent providers | PARTIAL / TARGET | delegation + UER + Nexus | provider seam |
| J | Background Execution control | CURRENT / PARTIAL | Background Tasks + UER | convergence + DX |
| K | Verified external event intake | PARTIAL | interactions/integrations + UER | generalization + durability |
| L | Artifacts, attachments, spill | PARTIAL | artifacts/storage + CE + tools | consolidation |
| M | Context compaction and retention | PARTIAL / TARGET | CE/UCL/token optimization | hardening |
| N | Runtime invariant service | GAP / PARTIAL | domain checks + diagnostics | shared runner, domain-owned rules |
| O | Dynamic orchestration proposals | TARGET | Orchestration/Nexus/Governance | proposal path |
| P | Capability Skills + Instruction Skills | CURRENT / PARTIAL / GAP | Skills + CE | safety fixes + instructional type |
| Q | Dynamic/reversible runtime registration | PARTIAL | Platform Plugins + domain registries | scoped runtime lifecycle only |
| R | Governance UX and permission presets | CURRENT / PARTIAL | Governance/HITL/ToolRuntime/sandbox | DX + hardening |
| S | Workspace/settings/credentials separation | PARTIAL | applications/workspace/security | semantic cleanup |
| T | Feedback/goals/plans/work-state | PARTIAL / OPTIONAL | collaboration/evaluation/AHI | scenario-driven |
| U | Scheduling convergence | PARTIAL | scheduler/intake + UER | convergence |
| V | Process/filesystem/terminal/code providers | PARTIAL / OPTIONAL | tools/execution providers | provider-first |
| W | SDK/API/ACP/MCP/host convergence | PARTIAL | host/application boundaries + UER | convergence |
| X | Generated architecture/capability metadata | PARTIAL | tooling/control plane | automation |
| Y | Documentation architecture synchronization | PARTIAL | documentation canon | re-baseline + simplification |
| Z | Security/trust/supply-chain hardening | PARTIAL | security/governance/plugins | hardening |
| AA | Memory and RAG hardening | CURRENT / PARTIAL | Memory + RAG + CE | hardening |
| AB | AHI expansion | CURRENT / PARTIAL | AHI + Evaluation/HOS | controlled expansion |
| AC | Governed Runtime Evolution | GAP / LATE TARGET | sandbox + UER + Governance + Plugins + AHI | strategic capability |
| AD | Test and qualification infrastructure | PARTIAL | test-support + qualification + proofs | hardening |
| AE | Developer/operator experience | PARTIAL | CLI/docs/inspection | DX |
| AF | Checkpoint, durability, and recovery convergence | **CURRENT / PARTIAL** | long-running runtime + UER + reliability | close remaining durability gaps |
| AG | Effective capability health/readiness | GAP / PARTIAL | operational projection over domain facts | read-model + lifecycle |
| AH | Controlled live composition reconfiguration | GAP / TARGET | ProfileResolution + lifecycle + UER | later controlled reconfiguration |
| AI | Human continuation interaction seam | PARTIAL | Governance/HITL/host interaction | convergence |

This matrix is a planning baseline, not a permanent truth. P0A below revalidates it before implementation.

---

# 5. Initiative A — Unified Execution Runtime convergence

## Current reality

The repository already contains substantial canonical UER implementation:

- canonical `ExecutionId`,
- root execution identity/context,
- `ExecutionRuntime`,
- strategy-neutral `ExecutionBoundary`,
- active Run/Attempt/Execution binding,
- `parent_execution_id`,
- `ChildExecutionRunner`,
- child authority resolution,
- child budget allocation/reservation/release,
- `StrategyExecutionRouter`,
- inference/agentic/orchestration strategy surfaces,
- `RuntimeEvent.execution_id`,
- graph-node work routed through child Executions on canonical paths,
- Execution-tree checkpoint models.

Therefore the task is **not** to invent these concepts again.

## Remaining work

1. Audit every public and internal entry path for canonical UER adoption.
2. Remove/bound remaining private execution identities or bypasses.
3. Complete Execution Tree lineage queries and subtree cancellation where incomplete.
4. Finish cancellation propagation across local, remote, background, model, tool, and delegated work.
5. Complete pause/resume semantics at all supported Execution boundaries.
6. Freeze and enforce retry ownership taxonomy:
   - provider/tool/internal retry,
   - Execution retry generation,
   - transport redelivery,
   - whole-Run retry.
7. Finish hierarchical budget dimensions, including:
   - tokens,
   - money,
   - tool calls,
   - child execution count,
   - concurrency,
   - wall-clock,
   - agent-loop/step limits.
8. Converge Execution Environment semantics without creating a new authority.
9. Converge neutral Execution Result ABI where legacy agent-centric results leak upward.
10. Ensure distributed worker/redelivery paths preserve logical runtime identity.
11. Prove inference, agentic, orchestration, delegated, background, and resumed flows on the same runtime spine.

**Acceptance:** remaining work is convergence and proof, not reimplementation of existing UER foundations.

---

# 6. Initiative B — Profile resolution and effective composition

1. Preserve `ApplicationEnvironmentProfile` as the sole Tier-3 composition authority.
2. Define canonical resolution order:
   platform defaults → product/distribution → application → agent delta → Run/Task delta → Execution delta.
3. Model every overlay as a typed delta submitted to `ProfileResolution`.
4. Record:
   - input layer revisions,
   - overrides,
   - rejected overrides,
   - authority clamps,
   - dependency failures,
   - warnings,
   - degraded capabilities.
5. Separate configured/requested state from effective state.
6. Bind a durable effective profile revision/fingerprint to every relevant Run/Execution.
7. Add semantic profile diff across tools, model routing, policy, sandbox, CE, orchestration, providers, and credentials references.
8. Validate dependencies such as Skill → Tool → Integration → Credential → Provider.
9. Required dependency failure = fail before execution.
10. Optional dependency failure = explicit degraded capability, never silent fallback.
11. Presets expand to canonical profile/policy semantics and never grant authority.
12. Audit legacy flat/wire compatibility accessors and remove unjustified compatibility surfaces after consumer audit.
13. Never mutate an effective profile in place; produce revision N+1.

---

# 7. Initiative C — Runtime inspection and explanation

Create one programmatic read-model API consumed by CLI, web, SDK, diagnostics, and future hosts.

It must expose:

- configured vs effective environment,
- effective profile revision,
- agent model/tools/skills/context/memory/authority/budgets,
- Execution identity/tree/state/strategy/children,
- model requests,
- tool calls,
- approvals/HITL,
- evidence references,
- tool registration/availability/visibility/selectability/executability/authorization,
- skill version/resolution/provenance,
- context inclusion/exclusion decisions and token cost,
- capability dependency graph,
- effective capability health/readiness.

It must answer:

- Why is this tool unavailable?
- Which authority narrowed it?
- Why was approval required?
- Why did this profile value win?
- Why was a context fragment included/excluded?
- Which provider/version is this Execution pinned to?
- Why was this strategy selected?
- Which invariant failed?

Inspection is a projection only. It may not mint canonical identities or create truth.

---

# 8. Initiative D — Reconstructable model execution

`LLM_CALL` telemetry is not sufficient by itself.

Introduce canonical model-request evidence linked to Execution.

Required fields/relations:

- `ModelRequestId`,
- `ExecutionId`,
- exact model/provider/route,
- provider/model revision where available,
- prompt/system revision,
- tool-schema-set revision,
- effective profile revision,
- context decision snapshot/provenance,
- sampling/runtime parameters,
- artifact references for large content,
- result identity,
- compaction lineage,
- retention/redaction events.

Three reconstruction levels:

1. **Exact** — exact model-visible payload can be reconstructed.
2. **Referential** — exact content is recoverable through immutable authorized artifact/reference.
3. **Structural provenance** — content was legitimately deleted/redacted, but source identity/hash/revision/reason/retention event remain.

Every request declares which level is promised and must satisfy it.

---

# 9. Initiative E — Context Engineering provider convergence

## Current reality

`ContextProvider`-style contracts and providers already exist. Workspace, session, tool-output, and other sources already feed canonical CE paths.

Therefore do **not** create a second provider abstraction.

Remaining work:

1. Converge all canonical model calls through CE.
2. Audit provider lifecycle and hot-refresh semantics.
3. Define fragment lifetime and replacement semantics.
4. Preserve source identity, version, hash, trust, freshness, policy decision, and inclusion/exclusion reason.
5. Add lazy activation for expensive instruction/context sources where useful.
6. Preserve mandatory fragments under all degradation/compaction strategies.
7. Make cache-aware/context-cost decisions attributable.
8. Add model-safe capability/context introspection without leaking secret/internal-only metadata.
9. Ensure Memory/RAG/Tools never bypass CE to inject model-visible content.

---

# 10. Initiative F — Canonical ToolRuntime pipeline

Preserve ToolRuntime as the execution authority for tools.

Canonical pipeline should converge around:

```text
resolve tool
→ effective permission intersection
→ validate input
→ pre-execute policy/guards
→ fresh meaningful-side-effect authorization when applicable
→ wrappers (timeout/cancel/retry/concurrency)
→ execute
→ authoritative typed result
→ post-execute
→ finalize model/user presentation
→ evidence
```

Required hardening:

- structured result ABI,
- structured errors (`error_id`, class, origin, retryability, user-safe message, diagnostic ref, cause),
- real timeout/cancellation semantics,
- parallel-safe/exclusive hints,
- nested invocation lineage,
- cost/resource attribution,
- removal/routing of private tool loops,
- small guard plugins/hooks for repeat-call/runaway/deadline behavior.

## P0-SAFETY-1 — Tool Authority Intersection Integrity

Current code path must be corrected so explicit caller/invoker allow-lists cannot override stricter `RuntimePolicyBundle.tool_access` or other upstream authority.

Required semantic rule:

```text
effective tool scope =
host availability
∩ agent/skill requirements
∩ runtime policy
∩ modality/plan narrowing
∩ invoker scope
∩ per-call narrowing
```

No downstream list may widen upstream permission.

**Acceptance:** regression tests prove explicit caller lists only narrow.

---

# 11. Initiative G — Runtime credentials and secret references

Preserve existing `SecretsStore` and provider integrations.

Add/converge the runtime credential layer around references, not raw secret values:

- `CredentialRef`,
- provider-neutral credential resolution,
- late/per-operation resolution,
- tenant/workspace ownership,
- Execution-scoped exposure,
- no model visibility by default,
- redacted logs/evidence,
- rotation applied to subsequent operations without mutating historical meaning,
- human/OAuth authorization flows,
- credential access evidence,
- expiry/revocation behavior,
- compatibility with sandbox/network policy.

Do not build a second secret store.

---

# 12. Initiative H — Execution sandbox and isolation convergence

## Current reality

Intergrax already has runtime sandbox models, manager/session, host wiring, sandbox integrations, sandbox tools, and CodeCraft sandbox resolution.

Goal is convergence into one Execution-facing isolation model.

Required work:

1. Define `ExecutionSandbox` / `ExecutionEnvironment` as provider-neutral runtime contract/projection.
2. Standard presets such as read-only, workspace-write, and explicitly privileged modes.
3. Bind filesystem, process, network, credential, workspace, and resource policies.
4. Scope sandbox lifetime to Execution/subtree where appropriate.
5. Support one-shot governed escalation.
6. Make effective isolation inspectable.
7. Allow local/container/remote/microVM providers without forcing feature parity.
8. Fail closed when requested isolation cannot be established.

Do not rebuild existing sandbox providers solely for parity.

---

# 13. Initiative I — Subagent and external-agent providers

Create one provider-neutral delegation seam over existing execution/delegation semantics.

Potential providers:

- Intergrax child agent,
- remote Intergrax worker,
- ACP agent,
- external coding-agent provider,
- vendor agent SDK,
- remote service agent.

Every delegated work unit that is independently meaningful becomes a child Execution and inherits/narrows:

- authority,
- budget,
- workspace scope,
- credential scope,
- sandbox/environment policy,
- cancellation,
- evidence requirements.

Support:

- create/delegate,
- follow-up/continuation,
- interrupt/cancel,
- list/status,
- completion notification.

External provider convenience must not bypass UER/Governance/Evidence.

---

# 14. Initiative J — Background Execution control surface

Background Tasks already exist. Do not create a second Job runtime.

Converge existing background work with final UER semantics:

- owner Execution/session/workspace fencing,
- non-blocking submit,
- list/status/wait/cancel,
- completion notification,
- durable detached semantics where supported,
- transport redelivery mapped to same logical Execution where appropriate,
- canonical cancellation and recovery,
- causal evidence before worker handler execution where required.

---

# 15. Initiative K — Verified external event intake

Generalize authenticated external events into canonical governed work:

```text
provider event
→ authenticate/verify
→ normalize
→ tenant/workspace resolve
→ dedup/idempotency
→ durable receipt/delivery state
→ policy/admission
→ Task/Execution
→ evidence
```

Required:

- replay protection,
- stable event identity,
- durable delivery semantics,
- retry/dead-letter strategy where applicable,
- no direct session-only execution dependency,
- safe crash/redelivery behavior,
- explicit relationship to AF recovery semantics.

---

# 16. Initiative L — Artifacts, attachments, and large-output spill

Consolidate around durable artifact identity rather than a separate context system.

Required:

- artifact ID/version,
- owner/tenant/workspace scope,
- producer Execution/tool/model identity,
- MIME/schema metadata,
- provenance/hash,
- retention policy,
- user-visible vs model-visible projection,
- large tool-output spill,
- bounded preview + locator,
- later read/search access,
- lineage across compaction and reconstruction.

Invariant:

> Artifact existence does not imply model-context inclusion.

---

# 17. Initiative M — Context compaction and retention

Implement/converge automatic and manual compaction under CE/UCL ownership.

Order of preference:

1. remove redundant representation,
2. prune/spill old tool output,
3. preserve durable artifact locator,
4. summarize conversation/work state with lineage,
5. degrade only within declared CE policy.

Protect:

- system/policy instructions,
- active approvals/constraints,
- mandatory context fragments,
- unresolved work state,
- reconstruction lineage.

Compaction never deletes audit/evidence merely to reduce model tokens.

---

# 18. Initiative N — Runtime invariant service

Rules remain domain-owned; execution is shared.

Build a central runner/catalog able to execute domain-provided invariant checks in runtime, diagnostics, and CI.

Each invariant has:

- stable ID,
- owner,
- severity,
- applicable scope,
- runtime/CI enablement,
- evidence references,
- remediation guidance.

Initial critical invariants include:

- model-visible reconstructability,
- child authority ≤ parent,
- tool permission monotonicity,
- Skill cannot expand host ToolProfile,
- side effect has fresh authorization,
- RuntimeEvent has ExecutionId,
- effective profile revision exists,
- checkpoint identity consistent with UER,
- required causal evidence persisted before work.

Do not create a central business-policy engine.

---

# 19. Initiative O — Dynamic orchestration proposals

Flow:

```text
model/agent proposes topology
→ parse typed proposal
→ static validation
→ capability/dependency checks
→ governance/authority checks
→ budget checks
→ accept/reject
→ accepted topology
→ Nexus
→ child Executions via UER
```

Proposal identity and accepted topology identity must be distinct.

No model-generated topology executes directly.

---

# 20. Initiative P — Capability Skills + Instruction Skills

## Current Capability Skills

Preserve existing deterministic Skill model, resolver, catalog, registry, bundles, dependencies, tool requirements, imports, and AHI recommendation semantics.

Specific hardening items:

1. Resolve version identity ambiguity.
2. Persist/retain `ResolvedSkillPack` provenance with effective agent/runtime revision.
3. Make prompt bridge consumption explicit and canonical where needed.
4. Make policy fragment bridge consumption explicit and canonical where needed.
5. Preserve deterministic no-LLM capability resolution.

## P0-SAFETY-2 — Skill Authority Integrity

Current Tier-3 wiring must not allow Skill requirements to silently expand host ToolProfile availability.

Target:

```text
Skill requires Tool X
        ↓
Host ToolProfile contains X?
   YES → eligible for later policy narrowing
   NO  → fail/degraded diagnostic
```

Never:

```text
Skill requires X
→ silently add X to host capability availability
```

## Instruction Skills

Add a distinct task-instruction abstraction for reusable instructions/playbooks that can be:

- discovered,
- indexed by short metadata,
- lazily loaded,
- versioned,
- workspace/application scoped,
- hot-refreshed where safe,
- included through CE.

Instruction Skills cannot grant Tool/Integration/credential authority.

---

# 21. Initiative Q — Dynamic and reversible runtime registration

The Platform Plugins implementation program is already closed at the package/control-plane layer and must **not** be reopened as a global plugin-engine rewrite.

This initiative applies only to runtime-scoped dynamic registration and remaining lifecycle/Protocol-v2 hardening.

Required runtime semantics:

```text
register
→ validate
→ stage
→ activate
→ inspect
→ deactivate
→ unregister
```

Requirements:

- reversible registration handles,
- rollback on partial activation failure,
- execution/run/session/workspace ownership,
- version coexistence,
- compatibility validation,
- conflict detection,
- in-flight Execution pinning,
- draining old versions,
- no self-expansion of authority.

Platform Plugins remain trusted in-process deployed packages; runtime-generated temporary extensions use a different bounded lifecycle until explicitly promoted.

---

# 22. Initiative R — Governance UX and permission presets

Preserve deep governance semantics while making them easier to operate.

Provide ergonomic presets that compose:

- Tool access,
- sandbox isolation,
- filesystem/network constraints,
- credential exposure,
- approval policy,
- external-effect restrictions.

Examples:

- read-only,
- workspace-write,
- governed-network,
- restricted-side-effect,
- explicitly privileged.

Presets are configuration shortcuts only; they never bypass canonical Governance/ToolRuntime enforcement.

---

# 23. Initiative S — Workspace, Settings, Credentials separation

Keep three different concepts:

- **Workspace** — runtime/project boundary containing files, instructions, artifacts, source bindings, configuration references.
- **Settings** — user/operator preferences and non-secret behavior configuration.
- **Credentials** — secret-bearing authorization resources and references.

Do not collapse them into one profile blob.

Execution obtains an effective scoped projection through ProfileResolution/runtime composition.

---

# 24. Initiative T — Feedback, goals, plans, and work-state

Do not create architecture simply because a state type is fashionable.

Implement only concrete scenario-driven collaboration state.

Useful canonical path:

```text
human/system feedback
→ typed feedback artifact
→ Evaluation
→ HOS evidence
→ optional AHI proposal input
```

Goals/plans/todos are work-state artifacts, not authority.

---

# 25. Initiative U — Scheduling convergence

Schedule definitions are admission sources, not a second execution lifecycle.

```text
Schedule
→ trigger
→ Task/Execution admission
→ UER
```

Required:

- durable schedule identity,
- owner/workspace scope,
- recurrence,
- pause/cancel,
- policy/authority evaluation,
- deduplication/idempotency,
- evidence,
- bounded retry semantics.

---

# 26. Initiative V — Process, filesystem, terminal, and code-runtime providers

Provider-first strategy:

- `ProcessProvider`,
- `FilesystemProvider`,
- `TerminalProvider`,
- `CodeRuntimeProvider`.

Implement local backends only where Scenario Proofs justify them. External/remote providers are acceptable if they remain under UER/Governance/Sandbox/Evidence.

Do not enter a feature-parity race for coding-agent subsystems.

---

# 27. Initiative W — SDK/API/ACP/MCP/host convergence

All transports and host adapters must route through the same canonical runtime semantics.

No private:

- execution identity model,
- ToolRuntime,
- policy model,
- CE implementation,
- retry ownership,
- result semantics.

Generate transport/client schemas from canonical contracts where practical.

---

# 28. Initiative X — Generated architecture and capability metadata

Generate and freshness-gate in CI where feasible:

- capability catalog,
- tool catalog,
- skill catalog,
- provider catalog,
- invariant catalog,
- configuration schema,
- dependency/module graph,
- generated API/RPC schemas,
- generated client types,
- compatibility/stability metadata.

Capability descriptors should include:

- ID/kind/version,
- provider,
- dependencies,
- risk/trust/qualification,
- required authority,
- stability (`experimental`, `preview`, `stable`, `deprecated`),
- compatibility range.

---

# 29. Initiative Y — Documentation architecture synchronization

## P0A documentation re-baseline

The repository currently contains at least one important case where older CURRENT documentation lags behind shipped runtime code.

Before code implementation begins:

1. Re-audit current code at HEAD.
2. Update stale CURRENT sections in canonical architecture documents.
3. Preserve TARGET sections where still valid.
4. Mark implemented milestones as CURRENT/DONE rather than asking future sessions to reimplement them.
5. Ensure cross-references do not contradict code reality.

Priority documents include:

- UEA/UER,
- UER satellites,
- Nexus execution flow,
- Background Tasks,
- Observability,
- Checkpoint/recovery,
- Tools,
- Skills,
- Context Engineering,
- Platform Plugins,
- Governance/HITL.

Gate:

> No implementation session may rely on a canonical CURRENT statement known to conflict with current code.

---

# 30. Initiative Z — Security, trust, and supply-chain hardening

Cover:

- package trust,
- extension qualification,
- signature/provenance where available,
- credential minimization,
- tenant/workspace isolation,
- safe default profiles,
- explicit threat models,
- adversarial conformance suites,
- authority monotonicity,
- side-effect fencing/idempotency,
- security-safe inspection/redaction.

Avoid absolute claims such as “escape is impossible.” Claims must be bounded to a threat model and accepted evidence.

---

# 31. Initiative AA — Memory and RAG hardening

Preserve existing domain depth.

Focus on convergence and evidence:

- tenant/workspace ownership,
- provenance/citations,
- publication generations,
- versioned retrieval configuration,
- durable source identity,
- retention/forget/tombstone semantics,
- hybrid/hierarchical/graph retrieval qualification,
- policy gates,
- CE-only model projection,
- provider/load evidence,
- reproducible retrieval decisions.

Memory remains source-of-truth lifecycle; vector stores remain retrieval indexes, not truth authority.

---

# 32. Initiative AB — Adaptive Harness Intelligence expansion

Preserve AHI as evidence-driven governed adaptation, not uncontrolled self-modification.

Flow:

```text
evidence
→ HOS/Evaluation
→ AdaptationEngine
→ proposal
→ Governance
→ profile/artifact version
→ shadow
→ canary
→ apply
→ verify
→ keep/rollback
```

Potential artifact families after foundations are proven:

- routing configuration,
- prompt/context policy,
- Skill recommendations,
- budget parameters,
- provider selection,
- workflow proposal heuristics,
- compaction strategy.

Auto-apply remains bounded and policy-controlled.

---

# 33. Initiative AC — Governed Runtime Evolution

**Late P4 only.**

Do not begin before UER, sandbox, credentials, reconstructability, runtime invariants, reversible registration, and recovery are proven.

Flow:

```text
CapabilityGap
→ RuntimeExtensionProposal
→ static validation
→ sandbox build/test
→ invariant suite
→ Governance
→ shadow activation
→ bounded evaluation
→ execution-scoped activation
→ expiry
→ optional explicit promotion / rollback
```

Hard rules:

- generated extension cannot grant itself authority,
- build/test runs in isolation,
- temporary capability is scoped and expiring,
- activation is reversible,
- old in-flight executions remain version-pinned,
- promotion is explicit,
- evidence is durable,
- failure does not mutate canonical package configuration.

A temporary runtime extension is not automatically a Platform Plugin installation.

---

# 34. Initiative AD — Test and qualification infrastructure

Standardize deterministic support for:

- model/provider replay,
- fault injection,
- tool timeout/cancellation,
- sandbox violations,
- credential absence/rotation,
- event redelivery,
- crash/restart,
- checkpoint resume,
- child authority narrowing,
- budget exhaustion,
- policy/HITL,
- runtime invariant violations,
- extension activation/rollback.

Every high-value architecture invariant needs at least one executable conformance gate.

---

# 35. Initiative AE — Developer and operator experience

Provide simple surfaces over canonical semantics:

- `inspect environment`,
- `inspect execution`,
- `inspect tools`,
- `inspect skills`,
- `inspect context`,
- `inspect health`,
- `explain ...`,
- `doctor`,
- profile dump/diff,
- execution tree view,
- approvals/pending interactions,
- background work controls,
- invariant violations,
- reconstruction/replay links.

DX must consume read models; it must not create parallel runtime logic.

---

# 36. Initiative AF — Checkpoint, durability, and recovery convergence

## Current reality

Checkpoint/recovery is substantially implemented already:

- `RuntimeCheckpoint` v2,
- `ExecutionTreeSnapshot`,
- root/child Execution entries,
- tree validation/cycle checks,
- execution status,
- historical vs active resume planning,
- completed work adoption/skipping,
- interrupted work resumption,
- UAEP cursor/state,
- graph/node state,
- prior outputs,
- pending decisions,
- pending human request.

Do not rebuild this foundation.

## Remaining audit/closure scope

1. Verify persistent checkpoint store semantics on all canonical paths.
2. Audit budget reservation/consumption recovery.
3. Audit meaningful-side-effect fence/idempotency restoration.
4. Audit transport/delivery cursor recovery.
5. Audit credential reference/lease restoration where applicable.
6. Audit external delegated child recovery.
7. Audit background worker crash/restart semantics.
8. Define checkpoint commit ordering versus meaningful work.
9. Preserve canonical identity ownership on resume.
10. Make retry/redelivery/resume relationships explicit and inspectable.
11. Reuse existing idempotency/fencing primitives; do not invent a generic exactly-once runtime.

Hard distinction:

```text
at-most-once authorization
!= exactly-once external effect
```

Acceptance is bounded by explicit delivery/failure models.

---

# 37. Initiative AG — Effective capability health, readiness, and atomic activation

Implement an operational read-model, not a new semantic authority.

Suggested effective states:

- READY,
- DEGRADED,
- UNAVAILABLE,
- FAILED,
- DRAINING.

Compute from canonical facts such as:

- registration,
- dependency resolution,
- compatibility/qualification,
- provider health,
- credential state,
- sandbox availability,
- policy/authority,
- host support.

Example:

```text
capability: jira.search
registered: yes
configured: yes
qualified: yes
credential: expired
provider: degraded
authorized: yes
effective: UNAVAILABLE
reason: credential unavailable
```

Activation must be staged and atomic from consumers' perspective:

```text
resolve
→ validate
→ stage
→ activate
→ readiness checks
→ commit
```

Failure before commit rolls back staged registrations/effects.

---

# 38. Initiative AH — Controlled live composition reconfiguration

Later-stage capability.

Never mutate an effective profile in place.

Flow:

```text
revision N
→ proposed typed patch
→ authority/dependency validation
→ stage providers/registrations
→ health checks
→ atomic commit revision N+1
→ new Executions bind N+1
→ existing Executions remain pinned to N
→ drain old versions
→ rollback if required
```

Candidate changes:

- model route,
- provider selection,
- context provider activation,
- instruction set,
- permission preset narrowing,
- runtime capability version.

No silent rebind of in-flight Executions.

---

# 39. Initiative AI — Canonical Human Continuation Interaction seam

Existing inbound `InteractionAdapter` normalizes external input into Tasks. Do not overload it as the outbound Human-in-the-Loop contract.

Add/converge a provider-neutral continuation seam for:

- approval,
- question,
- choice,
- clarification,
- credential authorization,
- review.

Conceptual model:

```text
Execution
→ HumanInteractionRequest
→ host/provider
→ human
→ typed response
→ same Execution resumes
```

Hosts may include CLI, web, Slack/Teams, mobile, or API.

If required interaction has no authorized delivery provider:

```text
pause / deny / fail according to policy
```

Never auto-allow.

---

# 40. P0A — As-built re-baseline

**No feature implementation in this phase.**

Purpose: prevent rebuilding shipped runtime and prevent stale docs from directing implementation backward.

Tasks:

1. Freeze starting `development` SHA for the audit.
2. For every P0/P1 initiative classify every sub-item:
   - DONE,
   - PARTIAL,
   - GAP,
   - DO NOT DO.
3. Cite concrete implementation paths/tests.
4. Reconcile UER documentation with shipped:
   - `ExecutionId`,
   - `ExecutionRuntime`,
   - `ExecutionBoundary`,
   - `StrategyExecutionRouter`,
   - `RuntimeEvent.execution_id`,
   - child execution lineage/budget,
   - Execution-tree checkpoint/resume.
5. Reconcile ContextProvider documentation with existing implementation.
6. Reconcile Platform Plugins roadmap-complete status with Q scope.
7. Reconcile Skills CURRENT gaps.
8. Reconcile Tool permission code vs architectural invariant.
9. Reconcile checkpoint CURRENT vs remaining gaps.
10. Update canonical CURRENT sections before code-changing implementation sessions.
11. Preserve TARGET sections that remain valid.

### P0A exit gate

- current code and canonical CURRENT docs no longer materially contradict each other in the areas this program will modify;
- every P0B/P0C task has an evidence-backed CURRENT/PARTIAL/GAP classification;
- no task asks Cursor/implementers to rebuild an already shipped foundation.

---

# 41. P0B — Safety and authority closure

Complete before adding stronger dynamic capability surfaces.

1. **P0-SAFETY-1 Tool Authority Intersection Integrity**.
2. **P0-SAFETY-2 Skill Authority Integrity**.
3. Child authority monotonicity gates.
4. Meaningful-side-effect fresh authorization integrity.
5. Required-causal-evidence-before-work gates.
6. Credential resolution/exposure boundary audit.
7. Sandbox/isolation fail-closed audit.
8. Retry/redelivery authorization semantics audit.
9. Side-effect idempotency/fence integration audit.

### P0B exit gate

No caller, Skill, child Execution, plugin/runtime extension proposal, or host shortcut can expand a stricter upstream capability/authority boundary on canonical tested paths.

---

# 42. P0C — Execution and durability convergence

After P0A/P0B:

1. Finish remaining UER entry-path adoption.
2. Finish Execution Tree queries/cancellation.
3. Finish pause/resume/cancel across supported providers.
4. Complete budget dimensions and child accounting.
5. Complete retry ownership taxonomy.
6. Complete AF durability gaps.
7. Converge background/transport runtime identity.
8. Converge delegated/external provider child execution semantics.
9. Bind effective profile revisions to execution evidence.
10. Establish ModelRequest evidence foundation.
11. Establish Human Interaction continuation foundation for pause/HITL.
12. Establish Runtime Inspection foundation.
13. Establish Runtime Invariant runner foundation.

### P0C exit gate

Representative inference, agentic, orchestration, background, delegated, pause/resume, and crash/recovery flows use the same canonical identity/authority/profile/evidence/checkpoint semantics for the tested scope.

---

# 43. P1 — Composition, inspection, credentials, sandbox, health

1. Full ProfileResolution layering and provenance.
2. Effective profile diff/versioning.
3. Capability dependency validation.
4. Runtime Inspection/explain surfaces.
5. Effective capability health/readiness projection.
6. Atomic activation lifecycle.
7. CredentialRef/late resolution.
8. Sandbox/ExecutionEnvironment convergence.
9. Context provider lifecycle/provenance hardening.
10. Skill version/provenance bridge hardening.
11. Governance permission presets.

---

# 44. P2 — Delegation, background UX, event intake, artifacts, compaction

1. External/subagent provider seam.
2. Background execution UX/control convergence.
3. Verified external-event intake durability.
4. Artifact/attachment/spill convergence.
5. Context compaction.
6. Scheduling convergence.
7. Host/API/SDK/ACP/MCP convergence.
8. Generated capability/invariant/dependency catalogs.
9. Scenario Proofs for cross-domain runtime claims.

---

# 45. P3 — Dynamic orchestration and controlled reconfiguration

1. Workflow/topology proposal path.
2. Instruction Skills.
3. Dynamic reversible runtime registration.
4. Controlled live composition reconfiguration.
5. Provider version coexistence/draining.
6. Model-safe RuntimeCapabilityView.
7. Feedback/Evaluation bridge expansion.
8. Scenario Proofs for topology/reconfiguration/provider health.

---

# 46. P4 — Governed Runtime Evolution

Only after P0–P3 are proven:

1. CapabilityGap detection.
2. RuntimeExtensionProposal.
3. Static validation.
4. Isolated build/test.
5. Runtime invariant suite.
6. Governance decision.
7. Shadow activation.
8. Evaluation.
9. Execution-scoped activation.
10. Expiry.
11. Explicit promotion/rollback.

---

# 47. Dependency map

```text
P0A AS-BUILT REBASE
        ↓
P0B AUTHORITY / SAFETY
        ↓
P0C EXECUTION + DURABILITY
        ↓
ProfileResolution ───────→ Inspection / Explain
        │                         │
        ├──────────────→ Health / Readiness
        │                         │
        └──────────────→ Atomic Activation
                                  │
UER + Checkpoint + Version Pinning│
        └─────────────────────────┴──→ Live Reconfiguration

Credentials + Sandbox + Governance + UER
        ↓
Delegation / External Providers
        ↓
Dynamic Runtime Registration
        ↓
Governed Runtime Evolution

CE + Artifacts
        ↓
ModelRequest Reconstruction
        ↓
Compaction / Replay / Evaluation

External Event Intake + UER + AF
        ↓
Durable Triggered Work

Feedback + HOS + Evaluation
        ↓
AHI
        ↓
Governed Adaptive Evolution
```

---

# 48. Explicit anti-goals

Do not build unless future evidence overturns the decision:

1. second composition authority beside `ApplicationEnvironmentProfile`,
2. universal plugin API that erases domain semantics,
3. second generic Job runtime beside UER,
4. dynamic workflow execution runtime parallel to Nexus,
5. private Session runtime replacing canonical Memory/Execution/Observability facts,
6. automatic global installation of model-generated code,
7. runtime extensions that widen their own authority,
8. coding-agent feature-parity race for terminal/LSP/code runtime,
9. new subsystem for simple guards/policies,
10. compaction that implies evidence deletion,
11. host-specific ToolRuntime/Governance/CE/Execution semantics,
12. direct RAG/Memory model injection outside CE,
13. plan/goal/todo state granting authority,
14. telemetry sink becoming canonical truth,
15. profile overlay becoming independent authority,
16. checkpoint minting competing execution identity/tree,
17. health/readiness projection deciding business policy,
18. live reconfiguration mutating in-flight Executions,
19. new ContextProvider abstraction parallel to existing CE providers,
20. reopening the completed Platform Plugins program as a universal runtime lifecycle engine,
21. new secret store parallel to `SecretsStore`,
22. rebuilding existing ExecutionId/ExecutionBoundary/Execution Tree foundations.

---

# 49. Scenario Proof program

Each major cross-domain claim requires bounded adversarial proof before strong maturity claims.

1. **Execution identity and lineage**  
   Root/child Executions, retries, cancellation, evidence, checkpoint relations.

2. **Effective composition explainability**  
   Layer resolution, rejected override, authority clamp, dependency degradation, profile diff.

3. **Tool/Skill Authority Safety**  
   Explicit caller scope cannot widen policy; Skill cannot widen host ToolProfile.

4. **Model-request reconstruction**  
   Exact, referential, and structural-provenance cases.

5. **Governed external delegation**  
   External provider under inherited authority/budget/sandbox/credential limits.

6. **Large-output artifact/spill**  
   Bounded context with retrievable full artifact and lineage.

7. **Runtime invariant violation**  
   Deliberate critical violation, detection, attribution, evidence.

8. **External event to governed work**  
   Authenticate, deduplicate, persist causal receipt, admit, execute, recover.

9. **Crash / Resume / Side-effect Safety**  
   Nontrivial Execution Tree, forced crash, restore, preserved lineage, bounded duplicate-effect safety under declared failure model.

10. **Credential + Sandbox Escalation**  
    Read-only start, no secret exposure, human-approved one-shot escalation, scoped use, grant consumption, authority restoration.

11. **Atomic Activation / Provider Replacement**  
    Mid-stage activation failure rolls back; provider v2 activates while v1-bound Executions drain safely.

12. **Governed Runtime Evolution**  
    Capability gap → proposal → isolated tests → governance → shadow → bounded activation → evaluation → expiry/promotion/rollback.

Proof language must name the threat/failure model and must not convert bounded evidence into absolute security claims.

---

# 50. Target user and operator experience

A developer describes high-level intent without manually wiring every subsystem:

```yaml
profile: governed-workspace-agent
model: primary
permissions: workspace
memory: standard
knowledge: enabled
sandbox: workspace-write
```

The platform resolves canonical domain contracts.

The operator can ask:

```text
What is configured?
What is effective?
What is healthy or degraded?
Why is this tool unavailable?
Which authority narrowed it?
Why did this context fragment enter the model?
Why was it excluded?
Which profile value won?
Which policy required approval?
What child Executions were created?
Which credential references were used?
Which provider/version is this Execution pinned to?
Which meaningful side effects occurred?
Can this model request be reconstructed exactly, referentially, or structurally?
What changed between profile revisions?
Which invariant failed?
Can this Execution be resumed safely after a crash?
```

Answers come from canonical state/evidence and read-model projections, not best-effort log interpretation.

---

# 51. Target architecture

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
      immutable effective revision
                 │
                 ▼
        Execution Runtime
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
   ┌─────────────┼──────────────────────────────┐
   ▼             ▼              ▼              ▼
  CE           Tools          Memory/RAG     Delegation
                 │                              │
             Governance                    child Exec
                 │
       Human continuation when required
                 │
   └─────────────┴──────────────┬───────────────┘
                                ▼
                  Required causal evidence
                                │
                       Runtime Events / HOS
                                │
                 Checkpoint / recovery state
                                │
                                ▼
              Inspection / Evidence / DIAG / Eval
                                │
                                ▼
                               AHI
                                │
                                ▼
                    Governed Runtime Evolution
```

Cross-cutting control plane:

```text
Platform Plugins / capability metadata
ProfileResolution / revision pinning
Inspection / explanation / health
Runtime invariants
Credentials
Sandbox / Execution Environment
Artifacts
Generated catalogs and schemas
Qualification / tests / Scenario Proofs
```

---

# 52. Program Definition of Done

The program is complete only when all applicable statements are proven against then-current `development`.

1. `ApplicationEnvironmentProfile` remains the sole Tier-3 composition authority.
2. Configured and effective state are explicitly separated.
3. Effective profile revisions are immutable and bound to Executions.
4. Overlays are deltas, not independent authorities.
5. Canonical UER foundations are used rather than rebuilt.
6. Remaining canonical execution paths use ExecutionId/ExecutionBoundary semantics.
7. RuntimeEvents carry canonical Execution identity on execution-scoped paths.
8. Child Executions inherit/narrow authority and budget.
9. Tool/capability permission intersection is monotonic.
10. Explicit caller scope cannot widen runtime policy.
11. Skills cannot silently expand host ToolProfile availability.
12. Meaningful side effects cross fresh authorization.
13. Required causal evidence precedes meaningful work at declared fail-closed boundaries.
14. Retry/redelivery/resume ownership is explicit.
15. Cancellation works across supported Execution subtrees/providers.
16. Checkpoint/recovery preserves canonical identity and required supported restore state.
17. Existing Execution-tree checkpoint foundations are reused and hardened, not replaced.
18. Side-effect recovery uses explicit idempotency/fence semantics without claiming universal exactly-once effects.
19. All canonical model calls cross CE.
20. Model requests have declared reconstructability level and evidence.
21. Artifacts can exist outside model context and retain lineage.
22. Compaction does not delete canonical evidence.
23. Runtime inspection answers effective-state and “why” questions from canonical facts.
24. Capability health/readiness is a projection, not a policy authority.
25. Activation is atomic from consumer perspective.
26. Live reconfiguration creates a new immutable revision and keeps in-flight Executions pinned.
27. Runtime credentials resolve by reference and remain non-model-visible by default.
28. Sandbox/isolation is execution-scoped and inspectable where supported.
29. External/subagent providers operate as governed child Executions.
30. Background work reuses UER rather than a second lifecycle.
31. External events enter through authenticated/deduplicated/durable admission.
32. Instruction Skills do not grant authority.
33. Platform Plugins remain package/control-plane coordination; runtime dynamic registration stays scoped and reversible.
34. Runtime invariants remain domain-owned with shared execution/diagnostic infrastructure.
35. Human continuation absence never auto-approves.
36. API/SDK/ACP/MCP/hosts do not create private execution/policy/tool/context semantics.
37. Generated catalogs/schemas are freshness-gated where used.
38. Memory/RAG keep domain ownership and project into models only through CE.
39. AHI remains proposal/evidence/governance driven.
40. Governed Runtime Evolution cannot start before its safety foundations are proven.
41. Security claims are bounded to declared threat/failure models.
42. Scenario Proofs exist for the most important cross-domain invariants.
43. Canonical CURRENT documentation does not knowingly contradict the code paths it directs implementers to change.

---

# 53. Immediate next action

The next implementation activity is **P0A — As-built re-baseline**.

It is an audit/documentation synchronization phase, not a code-feature phase.

After P0A, work proceeds to P0B safety closure and only then P0C execution/durability convergence.
