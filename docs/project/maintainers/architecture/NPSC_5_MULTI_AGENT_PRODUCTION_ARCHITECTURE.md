# NPSC-5 — Multi-Agent Production Architecture

**Status:** `ACTIVE`

**Series owner:** Agent Distribution + frozen Execution Engine

**Current phase:** NPSC-5D — Multi-Agent Governance (**FROZEN / PASS** · R1+R2+R3 unified)

**R1 reconciliation:** [`NPSC_5B_CROSS_SYSTEM_FANOUT_OWNERSHIP_RECONCILIATION.md`](NPSC_5B_CROSS_SYSTEM_FANOUT_OWNERSHIP_RECONCILIATION.md)

**Branch:** `development`

---

## 1. Purpose

NPSC-5 establishes the production-grade multi-agent coordination boundary for Intergrax without introducing a second agent execution framework.

The platform already owns:

```text
capability resolution
→ agent discovery
→ capability matching
→ agent selection
→ task-scoped agent acquisition
→ delegated specialist invocation
→ ChildExecutionPort
→ ChildExecutionRunner
```

NPSC-5 layers coordination **above** this stack. Coordination decides **who** should perform a bounded subtask. It does **not** own execution lifecycle, identity minting, governance enforcement, or root execution startup.

---

## 2. Frozen execution engine (immutable)

| Responsibility | Canonical owner |
| -------------- | --------------- |
| Root lifecycle | `ExecutionRuntime` |
| Child lifecycle | `ChildExecutionRunner` |
| Identity minting | `identity_authority` |
| Bind / reset | `ExecutionBoundary` |
| Strategy routing | `StrategyExecutionRouter` |
| Host root execution | `HostTaskExecutionPort` |
| Orchestration backend | `Nexus` (private) |

No NPSC-5 component may take ownership of these responsibilities.

---

## 3. Ownership model (NPSC-5A)

| Component | Package | Responsibility |
| --------- | ------- | -------------- |
| `CoordinationRequest` | `intergrax/agent_distribution/` | Parent semantic intent for one bounded specialist contribution |
| `CoordinationResult` | `intergrax/agent_distribution/` | Audit-friendly projection over delegated subtask evidence |
| `MultiAgentCoordinationService` | `intergrax/agent_distribution/` | Validate coordination intent → invoke `DelegatedSubtaskService` |
| `DelegatedSubtaskService` | `intergrax/agent_distribution/` | Discovery, selection, lease, child execution orchestration |
| `ChildExecutionPort` | contract in agent distribution; adapter in runtime | Canonical child execution boundary |
| `ChildExecutionRunner` | `intergrax/runtime/execution/child.py` | Child lifecycle owner |

**Package decision:** coordination lives in `intergrax/agent_distribution/` because it expresses functional specialist need and delegates through existing agent-distribution services. No new top-level package was introduced.

---

## 4. Coordination boundary

```text
Parent Agent / Orchestration Policy
            |
            v
MultiAgentCoordinationService
            |
            v
DelegatedSubtaskService
            |
            +--> TaskCapabilityResolver (only for UNRESOLVED_TASK)
            +--> AgentDiscoveryStrategy
            +--> CapabilityMatcher
            +--> AgentSelectionStrategy
            +--> TaskScopedAgentService
            |
            v
ChildExecutionPort
            |
            v
ChildExecutionRunner
            |
            v
ExecutionBoundary
            |
            v
StrategyExecutionRouter
            |
            v
Specialist Agent
```

### Coordination request semantics

A coordination request contains:

- `coordination_id` — stable coordination audit identity
- `delegation_id` — reused delegated-subtask identity
- `task_scope_id` — must match canonical active execution task scope
- typed capability need (`AgentDistributionCapabilityNeed`) — either unresolved task intent (`TaskCapabilityResolutionRequest`) or already-resolved canonical capability authority (`AgentCapabilityRequirement`); never concrete physical agent identity
- lease identity and application binding context
- optional `CoordinationPolicy` — typed selection constraints for audit / future policy wiring

Capability need paths:

- `UNRESOLVED_TASK` → `TaskCapabilityResolver` → discovery / matching / selection
- `RESOLVED_REQUIREMENT` → resolver bypass → discovery / matching / selection

A coordination request must **not** contain caller-minted execution identity, Nexus handles, agent instances, or LLM clients.

### Parent authority

Delegation authority derives from the active parent execution:

```text
active parent ExecutionBoundary
      → canonical task scope (ActiveExecutionTaskScopePort)
      → coordination
      → DelegatedSubtaskService
      → ChildExecutionRunner
```

The coordination layer rejects caller-supplied task scopes that do not match the canonical active scope.

---

## 5. Agent selection ownership

Selection remains platform-owned through injectable strategies:

- `TaskCapabilityResolver` (only when capability need is `UNRESOLVED_TASK`)
- `AgentDiscoveryStrategy`
- `CapabilityMatcher`
- `AgentSelectionStrategy`

The coordination layer is deterministic. It does not invoke an LLM to choose specialists.

---

## 6. Governance and budget relationship

NPSC-5A is governance-compatible from day one:

- coordination validates intent;
- `ChildExecutionRunner` enforces authority narrowing and budget narrowing;
- child requested permission scopes cannot exceed parent authority;
- child requested budget cannot exceed parent available budget.

No second governance or budget engine was introduced.

---

## 7. Legacy supervisor isolation

`intergrax/supervisor/**` is legacy. NPSC-5 production modules must not import it or depend on LangGraph as canonical runtime.

---

## 8. Recursion note

Nested delegation is already possible when a specialist acquired through `DelegatedSubtaskService` itself coordinates another subtask within the same active execution tree. NPSC-5A does not add voting, swarms, or unbounded supervisor recursion.

---

## 8.1 NPSC-5B — Bounded fan-out / fan-in

> **R1 ownership freeze:** Fan-out scheduling and bounded parallelism are **not** canonical Agent Distribution responsibilities. See [`NPSC_5B_CROSS_SYSTEM_FANOUT_OWNERSHIP_RECONCILIATION.md`](NPSC_5B_CROSS_SYSTEM_FANOUT_OWNERSHIP_RECONCILIATION.md).
>
> **Historical progression:** R2 CORRECTION REQUIRED → R3 BLOCKED → R4 implemented → final qualification PASS. See [`NPSC_5B_R3_NEXUS_FANOUT_CONTRACT_REQUIREMENT.md`](NPSC_5B_R3_NEXUS_FANOUT_CONTRACT_REQUIREMENT.md) for R3 contract requirements.
>
> **Current status:** **FROZEN / PASS** — qualification: [`NPSC_5B_FINAL_PRODUCTION_FANOUT_FANIN_QUALIFICATION.md`](../qualification/NPSC_5B_FINAL_PRODUCTION_FANOUT_FANIN_QUALIFICATION.md)

### Canonical integration (R4 — implemented)

```text
FanOutRequest (semantic contract — Agent Distribution)
        |
        v
BoundedMultiAgentFanOutService (validation + fan-in projection)
        |
        v
FanOutOrchestrationPort
        |
        v
OrchestrationTopologySubmissionPort
        |
        v
canonical NexusLoop graph_executor (composition-root wired)
        |
        v
typed dynamic topology + bounded scheduling
        |
        +---- per slot: child Execution → MultiAgentCoordinationService → DelegatedSubtaskService
        |
        v
typed per-slot outcomes → FanOutResult
```

| Component | Package | Status |
| --------- | ------- | ------ |
| `FanOutRequest` / `FanOutItem` | `intergrax/agent_distribution/` | `KEEP` — semantic contracts |
| `FanOutResult` / `FanOutItemOutcome` | `intergrax/agent_distribution/` | `KEEP` — semantic contracts |
| `FanOutOrchestrationPort` | `intergrax/agent_distribution/` | `KEEP` — semantic boundary |
| `BoundedMultiAgentFanOutService` | `intergrax/agent_distribution/` | `KEEP` — thin adapter (no scheduler) |
| `fan_out_orchestration_adapter.py` | `intergrax/runtime/execution/` | `KEEP` — thin R4 consumer adapter |
| `multi_agent_fanout_orchestration.py` | removed in R4 | `RETIRED` |
| `AsyncioSemaphoreBoundedFanOutExecutor` | removed in R2 | `REMOVED` |
| `BoundedFanOutExecutor` | removed in R2 | `REMOVED` |
| `MultiAgentCoordinationService` | `intergrax/agent_distribution/` | `KEEP` — single-delegation owner |
| `DelegatedSubtaskService` | `intergrax/agent_distribution/` | `KEEP` — specialist child orchestration |

**Canonical fan-out scheduler owner:** Nexus (via `ExecutionStrategy.ORCHESTRATION`), not Agent Distribution.

**Agent Distribution fan-out ownership:** semantic contracts (`FanOutRequest`, `FanOutResult`) and per-slot specialist coordination only.

**Child lifecycle owner:** unchanged — each slot flows through `MultiAgentCoordinationService` → `DelegatedSubtaskService` → `ChildExecutionPort` → `ChildExecutionRunner`.

**Platform caps:** `MAX_FAN_OUT_ITEMS`, `MAX_FAN_OUT_CONCURRENCY` remain validation limits on fan-out contracts.

**Not in NPSC-5B:** retry/recovery, global multi-agent budget allocation, planner routing, voting/swarm/quorum, durable checkpoint/resume, observability redesign.

---

## 8.2 NPSC-5C — Typed coordination intent + Decision integration (FROZEN)

NPSC-5C introduces a **semantic** coordination intent layer above frozen NPSC-5A / NPSC-5B, plus a pure Decision → NPSC projection path. It describes *what* multi-agent work is requested without owning scheduling, physical agent selection, or execution lifecycle.

```text
AuthoritativeAcceptedDecision
        |
        v
project_authoritative_accepted_decision_coordination (pure projection)
        |
        v
CoordinationIntent
        |
        v
CoordinationIntentExecutor
        +-- SINGLE --> MultiAgentCoordinationService (NPSC-5A)
        |
        +-- FAN_OUT --> BoundedMultiAgentFanOutService (NPSC-5B) --> Nexus
```

| Component | Package | Responsibility |
| --------- | ------- | -------------- |
| `DecisionCoordinationSemantic` | `intergrax/contracts/` | Decision-owned semantic WHAT |
| `project_authoritative_accepted_decision_coordination` | `intergrax/agent_distribution/` | Pure deterministic projection (no I/O) |
| `CoordinationIntent` / `CoordinationContribution` | `intergrax/agent_distribution/` | Semantic multi-agent work request |
| `CoordinationIntentPlanner` | `intergrax/agent_distribution/` | Neutral producer protocol (not LLM-specific) |
| `CoordinationIntentExecutor` | `intergrax/agent_distribution/` | Validate intent → governance admission → route to frozen NPSC-5A or NPSC-5B |
| `CoordinationIntentBinding` | `intergrax/agent_distribution/` | Runtime execution binding (leases, task scope) — not part of intent |

**Ownership:** Decision owns WHAT; Agent Distribution owns WHO; Execution owns lifecycle; Nexus owns HOW/WHEN for FAN_OUT scheduling.

**Qualification:** [`NPSC_5C_FINAL_QUALIFICATION_AND_FREEZE.md`](../qualification/NPSC_5C_FINAL_QUALIFICATION_AND_FREEZE.md)

**Hard boundaries preserved:**

- no `NexusLoop` / `GraphExecutor` / `ChildExecutionRunner` imports in NPSC-5C projection modules;
- no direct `DelegatedSubtaskService` calls from the intent executor;
- no physical `agent_id` / `lease_id` fields on semantic intent contributions;
- no `decision_system → agent_distribution` reverse dependency;
- authority and budget remain on the active Execution path.

---

## 8.1 NPSC-5D/R1 — semantic coordination governance admission

**Status:** NPSC-5D/R1 **FROZEN / PASS** · NPSC-5D/R2 **FROZEN / PASS** · NPSC-5D/R3 **FROZEN / PASS** · NPSC-5D **FROZEN / PASS**

| Component | Package | Responsibility |
| --------- | ------- | -------------- |
| `MultiAgentCoordinationGovernanceRequest` | `intergrax/contracts/` | Typed stage-1 coordination admission facts (no physical agent identity) |
| `MultiAgentCoordinationGovernancePort` | `intergrax/contracts/` | Public evaluator boundary reusing canonical `PolicyDecision` |
| `MultiAgentCoordinationGovernanceBoundary` | `intergrax/runtime/governance/` | Fail-closed admission over configured evaluator |
| `build_multi_agent_coordination_governance_request` | `intergrax/agent_distribution/` | Caller adapter from `CoordinationIntent` + binding |
| `materialize_coordination_intent_binding` | `intergrax/agent_distribution/` | Runtime binding projection from governed host Task |
| `CollaborativeWorkAuthorityResolverPort` | `intergrax/autonomous_work/execution_authority_admission.py` | Shared consumer seam (AW-3B) — Collaborative Work owns semantics |

**Canonical evaluation location:** `CoordinationIntentExecutor` after intent/binding validation and authoritative applicability reconciliation, before `MultiAgentCoordinationService` / `BoundedMultiAgentFanOutService`.

**Governance model (R1):** coordination-level all-or-nothing admission only; per-contribution physical authorization deferred to NPSC-5D/R2.

**Authority reconciliation (R1-H1):**

```text
NPSC delegation (specialist contribution) ≠ Collaborative Work authority delegation
RequestIdentity = identity reference only — never authority proof
Collaborative Work owns collaborative authority via CollaborativeWorkAuthorityResolverPort
Governance consumes authoritative EffectiveAuthorityDecision + coordination policy via compose_policy_decisions
Execution effective authority remains independent and monotonic on the active Execution path
```

**Collaborative applicability (R1-H2):** authoritative source is governed Execution / host Task context — not `CoordinationIntent`, Decision artifact, or caller workspace omission. `workspace_id` present → `REQUIRED`; absent → `NOT_APPLICABLE`; missing/malformed host → fail-closed.

**Qualification:** [`NPSC_5D_R1_FINAL_QUALIFICATION_AND_FREEZE.md`](../qualification/NPSC_5D_R1_FINAL_QUALIFICATION_AND_FREEZE.md)

**Distinction:** NPSC `CoordinationPolicy` remains Agent Distribution selection semantics — not platform Governance policy.

---

## 8.2 NPSC-5D/R2 — physical delegation governance admission

**Status:** NPSC-5D/R2 **FROZEN / PASS** · NPSC-5D/R3 **FROZEN / PASS** · NPSC-5D **FROZEN / PASS**

| Component | Package | Responsibility |
| --------- | ------- | -------------- |
| `PhysicalDelegationGovernanceRequest` | `intergrax/contracts/` | Typed post-selection physical delegation facts |
| `PhysicalDelegationGovernancePort` | `intergrax/contracts/` | Public evaluator boundary reusing canonical `PolicyDecision` |
| `PhysicalDelegationGovernanceBoundary` | `intergrax/runtime/governance/` | Fail-closed admission over configured evaluator |
| `build_physical_delegation_governance_request` | `intergrax/agent_distribution/` | Caller adapter from delegated subtask selection facts |

**Canonical evaluation location:** `DelegatedSubtaskService` after `require_selected_identity`, before `build_acquisition_plan` / `TaskScopedAgentService.acquire`.

**Governance model (R2):** per-contribution physical admission for the exact selected specialist identity — no re-selection, no AC-3 trust duplication, no lease/child side effects before ALLOW.

**Evaluation point:** `GovernanceEvaluationPoint.MULTI_AGENT_DELEGATION` (distinct from R1 `MULTI_AGENT_COORDINATION`).

### 8.2.1 NPSC-5D/R2-H1 — governed continuation identity preservation

**Status:** R2-H1 preserves exact post-selection physical delegation identity across Agent Distribution boundaries.

| Artifact | Package | Responsibility |
| -------- | ------- | -------------- |
| `PhysicalDelegationGovernedContinuation` | `intergrax/contracts/` | Immutable typed continuation binding `delegation_id`, exact `selected_identity`, `governance_result` / evidence, task scope, application binding |

**Propagation:** `DelegatedSubtaskGovernanceRequiresHuman` → `GovernanceRequiresHumanError.continuation` → `FanOutItemFailure.continuation` when `GOVERNANCE_REQUIRES_HUMAN`. Ordinary failures carry no continuation payload.

**Qualification:** [`NPSC_5D_R2_FINAL_QUALIFICATION_AND_FREEZE.md`](../qualification/NPSC_5D_R2_FINAL_QUALIFICATION_AND_FREEZE.md)

**R3 delivered:** canonical HITL pause/approval/grant/resume bound to exact `PhysicalDelegationGovernedContinuation` — no re-selection. See §8.2.2.

**Distinction from R1:** R1 `REQUIRE_HUMAN` is semantic coordination admission (`CoordinationGovernanceRequiresHuman`); R2-H1 is exact selected physical delegation (`PhysicalDelegationGovernedContinuation`).

### 8.2.2 NPSC-5D/R3 — canonical HITL governed continuation

**Status:** NPSC-5D/R3 **FROZEN / PASS** · NPSC-5D **FROZEN / PASS**

**Qualification:** [`NPSC_5D_R3_FINAL_QUALIFICATION_AND_FREEZE.md`](../qualification/NPSC_5D_R3_FINAL_QUALIFICATION_AND_FREEZE.md) · **NPSC-5D Final:** [`NPSC_5D_FINAL_MULTI_AGENT_GOVERNANCE_QUALIFICATION_AND_FREEZE.md`](../qualification/NPSC_5D_FINAL_MULTI_AGENT_GOVERNANCE_QUALIFICATION_AND_FREEZE.md)

#### 8.2.2.1 NPSC-5D/R3-H1 — truthful resume provenance and Nexus exact slot continuation

**Status:** R3-H1 incorporated into R3 freeze — governed resume without synthetic selection; Nexus-owned exact fan-out slot continuation.

| Artifact | Package | Responsibility |
| -------- | ------- | -------------- |
| `DelegatedSelectionProvenanceKind` | `intergrax/agent_distribution/` | Distinguishes real selection (`SELECTED`) from preserved governed resume (`PRESERVED_GOVERNED_CONTINUATION`) |
| `OrchestrationTopologyContinuationPort` | `intergrax/contracts/` | Canonical exact-slot continuation within a prior topology execution |
| `OrchestrationSlotContinuationExecutor` | `intergrax/contracts/` | Optional slot executor capability for resumed governed slots |
| `CanonicalOrchestrationTopologySubmissionPort` | `intergrax/runtime/execution/` | Stores in-process topology execution context; continues one registered slot without whole-topology resubmit |
| `FanOutCoordinationSlotExecutor.continue_slot` | `intergrax/runtime/execution/` | Projects Nexus slot continuation into `continue_governed_coordination` |

**Provenance rule:** resume does **not** constitute a new selection. `continue_governed_delegation` preserves exact `selected_identity` from `PhysicalDelegationGovernedContinuation` and records `PRESERVED_GOVERNED_CONTINUATION` without fabricating `AgentSelectionDecision`.

**Ownership:** Nexus resumes the exact blocked slot and scheduling context; Agent Distribution resumes physical delegation semantics (`continue_governed_coordination` / `continue_governed_delegation`). Governance/HITL remain canonical pause/grant owners.

**Fan-out:** `GOVERNANCE_REQUIRES_HUMAN` items register continuable slot identities on the topology execution record. Approval resumes **only** that slot — successful siblings are not re-executed; whole fan-out and whole topology resubmit are forbidden on the resume path.

---

## 9. Future NPSC-5 phases

| Phase | Scope |
| ----- | ----- |
| **NPSC-5A** | Single parent → single bounded specialist delegation contracts |
| **NPSC-5B** | Bounded fan-out / fan-in (**FROZEN / PASS**) |
| **NPSC-5C** | Typed coordination intent + Decision integration (**FROZEN / PASS**) |
| **NPSC-5D** | Multi-agent governance (**FROZEN / PASS** — R1+R2+R3 unified governance plane) |
| **NPSC-5E** | Retry / checkpoint / recovery |
| **NPSC-5F** | Audit / observability hooks expansion |

---

## 10. Hard fail conditions

NPSC-5 must never require:

- new root or child lifecycle engines;
- direct parent → `AgentExecutor` or `NexusLoop`;
- caller-generated execution identity as authority;
- legacy supervisor as production dependency;
- generic service locators or implicit fallback agents.
