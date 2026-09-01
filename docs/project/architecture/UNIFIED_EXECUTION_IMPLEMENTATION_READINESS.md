# Unified Execution - Implementation Readiness Gate

**Status:** Final pre-runtime consistency audit (UE-DOC-0.10)  
**Classification:** `SUPPORTING_AUDIT` / `GATE` - subordinate to [`UNIFIED_EXECUTION_ARCHITECTURE.md`](UNIFIED_EXECUTION_ARCHITECTURE.md); **not** architecture authority, **not** a domain pair, **not** an implementation map  
**Owner:** Intergrax Platform Architecture (gate coordination)  
**Audience:** Principal architects, UE-1+ implementers, audit reviewers  
**Authority chain:** UEA (`UEA-INV-001`..`021`) → domain architecture hubs → [`UNIFIED_EXECUTION_IMPLEMENTATION_MAP.md`](UNIFIED_EXECUTION_IMPLEMENTATION_MAP.md) → **this gate** → UE-1+ slices

**Subordinate links:** [`UNIFIED_EXECUTION_ARCHITECTURE.md`](UNIFIED_EXECUTION_ARCHITECTURE.md) · [`UNIFIED_EXECUTION_IMPLEMENTATION_MAP.md`](UNIFIED_EXECUTION_IMPLEMENTATION_MAP.md) · [`UNIFIED_EXECUTION_RUNTIME.md`](UNIFIED_EXECUTION_RUNTIME.md) · [`../maintainers/plans/UNIFIED_EXECUTION_RUNTIME.md`](../maintainers/plans/UNIFIED_EXECUTION_RUNTIME.md)

---

## 1. Baseline / concurrency

| Item | Value |
|------|-------|
| **Start pin** | `fc7c76c999e3d49d0532c4bdd07941c688e2553c` |
| **Pin ancestor check** | **PASS** - pin is ancestor of audited HEAD |
| **Audited baseline (HEAD)** | `fc7c76c999e3d49d0532c4bdd07941c688e2553c` |
| **Branch** | `development` |
| **Concurrency between start pin and HEAD** | **None** - start pin equals HEAD |
| **Concurrent commits reviewed** | `fc7c76c99` - DIAG-6 operator `DiagnosticReadService` read projection (runtime + OBS doc) |

**DIAG-6 concurrency verdict:** DIAG-6 adds a read-only operator projection over persisted `ProblemId` records and bounded DIAG-2→4 reconstruction. It does **not** mint `ExecutionId`, does **not** own execution lifecycle, and does **not** introduce a competing identity hierarchy. Compatible with UEA-INV-015/016 and OBS DIAG-6 non-goals.

**Working tree at audit close:** unrelated local edits outside UE scope (`connected_source_discovery.py`, proof outputs) - not included in gate commit.

---

## 2. Authority hierarchy

Audit posture: skeptical external Principal Architect - disprove readiness where possible.

| Layer | Role | Gate finding |
|-------|------|--------------|
| **UEA** | Frozen meta-architecture, `UEA-INV-001`..`021` | Internally consistent; no invariant weakened |
| **Domain hubs** | Semantic owners (UER, Nexus, Agent, Tools, OBS, GOV, REL, BG) | Aligned to UEA target; CURRENT/debt explicitly labeled |
| **Implementation map** | Migration mapping only | Consistent with UEA; DAG valid; dispositions defensible |
| **Plans** | Delivery sequencing | Subordinate; no architecture override detected |

Lower layers do not contradict higher authority on frozen semantics.

---

## 3. Cross-domain consistency matrix

| Domain pair | Fundamental model | Identity | Boundary | Strategy | Verdict |
|-------------|-------------------|----------|----------|----------|---------|
| UEA ↔ UER | Execution fundamental unit | Five-ID target aligned | Boundary coordinates, not absorbs | inference/agentic/orchestration | **CONSISTENT** |
| UEA ↔ Nexus | Nexus schedules child Executions | No `OrchestrationRunId` | Child re-enters boundary | Orchestration-only | **CONSISTENT** |
| UEA ↔ Agent/UAEP | Agent ≠ Execution | UAEP agent-specific | Agentic behind boundary | UAEP owns progression | **CONSISTENT** |
| UEA ↔ Tools | Orthogonal axes | Tool loop under one ExecutionId | ToolRuntime executes calls | Selection narrows only | **CONSISTENT** |
| UEA ↔ CE/Memory | CE assembles; Memory durable | No identity mint | CE no lifecycle | ToolResult → CE path | **CONSISTENT** |
| UEA ↔ LLM/Streaming | PARTIAL ≠ FINAL | Stream budget canonical | Governance before release | No StreamingRuntime | **CONSISTENT** |
| UEA ↔ GOV/Budget | Narrow-only authority | Admission distinct | RunBudget single ledger | No strategy-owned ledger | **CONSISTENT** |
| UEA ↔ REL/HITL | Retry taxonomy A–E | Pause preserves ids | HITL decision vs lifecycle | No auto new Attempt on HITL | **CONSISTENT** |
| UEA ↔ OBS/DIAG | Record vs interpret | OBS does not mint | DIAG derived only | ProblemId ≠ ExecutionId | **CONSISTENT** |
| UEA ↔ BG/Distributed | Transport ≠ runtime | Same-work redelivery | Worker re-entry | Envelope preserves ids | **CONSISTENT** (target) |

**Cross-domain verdict:** **PASS** - no unresolved architecture contradictions.

---

## 4. Identity consistency matrix

| ID / relation | Minting owner (target) | Retry / pause / redelivery | Competing use forbidden? | Verdict |
|---------------|------------------------|----------------------------|--------------------------|---------|
| `TaskId` | Task/intake | Preserved except new Task | Not substituted by transport id | **OK** |
| `RunId` | UER / execution admission | Preserved; whole-Run retry same Run | No `OrchestrationRunId` | **OK** |
| `AttemptId` | UER / execution lifecycle | New only on whole-Run retry (class C) | Not per local retry/redelivery | **OK** (target) |
| `ExecutionId` | Execution lifecycle layer | Preserved; tree via `parent_execution_id` | Not NodeId/Agent/ProblemId | **OK** (target; not in code yet) |
| `EventId` | Event factory / OBS contract | Causal via `parent_event_id` | Not substitute for Execution lineage | **OK** |
| `parent_execution_id` | Execution tree | Canonical lineage | Not mandatory on every RuntimeEvent | **OK** |
| `parent_event_id` | Event causality | Distinct from execution lineage | Cannot replace `parent_execution_id` | **OK** |
| `NodeId` | Orchestration definition | Topology only | ≠ `ExecutionId` | **OK** |
| Transport ids | Queue/broker/worker | Infrastructure only | ≠ Run/Attempt/Execution | **OK** |
| `ProblemId` | DIAG lifecycle (derived) | Stable recurrence bucket | ≠ ExecutionId, ≠ root cause | **OK** |

**DIAG-6 check:** `DiagnosticReadService` reads `ProblemId` and occurrence views only; no execution identity authority introduced.

---

## 5. Execution Boundary consistency

| Check | Result |
|-------|--------|
| Every independently schedulable Execution enters canonical boundary | **YES** (target) |
| Every child Execution re-enters same boundary | **YES** - UEA-INV-021 |
| Direct inference path avoids Nexus | **YES** |
| Agentic path: Boundary → AgentExecutor → AgentEngine → UAEP | **YES** |
| Orchestration: Nexus requests child Executions through boundary | **YES** |
| `GraphExecutor` → `AgentEngine` direct | **CURRENT debt** - mapped UE-7B; not target |
| Boundary coordinates; does not own PolicyEngine, budget ledger, OBS store, DIAG, checkpoint backend | **YES** - UEA-INV-018 |

**Anti-god-object verdict:** Implementation map §4 explicitly rejects central mega-runtime; coordination interfaces only.

---

## 6. Strategy consistency

| Requirement | Verdict |
|-------------|---------|
| Developer says WHAT; platform resolves HOW | **OK** |
| Strategies: inference, agentic, orchestration, future | **OK** |
| StrategyResolver deterministic from capabilities - not LLM router | **OK** (UE-3A scope defined) |
| Resolver does not invent topology | **OK** - UEA-INV-020 |
| Dynamic topology → validated OrchestrationDefinition before Nexus | **OK** - ORCHESTRATION + Nexus hubs |

---

## 7. Agent / tool / context / stream consistency

| Area | Verdict | Notes |
|------|---------|-------|
| Iterative tool use under one ExecutionId | **OK** | Ordinary LLM↔tool iterations = internal steps |
| Nexus not required for tool loop | **OK** | Agentic/inference paths |
| `bounded_react` under UAEP mechanics | **OK** | Not competing lifecycle owner |
| Tool selection narrows only | **OK** - TOOLS-INV-02 |
| ToolResult → CE fragments | **OK** (target wiring UE-6C) |
| Memory ≠ auto-persist every tool result | **OK** |
| Streaming orthogonal; PARTIAL ≠ tool trigger | **OK** |
| Raw stream → user → later DENY forbidden | **OK** |
| No StreamingRuntime | **OK** |

---

## 8. Nexus / orchestration consistency

| Check | Verdict |
|-------|---------|
| Nexus owns WHAT EXECUTES NEXT | **OK** |
| Nexus does not own Run lifecycle, ExecutionId authority, budget, GOV, OBS, DIAG | **OK** |
| NodeId ≠ ExecutionId; one node → many Executions | **OK** |
| Nested orchestration legal; same Run/Attempt | **OK** |
| No OrchestrationRunId / second graph runtime | **OK** |

Stale-language search (§41): no canonical doc states "every execution/task runs through Nexus" as **target**; CURRENT coupling documented as debt.

---

## 9. Governance / budget / reliability consistency

| Area | Verdict |
|------|---------|
| Authority inheritance narrow-only | **OK** - UEA-INV-009 |
| RunBudget single canonical ledger | **OK** - UEA-INV-010 |
| Parallel fan-out must not overcommit parent | **OK** (target UE-8B) |
| Retry taxonomy A–E aligned across UEA, UER, REL, OBS | **OK** |
| HITL: Governance owns decision; UER owns pause/resume | **OK** |
| Cancellation follows Execution tree | **OK** |
| Side-effectful tools: governed, budgeted, not blindly retried | **OK** |
| Failure semantics per strategy documented | **OK** - sufficient for implementation |

---

## 10. OBS / DIAG consistency

| Check | Verdict |
|-------|---------|
| RuntimeEvent target spine includes ExecutionId | **OK** |
| OBS records; does not mint ExecutionId | **OK** |
| DIAG interprets; does not mint ExecutionId | **OK** |
| `parent_event_id` ≠ `parent_execution_id` | **OK** |
| `parent_execution_id` not mandatory on every event | **OK** |
| DIAG-5D ProblemId stable; DIAG-6 read projection derived | **OK** |
| Audit evidence vs optional telemetry distinguishable | **OK** |

---

## 11. Distributed / checkpoint consistency

| Check | Verdict |
|-------|---------|
| Same-work redelivery preserves Task/Run/Attempt/Execution | **OK** (target) |
| Worker re-enters boundary with same identity | **OK** |
| Checkpoint does not mint identity | **OK** |
| Tree-aware checkpoint target defined | **OK** - UE-9C |
| No AgentCheckpoint / NexusCheckpoint competing identity | **OK** |

**CURRENT debt:** `bootstrap_background_execution` mints new Attempt per worker boundary - mapped UE-9A; architecture answer exists.

---

## 12. Implementation-map disposition audit

Challenged components - all dispositions consistent with frozen UEA:

| Component | Disposition | Challenge result |
|-----------|-------------|------------------|
| `UnifiedTaskRunner` | KEEP_AND_REWIRE | Not legacy-as-target |
| `NexusLoop` | SPLIT_RESPONSIBILITY | Orchestration internals preserved |
| `GraphExecutor` | TRANSFORM | Child Execution admission required |
| `AgentEngine` / `UAEPExecutor` | KEEP / KEEP_AND_REWIRE | Agent-specific preserved |
| `ToolInvocationPattern` | KEEP | Mechanics only |
| `RuntimeExecutionContext` | KEEP | Agent scope - not universal |
| `RunBudget` | TRANSFORM | Hierarchical - not replaced |
| `ProblemGroupingEngine` / `ProblemLifecycleEngine` | KEEP / KEEP_AND_REWIRE | Derived DIAG only |
| `AgentExecutionResult` | DEPRECATE as universal | Neutral result in UE-5B |

**Verdict:** Map does not preserve legacy as target; does not require big-bang rewrite (§30).

---

## 13. Migration DAG audit

```text
ExecutionId → Boundary → strategies → OBS/checkpoint/background/DIAG
```

| DAG check | Verdict |
|-----------|---------|
| Missing prerequisite | **None blocking** |
| Cycle | **None** |
| Incorrect ordering | **None requiring architecture decision** |
| Implicit architecture decision in ordering | **None** |

**Notable ordering (validated):** Nexus child Execution requires boundary skeleton (UE-7 after UE-4); budget hierarchy after execution tree (UE-8 after UE-7); OBS/DIAG ExecutionId after contract (UE-9B).

**Migration DAG verdict:** **PASS**

---

## 14. UE-1..UE-10 readiness matrix

| Slice | Can proceed without new architecture? | Hidden question? | Verdict |
|-------|--------------------------------------|------------------|---------|
| **UE-1A** | ExecutionId types in `intergrax/contracts/` | None | **READY** |
| **UE-1B** | Boundary skeleton + compat shim | None - interfaces only | **READY** |
| **UE-1C** | `execution.execute` facade | None | **READY** |
| **UE-2A/B** | Neutral request DTO + Task bridge | Names not frozen - implementation choice | **READY** |
| **UE-3A/B** | StrategyResolver + runner rewire | Capability signals listed in map §5 | **READY** |
| **UE-4A/B** | Admission hooks + active binding | None | **READY** |
| **UE-5A/B** | Structured output + neutral result | None | **READY** |
| **UE-6A/B/C** | Agentic + tool loop + CE bridge | UAEP/pattern wiring - ownership frozen | **READY** |
| **UE-7A/B** | Child Execution API; remove direct engine | None | **READY** |
| **UE-8A/B** | Authority + budget hierarchy | None | **READY** |
| **UE-9A/B/C** | Background identity, events, checkpoint tree | None | **READY** |
| **UE-10** | Scenarios A–J proof harness | Evidence classes defined below | **READY** |

**UE-1..UE-10 verdict:** **READY** - all slices derivable from UEA + domain docs + implementation map.

---

## 15. UE-1A first-slice readiness (deep review)

| Requirement | Status |
|-------------|--------|
| Add typed `ExecutionId` + validators + minting | **Clear** - mirror `TaskId`/`RunId` pattern in `execution_identity.py` |
| `parent_execution_id` contract | **Defined** in UEA §4, UER identity section |
| Structural compatibility | **Additive** - no behavior cutover in UE-1A |
| Excludes: boundary layout, strategy, Nexus rewrite, DIAG, budget | **Honored** in map §31 UE-1A exclusions |
| No dict/getattr discovery | **Explicit** in map §3 anti-patterns |

**UE-1A verdict:** **READY** - safely implementable as first runtime slice.

---

## 16. Proof matrix A–L

Maps UEA §28 scenarios to implementation proof obligations. **Do not implement in UE-DOC-0.10.**

| ID | Scenario | Architecture property proved | Prerequisite slice | Evidence class |
|----|----------|------------------------------|--------------------|----------------|
| **A** | Direct inference | Boundary-gated inference without Nexus/Agent | UE-2, UE-3, UE-5 | Integration test + RuntimeEvents |
| **B** | Iterative agentic tool use | One ExecutionId; tool loop; CE feedback | UE-6 | UAEP + tool loop test |
| **C** | Orchestration child Execution | Nexus child via boundary re-entry | UE-7 | Graph scenario unit |
| **D** | Child authority narrowing | effective child ≤ parent | UE-8A | Policy admission test |
| **E** | Parallel budget reservation | No parent overcommit | UE-8B | Fan-out load test |
| **F** | HITL pause/resume | Identity preserved | UE-4 + REL alignment | Pause/resume integration |
| **G** | Cancellation subtree | Cancel Execution → descendants | UE-4 + UER cancel | Subtree cancel test |
| **H** | Checkpoint/recovery identity | Resume without id remint | UE-9C | Round-trip resume |
| **I** | Worker redelivery same identity | UEA-INV-011 | UE-9A | Redelivery integration |
| **J** | OBS/DIAG reconstruction | Read-only traversal; no mint | UE-9B | Event spine + DIAG read |
| **K** | Streaming final response | FINAL semantics; no pre-governance leak | UE-5 + LLM adapters | Stream governance test |
| **L** | Structured output | `output_type` validation contract | UE-5A | Schema conformance |

---

## 17. Skeptic challenge - "one LLM + memory + tools + LangGraph"

A minimal agent stack **cannot** satisfy Intergrax platform guarantees without Unified Execution semantics:

| Platform requirement | Why simple stack insufficient |
|---------------------|------------------------------|
| Canonical five-ID hierarchy + Execution tree | LangGraph state ≠ governed ExecutionId/parent chain |
| Child execution isolation + boundary re-entry | Flat graph nodes bypass per-child GOV/budget/OBS |
| Authority inheritance (narrow-only) | Ad-hoc tool allowlists expand permissions |
| Hierarchical RunBudget + reservations | Private ReAct counters compete with ledger |
| HITL pause/resume with identity preservation | Framework pause ≠ auditable governance decision |
| Distributed same-identity redelivery | Queue redelivery mints new attempts in naive designs |
| Checkpoint tree + UAEP cursors | Single checkpoint blob ≠ tree-aware recovery |
| Causal evidence spine | Log stitching ≠ `parent_execution_id` + `parent_event_id` |
| DIAG derived Problems without identity mint | Error grouping ≠ execution truth |
| Multiple strategies (inference/agentic/orchestration) | Single graph topology conflates strategy with structure |
| Tool governance + side-effect safety | Ungoverned tool nodes duplicate external effects on retry |
| Streaming release governance | Raw provider stream to user violates GOV |

**Conclusion:** Unified Execution complexity maps to **concrete platform requirements** - not decorative abstraction. No unjustified complexity flagged for removal.

---

## 18. No sixth framework check

Searched UEA, implementation map, domain hubs for parallel engines.

| Forbidden artifact | Found as target? |
|--------------------|------------------|
| NewExecutionFrameworkV2 / ExecutionGraphV2 | **NO** |
| AgentRuntimeV2 / StreamingRuntime / ToolLoopRuntime / ReActRuntime | **NO** (explicitly rejected) |
| InferenceRuntime with private lifecycle | **NO** |
| NexusV2 / second event spine / second checkpoint / second budget / second DIAG | **NO** |

Small typed coordinators/adapters allowed - map uses boundary skeleton, not new framework.

---

## 19. Current vs target honesty

Docs correctly label **TARGET** vs **CURRENT** vs **MIGRATION GAPS**. Not claimed implemented:

- Canonical `ExecutionId` / neutral Execution Boundary
- Direct inference executor behind boundary
- StrategyResolver
- Nexus child re-entry
- Hierarchical execution budget
- Tree-aware checkpoint
- Same-identity worker redelivery
- Platform-wide streaming release governance
- Full ExecutionId OBS/DIAG propagation

Architecture readiness ≠ runtime completeness - **honest**.

---

## 20. Risk register

### Blocking risks

**Count: 0**

### Non-blocking implementation risks

| ID | Risk | Mitigation slice |
|----|------|------------------|
| R-01 | `GraphExecutor` direct `AgentEngine` bypass | UE-7B |
| R-02 | Background Attempt mint on redelivery | UE-9A |
| R-03 | `bounded_react` / UAEP wiring overlap | UE-6B |
| R-04 | Hierarchical budget incomplete | UE-8B |
| R-05 | Streaming governance not platform-wide | UE-5 + LLM path |
| R-06 | Dual lifecycle during compat shims | Feature flags + caller inventory |
| R-07 | Partial ExecutionId rollout | Phased optional→required fields |
| R-08 | UEA §24 follow-up note on OBS/UER tables - superseded by UE-DOC-0.6 alignment | Mechanical note update |

**Non-blocking count: 8**

---

## 21. Mechanical corrections performed (UE-DOC-0.10)

| File | Correction |
|------|------------|
| `UNIFIED_EXECUTION_IMPLEMENTATION_MAP.md` | UE-5A prerequisite `UE-2 inference` → `UE-2A, UE-3A (inference path)` |
| `UNIFIED_EXECUTION_ARCHITECTURE.md` | §24 follow-up note: OBS/UER ExecutionId alignment addressed in UE-DOC-0.6 |
| `UNIFIED_EXECUTION_ARCHITECTURE.md` | Gate cross-link to this document |
| `UNIFIED_EXECUTION_RUNTIME.md` | Gate status cross-link |
| `UNIFIED_EXECUTION_IMPLEMENTATION_MAP.md` | Gate cross-link |
| `maintainers/plans/UNIFIED_EXECUTION_RUNTIME.md` | Gate status line |

**UEA-INV-001..021:** unchanged.

---

## 22. Acceptance questions (§43)

| Question | Answer |
|----------|--------|
| Is Execution still the single fundamental runtime unit? | **YES** |
| Is there exactly one canonical Execution Tree? | **YES** |
| Does every independent child Execution re-enter the same boundary? | **YES** |
| Does direct inference avoid Nexus? | **YES** |
| Does ordinary iterative tool use avoid Nexus? | **YES** |
| Can Nexus orchestrate nested child Executions? | **YES** |
| Does AgentEngine/UAEP remain agent-specific? | **YES** |
| Does ToolInvocationPattern remain mechanics-only? | **YES** |
| Can selection, invocation and streaming vary independently? | **YES** |
| Can semantic tool selection expand permissions? | **NO** |
| Can PARTIAL stream output trigger tool execution? | **NO** |
| Can user-visible stream bypass required governance? | **NO** |
| Does client disconnect automatically cancel Execution? | **NO** |
| Does Observability create execution identity? | **NO** |
| Does DIAG create execution identity? | **NO** |
| Does ProblemId substitute for ExecutionId? | **NO** |
| Does worker redelivery create new Attempt in target? | **NO** |
| Does pause/resume create new Execution? | **NO** |
| Does whole-Run retry create new Attempt? | **YES** |
| Does child authority ever expand parent authority? | **NO** |
| Can parallel child reservations overcommit parent? | **NO** |
| Does checkpoint own identity? | **NO** |
| Can RuntimeEvent parent_event_id replace parent_execution_id? | **NO** |
| Must parent_execution_id be mandatory on every RuntimeEvent? | **NO** |
| Does the implementation map preserve legacy just because it exists? | **NO** |
| Does the map require a big-bang rewrite? | **NO** |
| Can UE-1A be implemented without another architecture design decision? | **YES** |
| Can all UE-1..UE-10 slices proceed from existing authorities? | **YES** |

---

## 23. Audit sections 1–35 summary

| Audit section | Result |
|---------------|--------|
| 1 Fundamental execution model | **PASS** |
| 2 Identity consistency | **PASS** |
| 3 One execution tree | **PASS** |
| 4 Execution boundary | **PASS** (CURRENT bypasses mapped) |
| 5 Anti-god-object | **PASS** |
| 6 Strategy model | **PASS** |
| 7 Direct inference | **PASS** |
| 8 Agentic execution | **PASS** |
| 9 Iterative tool use | **PASS** |
| 10 Tool orthogonality | **PASS** |
| 11 Context / memory | **PASS** |
| 12 Streaming | **PASS** |
| 13 Structured output / result | **PASS** |
| 14 Nexus ownership | **PASS** |
| 15 Topology vs execution tree | **PASS** |
| 16 Governance / authority | **PASS** |
| 17 Budget | **PASS** |
| 18 Retry taxonomy | **PASS** |
| 19 HITL | **PASS** |
| 20 Cancellation | **PASS** |
| 21 Checkpoint / recovery | **PASS** |
| 22 Observability | **PASS** |
| 23 DIAG | **PASS** |
| 24 Distributed execution | **PASS** |
| 25 Side effects / idempotency | **PASS** |
| 26 Failure semantics | **PASS** |
| 27 Implementation map consistency | **PASS** |
| 28 Dependency DAG | **PASS** |
| 29 UE-1..UE-10 implementability | **PASS** |
| 30 UE-1A special review | **PASS** |
| 31 UE-1B / boundary special review | **PASS** |
| 32 Proof matrix | **DEFINED** (§16) |
| 33 Skeptic challenge | **PASS** (§17) |
| 34 No sixth framework | **PASS** |
| 35 Current vs target honesty | **PASS** |

---

## 24. FINAL GATE

All PASS conditions met:

1. No unresolved architecture contradictions.
2. Ownership boundaries unambiguous.
3. Implementation map consistent with target.
4. Migration DAG has no blocking ambiguity.
5. UE-1..UE-10 can proceed without new architecture design.
6. UE-1A safely implementable as first slice.
7. Known CURRENT gaps are implementation work, not unanswered architecture questions.

**IMPLEMENTATION GATE: PASS**

Runtime implementation (UE-1+) may proceed. Runtime gate for UE-1A first slice: **OPEN**.
