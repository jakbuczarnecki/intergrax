# NPSC-5B/R1 — Cross-System Fan-Out Ownership Reconciliation

**Status:** `FROZEN` (R1 architecture decision) · **R2:** `CORRECTION REQUIRED` · **R3:** `BLOCKED ON SHARED CONTRACT`

**Series:** NPSC-5B — Bounded Multi-Agent Fan-Out / Fan-In

**Branch:** `development`

**Related:** [`NPSC_5_MULTI_AGENT_PRODUCTION_ARCHITECTURE.md`](NPSC_5_MULTI_AGENT_PRODUCTION_ARCHITECTURE.md)

---

## 1. Purpose

NPSC-5B/R1 freezes canonical ownership between Decision, Execution, Nexus orchestration, and Agent Distribution multi-agent coordination. R1 does **not** change runtime behavior; it resolves the competing fan-out / parallelism models before R2 implementation.

---

## 2. Current conflict

Three mechanisms currently overlap on parallelism and fan-out semantics:

| Mechanism | Location | What it does today |
| --------- | -------- | ------------------ |
| **Agent Distribution fan-out scheduler** | `BoundedMultiAgentFanOutService` → `AsyncioSemaphoreBoundedFanOutExecutor` | Bounded parallel `MultiAgentCoordinationService.coordinate` calls with local `asyncio.Semaphore`, deterministic fan-in |
| **Execution concurrency primitive** | `execute_concurrent_execution_work` in `concurrent_execution_work.py` | Unbounded parallel submission of pre-defined canonical `ExecutionRequest` values through `ExecutionWorkPort` |
| **Nexus orchestration ownership** | `ExecutionStrategy.ORCHESTRATION` → `OrchestrationExecutor` → `NexusLoop` / `GraphExecutor` | Topology traversal, dependency readiness, bounded node scheduling (`max_parallel_nodes`, `max_inflight_nodes`), fan-out/fan-in, orchestration failure semantics |

**Conflict:** NPSC-5B documentation (§8.1) assigns fan-out validation, bounded concurrency scheduling, and fan-in projection to `BoundedMultiAgentFanOutService`. Frozen Nexus and Orchestration architecture assign the same responsibilities to Nexus when work carries orchestration topology semantics (fan-out + bounded scheduling + fan-in + delegation topology). A second `asyncio.Semaphore` scheduler in Agent Distribution duplicates canonical ownership.

**Non-conflict (clarified):** `MultiAgentCoordinationService` single-delegation specialist resolution remains correctly owned by Agent Distribution. The conflict is the **parallel scheduling and fan-in aggregation layer**, not specialist discovery/selection.

---

## 3. Canonical ownership definitions

| Responsibility | Question answered | Canonical owner |
| -------------- | ----------------- | --------------- |
| **Decision** | What is the authoritative semantic decision? | Decision System |
| **Execution** | How does one Execution live and behave? | Execution System (`ExecutionRuntime`, `ChildExecutionRunner`, `ExecutionBoundary`) |
| **ExecutionWorkPort** | How does active Execution submit neutral child work? | Execution System (contract); consumers bind at composition root |
| **concurrent_execution_work** | How are N independent, pre-defined `ExecutionRequest` values executed in parallel without topology? | Execution System |
| **Nexus** | What independently schedulable work executes next, and how is orchestration topology coordinated? | Nexus (private backend behind `ExecutionStrategy.ORCHESTRATION`) |
| **Agent Distribution** | Which specialist capability/agent satisfies a specialist requirement? | Agent Distribution (`TaskCapabilityResolver`, discovery, matching, selection, `DelegatedSubtaskService`) |
| **NPSC Multi-Agent Coordination** | How is one bounded specialist contribution requested and projected? | Agent Distribution (`MultiAgentCoordinationService`) |

No single component may own more than one of these four boundaries: Decision, Execution lifecycle, Nexus orchestration control, Agent selection.

---

## 4. Canonical ownership table

| Surface | Owner | Role |
| ------- | ----- | ---- |
| Decision lifecycle / authoritative conclusion | Decision System | Semantic decision only; no scheduler, no Nexus, no fan-out |
| `ExecutionWorkPort` | Execution System | Neutral typed seam for child `ExecutionRequest` submission |
| `ChildExecutionWorkPort` | Execution System | Default adapter: `ChildExecutionRunner` + wired `StrategyExecutionRouter` delegate |
| `execute_concurrent_execution_work` | Execution System | Lowest-level parallel primitive for independent `ExecutionRequest` tuples; no topology, no specialist selection, no merge semantics |
| `StrategyExecutionRouter` | Execution System | Sole strategy resolution and backend routing (INFERENCE / AGENTIC / ORCHESTRATION) |
| `OrchestrationExecutor` / `NexusOrchestrationPort` | Execution composition → Nexus | Public Execution-layer orchestration delegate; Nexus implementation is private |
| Nexus scheduling / fan-out / fan-in / merge | Nexus | Bounded parallelism, dependency readiness, topology interpretation |
| `MultiAgentCoordinationService` | Agent Distribution | Single specialist coordination: validate intent → `DelegatedSubtaskService` |
| `BoundedMultiAgentFanOutService` | **R2: adapter only** (R1: conflict) | Today: duplicate scheduler; target: semantic adapter forwarding to Execution orchestration seam |
| `ChildExecutionPort` | Agent Distribution contract; runtime adapter | Specialist child invocation boundary (distinct from `ExecutionWorkPort`; used inside `DelegatedSubtaskService`) |

---

## 5. R1 answers (Q1–Q7)

### Q1 — Does `AsyncioSemaphoreBoundedFanOutExecutor` duplicate Execution/Nexus ownership?

**YES.** It implements bounded parallel scheduling and completion gathering via local `asyncio.Semaphore` and `asyncio.gather` at the Agent Distribution layer. Canonical bounded fan-out scheduling belongs to Nexus (`GraphExecutor` caps) or, for topology-free independent `ExecutionRequest` work, to the Execution concurrency primitive. Agent Distribution must not be a second scheduler.

### Q2 — Is `BoundedFanOutExecutor` a valid Agent Distribution variation point?

**NO — retire in R2.** Scheduling variation belongs at Execution (`concurrent_execution_work` extensions) or Nexus (orchestration policies), not Agent Distribution. A `FanOutExecutor` protocol at this layer blurs ownership and invites duplicate schedulers.

### Q3 — What should `BoundedMultiAgentFanOutService` become?

**C — pass work to the canonical orchestration port** (via Execution-owned `ExecutionWorkPort` with `ExecutionCapability.ORCHESTRATION`).

`FanOutRequest` / `FanOutItem` contracts remain as semantic multi-specialist intent (see Q4). The service must not retain a local scheduler. R2 rewires it to translate validated fan-out intent into orchestration submission through the Execution seam; per-item specialist resolution continues through Agent Distribution inside the orchestrated child path.

### Q4 — Which fan-out contracts are valuable domain contracts?

| Contract | R2 status | Rationale |
| -------- | --------- | --------- |
| `FanOutRequest` | `KEEP_AS_CONTRACT_ONLY` | Parent semantic intent for many bounded specialist contributions |
| `FanOutItem` | `KEEP_AS_CONTRACT_ONLY` | One specialist contribution within a fan-out operation |
| `FanOutResult` | `KEEP_AS_CONTRACT_ONLY` | Deterministic aggregate outcome in stable item order |
| `FanOutItemOutcome` | `KEEP_AS_CONTRACT_ONLY` | Per-item typed success/failure projection |

These are domain contracts independent of which layer performs scheduling.

### Q5 — Does Nexus have a sufficient public typed seam?

**YES** (via Execution layer; Nexus itself remains private).

Agent Distribution must **not** import `intergrax.runtime.nexus` or `NexusLoop`. The public seam is:

```text
ExecutionWorkPort
      ↓ ChildExecutionRunner (child lineage under active parent)
StrategyExecutionRouter
      └── ORCHESTRATION
            ↓ OrchestrationRouterDelegate (OrchestrationExecutor)
            ↓ NexusOrchestrationPort.handle_task (private Nexus)
```

Proof gates: DS-NEXUS-01 (`test_decision_execution_work.py`), DS-NEXUS-02 (`test_decision_orchestration_recovery.py`).

No new Nexus import surface is required in NPSC R2. Composition root wires `ExecutionWorkPort` with ORCHESTRATION backend configured.

### Q6 — Is `ExecutionWorkPort` sufficient for Agent Distribution to request ORCHESTRATION child work without knowing Nexus?

**YES.**

Exact path:

1. Composition root injects `ExecutionWorkPort` (typically `ChildExecutionWorkPort` backed by `StrategyExecutionRouter` with `orchestration_executor` configured).
2. Agent Distribution adapter builds `ExecutionRequest` with `capabilities={ExecutionCapability.ORCHESTRATION}` and topology-bearing input (or delegates to a host that does).
3. `port.execute(request)` → `ChildExecutionRunner` → `StrategyExecutionRouter` → `OrchestrationExecutor` → private Nexus.
4. Specialist resolution for individual topology slots remains Agent Distribution (`MultiAgentCoordinationService` / `DelegatedSubtaskService`) when the orchestration plan routes to specialist positions.

For **single** specialist delegation (NPSC-5A), the existing `ChildExecutionPort` path through `DelegatedSubtaskService` remains correct and does not require `ExecutionWorkPort`.

### Q7 — Does Decision System need changes for NPSC-5B?

**NO.**

Decision consumes/produces neutral typed contracts and uses optional `ExecutionWorkPort` without Nexus knowledge (`DECISION_SYSTEM.md` § Execution-hosted Decision work submission). NPSC-5B reconciliation is entirely between Execution, Nexus, and Agent Distribution.

---

## 6. Integration model (frozen)

```text
Decision System
      │ authoritative semantic decision / plan
      ▼
Execution System
      │ canonical work request (ExecutionRequest)
      ▼
Execution Strategy Resolution (StrategyExecutionRouter)
      │
      ├── simple independent concurrent work (no topology)
      │       → execute_concurrent_execution_work (Execution-owned)
      │
      └── orchestration topology (fan-out + fan-in + deps + bounded scheduling)
              ↓
        ORCHESTRATION capability
              ↓
            Nexus (private)
              ↓
       child Executions
              ↓
       Agent Distribution (per specialist slot)
              ↓
        specialist Execution
```

**Tier rule for NPSC-5B fan-out:** When work carries multi-unit fan-out, bounded scheduling, and deterministic fan-in over independently meaningful child contributions, it is orchestration topology work → Nexus via ORCHESTRATION. Agent Distribution owns **what specialist** satisfies each slot, not **when and how many** slots run in parallel.

**`concurrent_execution_work` boundary (confirmed):** Execution-owned primitive for N independent, already-defined `ExecutionRequest` values without dependency graph, topology decisions, specialist selection, orchestration merge semantics, replan, or delegation topology. It must not become a competing orchestration engine. Note: current implementation uses unbounded `asyncio.gather`; bounded parallelism for non-orchestration concurrent work is an Execution-layer concern, not Agent Distribution.

---

## 7. DO NOT DUPLICATE rules

| Forbidden duplicate | Owner |
| ------------------- | ----- |
| Second scheduler (`asyncio.Semaphore` or equivalent for multi-unit fan-out) | Nexus (orchestration) / Execution (simple concurrent primitive) |
| Second `ExecutionWorkPort` | Execution |
| Second child execution API | Execution (`ChildExecutionRunner`) |
| Decision-owned Nexus adapter | Decision (must use neutral `ExecutionWorkPort` only) |
| Agent Distribution-owned orchestration engine | Agent Distribution |
| Nexus-owned agent selection engine | Nexus |
| Agent Distribution copy of `ExecutionWorkPort` | Agent Distribution |
| Local substitute orchestration contract when Execution seam exists | Any NPSC module |

---

## 8. NPSC-5B component classification (R2)

| Component | R2 status | Rationale |
| --------- | --------- | --------- |
| `FanOutRequest` | `KEEP_AS_CONTRACT_ONLY` | Valuable parent multi-specialist intent contract |
| `FanOutItem` | `KEEP_AS_CONTRACT_ONLY` | One bounded contribution within fan-out |
| `FanOutResult` | `KEEP_AS_CONTRACT_ONLY` | Deterministic fan-in aggregate |
| `FanOutItemOutcome` | `KEEP_AS_CONTRACT_ONLY` | Per-item typed outcome |
| `BoundedMultiAgentFanOutService` | `DEPRECATE_IN_R2` → adapter to orchestration port | Remove local scheduling; retain validation and contract projection |
| `BoundedFanOutExecutor` | `REMOVE_IN_R2` | Invalid variation point at Agent Distribution layer |
| `AsyncioSemaphoreBoundedFanOutExecutor` | `REMOVE_IN_R2` | Duplicate scheduler; violates Nexus ownership |
| `MAX_FAN_OUT_ITEMS` | `KEEP` | Platform validation cap on fan-out contract |
| `MAX_FAN_OUT_CONCURRENCY` | `KEEP` | Platform cap; enforce at orchestration/Execution policy layer in R2, not via AD semaphore |

---

## 9. Cross-session ownership matrix

| Surface | Owner | NPSC may modify? |
| ------------------------------ | ----------------------- | --------------------- |
| Decision lifecycle contracts | Decision | NO |
| Decision strategies | Decision | NO |
| `ExecutionWorkPort` | Execution | NO in R1; NO without Execution session in R2 |
| `concurrent_execution_work` | Execution | NO in R1; NO without Execution session in R2 |
| Execution lifecycle | Execution | NO |
| Nexus orchestration internals | Nexus | NO |
| `MultiAgentCoordinationService` | Agent Distribution / NPSC | YES |
| Agent selection contracts | Agent Distribution | YES |
| NPSC fan-out adapter / contracts | NPSC | YES after R1 freeze |
| `OrchestrationExecutor` / strategy routing | Execution | NO without Execution session |

---

## 10. Cross-session contract status

| Question | Answer |
| -------- | ------ |
| Decision changes required | **NO** |
| Execution changes required | **YES** — minimal shared orchestration topology submission + typed slot outcome contracts (R3) |
| Nexus public seam sufficient | **NO** — `ExecutionWorkPort` + ORCHESTRATION routes strategy, but no typed dynamic topology submission to canonical `NexusLoop` host |
| Missing contract | **YES** — see [`NPSC_5B_R3_NEXUS_FANOUT_CONTRACT_REQUIREMENT.md`](NPSC_5B_R3_NEXUS_FANOUT_CONTRACT_REQUIREMENT.md) |
| R2 production status | **RETIRED** — invalid mini-runtime removed in R4 |
| R4 production status | **PASS** — `fan_out_orchestration_adapter.py` consumes canonical topology submission |

---

## 11. R2 migration plan

**Status: R4 COMPLETE** (R2 mini-runtime retired; canonical adapter implemented).

Completed in R2:

1. **Freeze contracts** — `FanOutRequest`, `FanOutItem`, `FanOutResult`, `FanOutItemOutcome` retained as Agent Distribution semantic contracts.
2. **Remove duplicate scheduler** — deleted `AsyncioSemaphoreBoundedFanOutExecutor` and `BoundedFanOutExecutor`.
3. **Rewire `BoundedMultiAgentFanOutService`** — validation + `FanOutOrchestrationPort` adapter shell.

**Retired (R4):** invalid `multi_agent_fanout_orchestration.py` removed.

### R4 canonical path (implemented)

```text
FanOutRequest
→ BoundedMultiAgentFanOutService (validate)
→ FanOutOrchestrationPort
→ OrchestrationTopologySubmissionPort
→ canonical NexusLoop graph_executor
→ real child Execution per slot
→ MultiAgentCoordinationService → DelegatedSubtaskService
→ typed fan-in projection
→ FanOutResult
```

See [`NPSC_5B_R3_NEXUS_FANOUT_CONTRACT_REQUIREMENT.md`](NPSC_5B_R3_NEXUS_FANOUT_CONTRACT_REQUIREMENT.md).

---

## 12. DO NOT TOUCH IN R2 WITHOUT OWNER APPROVAL

| Surface | Owner session |
| ------- | ------------- |
| `intergrax/contracts/decision*` | Decision |
| `intergrax/decision/**` | Decision |
| `intergrax/runtime/decision*` | Decision |
| `intergrax/runtime/execution/decision*` | Decision |
| `intergrax/runtime/execution/execution_work_port.py` | Execution |
| `intergrax/runtime/execution/concurrent_execution_work.py` | Execution |
| `intergrax/runtime/execution/runtime.py` | Execution |
| `intergrax/runtime/execution/orchestration.py` | Execution / Nexus |
| `intergrax/runtime/nexus/**` internals | Nexus |
| `docs/project/architecture/DECISION_SYSTEM.md` | Decision |

---

## 13. Enterprise extension model

Allowed variation points (pluginability at correct layer):

| Variation point | Layer |
| --------------- | ----- |
| `DecisionStrategy` | Decision |
| `AgentSelectionStrategy` | Agent Distribution |
| `AgentDiscoveryStrategy` | Agent Distribution |
| Execution strategy implementation | Execution |
| Nexus orchestration policies | Nexus |

**Not allowed:** generic `FanOutExecutor` at Agent Distribution solely to abstract scheduling that belongs to Nexus/Execution.

---

## 14. Architecture conflicts

**R3 BLOCKED** — R1 scheduler conflict was retired, but R2 introduced an orchestration mini-runtime bypass. Minimal shared Execution/Nexus contracts are required before NPSC-5B can reach production qualification. Decision System impact: **NONE**.

---

## 15. Allowed variation points (NPSC post-R2)

| Component | Allowed role |
| --------- | ------------ |
| `MultiAgentCoordinationService` | Single specialist coordination |
| `CoordinationRequest` / `CoordinationResult` | Specialist intent contracts |
| `FanOutRequest` / `FanOutResult` | Multi-specialist semantic contracts (no scheduler) |
| Fan-out → orchestration adapter | Thin translation to `ExecutionWorkPort` |
