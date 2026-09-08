# NPSC-5 — Multi-Agent Production Architecture

**Status:** `ACTIVE`

**Series owner:** Agent Distribution + frozen Execution Engine

**Current phase:** NPSC-5B — Bounded Multi-Agent Fan-Out / Fan-In

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
            +--> TaskCapabilityResolver
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
- capability resolution request — functional need, not concrete agent identity
- lease identity and application binding context
- optional `CoordinationPolicy` — typed selection constraints for audit / future policy wiring

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

- `TaskCapabilityResolver`
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

```text
Parent execution
        |
        v
BoundedMultiAgentFanOutService
        |
        +---- canonical coordination (item A)
        |
        +---- canonical coordination (item B)
        |
        +---- canonical coordination (item C)
        |
        v
deterministic fan-in (request order)
```

| Component | Package | Responsibility |
| --------- | ------- | -------------- |
| `FanOutRequest` | `intergrax/agent_distribution/` | Parent semantic intent for many bounded specialist contributions |
| `FanOutResult` | `intergrax/agent_distribution/` | Deterministic aggregate outcome in stable item order |
| `BoundedMultiAgentFanOutService` | `intergrax/agent_distribution/` | Validate fan-out intent → bounded parallel calls to `MultiAgentCoordinationService.coordinate` |
| `AsyncioSemaphoreBoundedFanOutExecutor` | `intergrax/agent_distribution/` | Default bounded concurrency executor (coordination-local; no lifecycle ownership) |
| `MultiAgentCoordinationService` | `intergrax/agent_distribution/` | Unchanged NPSC-5A single-delegation owner |
| `DelegatedSubtaskService` | `intergrax/agent_distribution/` | Unchanged child lifecycle orchestration owner |

**Fan-out owner:** `BoundedMultiAgentFanOutService` owns fan-out validation, bounded concurrency scheduling, and deterministic fan-in projection. It does **not** mint execution identity, acquire leases directly, or invoke child execution.

**Child lifecycle owner:** unchanged — each fan-out item still flows through `MultiAgentCoordinationService` → `DelegatedSubtaskService` → `ChildExecutionPort` → `ChildExecutionRunner`.

**Concurrency bound:** `max_concurrency` is validated (`> 0`, platform cap `MAX_FAN_OUT_CONCURRENCY`). At most that many delegations may be active concurrently for one fan-out operation.

**Deterministic result:** aggregate item order matches request order regardless of completion order.

**Partial failure:** per-item typed outcomes (`FanOutItemOutcome`) preserve successes and failures; aggregate result exposes `any_failed` / `all_succeeded` without substituting defaults.

**Not in NPSC-5B:** retry/recovery, global multi-agent budget allocation, planner routing, voting/swarm/quorum, durable checkpoint/resume, observability redesign.

---

## 9. Future NPSC-5 phases

| Phase | Scope |
| ----- | ----- |
| **NPSC-5A** | Single parent → single bounded specialist delegation contracts |
| **NPSC-5B** | Bounded fan-out / fan-in (implemented) |
| **NPSC-5C** | Planner-produced typed coordination intents |
| **NPSC-5F** | Audit / observability hooks expansion |

---

## 10. Hard fail conditions

NPSC-5 must never require:

- new root or child lifecycle engines;
- direct parent → `AgentExecutor` or `NexusLoop`;
- caller-generated execution identity as authority;
- legacy supervisor as production dependency;
- generic service locators or implicit fallback agents.
