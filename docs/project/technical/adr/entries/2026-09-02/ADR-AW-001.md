# ADR-AW-001: Autonomous Work domain ownership and Virtual Worker classification

| Field | Value |
|---|---|
| **Status** | Accepted — architecture and planning only; runtime implementation NOT STARTED |
| **Date** | 2026-09-02 |
| **Deciders** | Intergrax platform architecture |
| **Related** | [`architecture/AUTONOMOUS_WORK.md`](../../../../architecture/AUTONOMOUS_WORK.md) · [`plan/AUTONOMOUS_WORK.md`](../../../../maintainers/plans/AUTONOMOUS_WORK.md) · [`COLLABORATIVE_WORK.md`](../../../../architecture/COLLABORATIVE_WORK.md) · [`CODE_CRAFT.md`](../../../../architecture/CODE_CRAFT.md) |

## Context

Intergrax is extending from task-oriented agent execution toward persistent autonomous work: systems that own a durable business responsibility, remain available over time, react to events, proactively evaluate goals, create or accept work, dispatch governed executions, recover from obstacles and continue responsibility across many runs.

The central architectural question is whether a Virtual Worker should be:

1. a specialized Agent,
2. a Tier-3 application,
3. a cross-layer feature under `docs/project/capabilities`,
4. a new full canonical domain under `docs/project/architecture`, or
5. only a documentation/marketing concept.

The repository already defines the distinction:

- `docs/project/architecture/<DOMAIN>.md` ↔ `docs/project/maintainers/plans/<DOMAIN>.md` is used for a reusable platform domain with canonical ownership and implementation truth;
- `docs/project/capabilities/architecture/<FEATURE>.md` ↔ `docs/project/capabilities/plan/<FEATURE>.md` is used for cross-layer feature coordination that composes multiple domains without replacing their ownership.

The architecture/gap audit and independent review established that Virtual Worker requires genuinely new platform semantics with no existing single owner:

- `WorkerDefinition`,
- `WorkerInstance`,
- durable `Responsibility`,
- `WorkerGoal`,
- worker lifecycle,
- event/schedule/goal wake-up semantics,
- worker-level recovery orchestration,
- capability-acquisition decision semantics,
- worker→execution correlation,
- worker control-plane semantics.

At the same time, many mechanisms must remain owned elsewhere: Principal/authority, WorkItem, Execution, Agents, policy/HITL, CodeCraft, Sandbox, Memory, Observability, Diagnostics and Application Hosting.

## Alternatives considered

### 1. `VirtualEmployeeAgent(BaseAgent)`

Rejected.

It would collapse durable responsibility, lifecycle, identity binding, work intake, recovery, budgets, policy and evidence into the cognitive layer. It would also encourage an immortal `while true` model loop and duplicate platform mechanisms.

### 2. Tier-3 `Virtual Workforce` application as owner

Rejected as primary ownership.

An application may provide worker builder, fleet UI, dashboard, approvals and controls, but reusable worker semantics must be available to multiple applications. Application-owned worker types would violate platform reuse and create product-local ownership.

### 3. `VIRTUAL_WORKFORCE` as capability feature only

Rejected as primary ownership.

A future `VIRTUAL_WORKFORCE` cross-layer feature may coordinate Autonomous Work + Collaborative Work + Governance + CodeCraft + Observability + Hosting. However, feature documentation cannot be the source of truth for new durable runtime/domain contracts.

This is the same distinction already used by Multiplayer AI: `MULTIPLAYER_AI` is a cross-layer feature while `COLLABORATIVE_WORK` owns core collaborative semantics.

### 4. Extend Collaborative Work to own Worker

Rejected.

Collaborative Work owns who collaborates, membership, delegation, effective authority and the collaborative work plane. A worker's durable responsibility, goals, lifecycle and autonomous wake-up/recovery semantics are different concerns. Merging them would turn Collaborative Work into a broad organizational-agent domain.

### 5. Extend Unified Execution Runtime / Nexus

Rejected.

Execution owns Task/Run/Attempt/Execution lifecycle. Worker responsibility deliberately outlives any execution and may exist while no task is active.

### 6. Create dedicated `AUTONOMOUS_WORK` domain

Accepted.

## Decision

1. Introduce canonical domain pair:

```text
docs/project/architecture/AUTONOMOUS_WORK.md
docs/project/maintainers/plans/AUTONOMOUS_WORK.md
```

2. `AUTONOMOUS_WORK` is the canonical technical domain name.

3. `Virtual Worker` is the primary product-facing abstraction implemented by the domain.

4. `Virtual Workforce` is reserved for multi-worker product/cross-layer positioning and may later receive a capability pair only when a real composition across domains requires feature-level coordination.

5. Autonomous Work owns:

- WorkerDefinition,
- WorkerInstance,
- Responsibility,
- WorkerGoal,
- worker lifecycle,
- work-intake/wake-up semantics,
- proactive goal-evaluation semantics,
- worker-level obstacle recovery orchestration,
- capability acquisition decision semantics,
- worker-level correlations/profile composition,
- **long-horizon work continuity semantics**,
- **durable work orientation requirements** (including Work Continuity State semantics).

6. Autonomous Work explicitly does not own:

| Concern | Canonical owner |
|---|---|
| Principal, Membership, Delegation, effective authority | Collaborative Work |
| business WorkItem/Assignment | Collaborative Work |
| Task/Run/Attempt/Execution | Unified Execution Runtime / Nexus |
| agent cognition | Agent Contracts / Reasoning |
| policy enforcement / HITL | Governed Execution / Reliability-HITL |
| generated code | CodeCraft |
| execution isolation | existing Sandbox runtime/substrate |
| memory storage | Memory |
| context assembly / token budget | Context Engineering |
| durable context optimization / reuse | UCL / Token Optimization |
| external knowledge retrieval | RAG |
| execution evidence | Observability / HOS |
| problem evidence/diagnostics | Diagnostics |
| process lifecycle | Application Hosting |
| product UX | Tier-3 application |

7. Worker identity and authority remain separate:

```text
WorkerInstance
  → explicit Collaborative Principal binding
  → Membership / Authority / Delegation
  → Governed Execution
```

Worker role, responsibility or goal never grants permission.

8. Persistent worker availability must be event/schedule/goal-driven. An unbounded continuous LLM loop is not the canonical runtime model.

9. Adaptive capability recovery follows reuse-before-create and risk tiers A0-A4. CodeCraft is canonical for generated executable code but is not the first recovery action for every failure.

10. Capability may expand within policy; authority may never self-expand.

11. Durable generated capability promotion is a separate governed control-plane lifecycle; ephemeral CraftResult does not silently become a global production tool/integration.

12. Virtual Worker cannot depend on full history replay or process-local context for continuity.

## Consequences

### Positive

- Creates one canonical owner for worker semantics.
- Keeps Agent, Principal, WorkItem and Execution identities distinct.
- Preserves existing domain investment instead of duplicating runtime mechanisms.
- Makes Virtual Worker reusable across many applications and business roles.
- Enables enterprise lifecycle, governance, observability, budgets and recovery without an application-local super-agent.
- Creates a clean product story: Autonomous Work domain → Virtual Worker abstraction → future Virtual Workforce product/capability.

### Negative

- Adds a new domain pair and long implementation program.
- Requires explicit bridges to Collaborative Work, Execution, Governance, CodeCraft and Observability.
- Mature Virtual Workforce depends on future Collaborative Work MP-2+ semantics.
- Production autonomous code recovery is blocked by existing CodeCraft/Sandbox hardening gaps.

## Compliance / invariants

The decision requires at minimum:

- `WorkerInstance != AgentDefinition != AgentRun`,
- `WorkerInstance != Collaborative Principal`,
- `WorkerInstance != WorkItem != Task != Execution`,
- responsibility survives zero or many executions,
- capability growth cannot expand authority,
- CodeCraft + approved Sandbox is the only generated-code execution path,
- required hardened isolation fails closed if unavailable,
- policy DENY is never treated as a recoverable obstacle,
- HOS/RuntimeEvent remains execution source of truth,
- worker control-plane mutations are governed/evidenced,
- worker persistence is independent of host-process lifetime.

## Implementation gate

Runtime implementation begins only after an independent review of the canonical `AUTONOMOUS_WORK` architecture/plan pair.

The first implementation slice is AW-1A semantic contracts only. Do not begin with a Virtual Workforce UI, infinite worker loop, Recovery Controller or CodeCraft adaptation before the contracts/ownership gate is accepted.
