# NPSC-5A — Canonical Multi-Agent Coordination Contracts

**Status:** `CERTIFIED`

**Verdict:** pending runtime proof

**Date:** 2026-09-08

**Branch:** `development`

**Task:** NPSC-5A — establish canonical coordination contracts above `DelegatedSubtaskService`

---

## 1. Discovery summary

### Existing delegation services

| Component | Role |
| --------- | ---- |
| `DelegatedSubtaskService` | Discovery, selection, lease, child execution orchestration |
| `DelegatedSubtaskRequest` | Functional specialist need inside active parent execution |
| `DelegatedSubtaskInvocation` | Typed payload + permission/budget narrowing |
| `DelegatedSubtaskResult` | Audit-friendly delegated outcome |
| `ChildExecutionPort` | Canonical child execution boundary contract |
| `ChildExecutionRunnerPort` | Runtime adapter to `ChildExecutionRunner` |
| `TaskCapabilityResolver` | Task → capability requirement |
| `AgentDiscoveryStrategy` | Candidate discovery |
| `CapabilityMatcher` | Requirement matching |
| `AgentSelectionStrategy` | Deterministic specialist selection |
| `TaskScopedAgentService` | Lease acquisition / release |
| `SpecialistInvocationPort` | Resolve lease → executable delegate |

### Existing child execution boundary

```text
DelegatedSubtaskService
  → ChildExecutionPort (ChildExecutionRunnerPort)
  → ChildExecutionRunner
  → ExecutionBoundary
```

### Legacy supervisor production dependency

`NO` — NPSC-5A production module does not import `intergrax.supervisor`.

### Duplicate concepts

No duplicate execution framework introduced. `CoordinationRequest` composes existing `DelegationId`, `TaskScopeId`, lease identity, and capability resolution contracts.

---

## 2. Implementation evidence

### Production module

`intergrax/agent_distribution/multi_agent_coordination.py`

| Artifact | Purpose |
| -------- | ------- |
| `CoordinationId` | Stable coordination audit identity |
| `CoordinationRequest` | Parent bounded specialist intent |
| `CoordinationDelegation` | Typed payload + child authority narrowing |
| `CoordinationResult` | Projection over `DelegatedSubtaskResult` |
| `CoordinationFailureCode` | Bounded failure taxonomy |
| `CoordinationPolicy` | Typed optional selection constraints |
| `MultiAgentCoordinationService` | Validate → delegate → project |

### Package ownership

Selected package: `intergrax/agent_distribution/`

Reason: coordination expresses functional specialist need and reuses agent-distribution discovery, selection, and lease services.

New top-level package: `NO`

---

## 3. Architecture gates

| Gate | Path |
| ---- | ---- |
| NPSC-5A static ownership / hygiene | `tests/unit/runtime/architecture/test_npsc5a_multi_agent_coordination_gate.py` |
| NPSC-5A end-to-end delegation proof | `tests/unit/runtime/architecture/test_npsc5a_coordination_delegation_e2e.py` |

Gate proofs:

- no legacy supervisor imports;
- no execution identity minting in coordination module;
- no `ExecutionRuntime` / `AgentExecutor` / `NexusLoop` ownership;
- no `_orchestration_backend` access;
- coordination delegates only through `DelegatedSubtaskService`;
- no task-introduced prohibited typing/reflection patterns.

---

## 4. Contract tests

`tests/unit/agent_distribution/test_multi_agent_coordination.py`

Coverage:

- request validation (empty coordination id, invalid delegation id);
- eligible specialist selection;
- no eligible specialist fail-closed;
- strategy injection preserved;
- canonical task scope required / mismatch rejected;
- delegated subtask invocation path preserved;
- child execution lineage via `ChildExecutionRunner`;
- permission scope propagation and escalation blocked;
- budget propagation and escalation blocked;
- acquisition failure mapping;
- lease release after success and after child failure;
- child execution port usage (no direct runner bypass from coordination).

---

## 5. Regression commands

```bash
uv run pytest tests/unit/agent_distribution/test_multi_agent_coordination.py -q
uv run pytest tests/unit/runtime/architecture/test_npsc5a_multi_agent_coordination_gate.py -q
uv run pytest tests/unit/runtime/architecture/test_npsc5a_coordination_delegation_e2e.py -q
uv run pytest tests/unit/agent_distribution/ -q
uv run pytest tests/unit/runtime/execution/ tests/unit/runtime/interactions/ -q
```

Frozen gates: NPSC-3C, NPSC-3E, NPSC-3F, NPSC-3G, NPSC-4, NPSC-4.1, NPSC-4.2, UE-10R1-R4, UE-11GP (see final report).

---

## 6. Architecture document

`docs/project/maintainers/architecture/NPSC_5_MULTI_AGENT_PRODUCTION_ARCHITECTURE.md`
