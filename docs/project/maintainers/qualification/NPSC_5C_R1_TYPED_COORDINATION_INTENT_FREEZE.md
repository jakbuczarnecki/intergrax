# NPSC-5C/R1-F — Typed Coordination Intent Freeze

**Status:** `FROZEN / PASS`

**Verdict:** **PASS** (freeze certification)

**Date:** 2026-09-09

**Branch:** `development`

**Task:** NPSC-5C/R1-F — Typed Coordination Intent Freeze + External Decision Contract Requirement

**Baseline SHA (NPSC-5B frozen):** `473b790bcbae18cc5669015264018259783a5839` *(pre-R1 coordination-intent checkpoint; H1 hardening landed on same revision)*

**R1 implementation SHA:** `edcb51949b8622de2fe6c17eae6b245c8056d94d`

**H1 hardening SHA:** `473b790bcbae18cc5669015264018259783a5839`

**Predecessors:** NPSC-5A `FROZEN / PASS` · NPSC-5B `FROZEN / PASS` · NPSC-5C/R1 `FORMAL PASS` · NPSC-5C/R1-H1 `PASS`

---

## 1. Formal verdict

```text
NPSC-5C/R1 = FROZEN / PASS
NPSC-5C/R2 = BLOCKED ON DECISION OWNER
NPSC-5C/R3 = NOT STARTED
```

NPSC-5C/R1 establishes a **thin semantic coordination-intent layer** above frozen NPSC-5A (single delegation) and NPSC-5B (bounded fan-out). Decision System integration is **explicitly external** and **not required** for R1 runtime operation.

---

## 2. Canonical execution model

```text
Producer (deterministic / future Decision-backed)
        |
        v
CoordinationIntentPlanner (neutral protocol)
        |
        v
CoordinationIntent
        |
        v
CoordinationIntentExecutor
        +-- SINGLE  --> MultiAgentCoordinationService (NPSC-5A)
        |
        +-- FAN_OUT --> BoundedMultiAgentFanOutService (NPSC-5B)
```

Bypassing these entrypoints for equivalent coordination work requires an explicit architecture revision.

---

## 3. Frozen public surface

| Artifact | Module | Role |
| -------- | ------ | ---- |
| `CoordinationIntentId` | `coordination_intent.py` | Stable intent audit identity |
| `CoordinationContributionId` | `coordination_intent.py` | Stable contribution identity |
| `CoordinationExecutionMode` | `coordination_intent.py` | `SINGLE` / `FAN_OUT` semantic shape |
| `CoordinationContribution` | `coordination_intent.py` | One bounded specialist contribution |
| `CoordinationIntent` | `coordination_intent.py` | Semantic multi-agent work request |
| `CoordinationIntentPlanner` | `coordination_intent.py` | Neutral producer protocol |
| `CoordinationContributionBinding` | `coordination_intent_executor.py` | Per-contribution runtime lease binding |
| `CoordinationIntentBinding` | `coordination_intent_executor.py` | Execution-time binding (not part of intent) |
| `CoordinationIntentExecutor` | `coordination_intent_executor.py` | Validate → route to NPSC-5A or NPSC-5B |
| `CoordinationIntentResult` | `coordination_intent_executor.py` | Discriminated aggregate outcome |
| `CoordinationIntentContractError` | `coordination_intent.py` | Expected contract failure |

Private helpers (`_validate_and_index_binding`, `_materialize_*`, etc.) are **not** frozen; public semantics and ownership are.

---

## 4. Freeze invariants

### 4.1 Intent is semantic

`CoordinationIntent` describes **what** coordinated work is required — not **how** it is scheduled or **who** physically executes it. Intent carries no runtime topology.

### 4.2 No physical agent identity on semantic contributions

Semantic contributions do **not** carry `agent_id`, `agent_instance_id`, `lease_id`, `ExecutionId`, `OrchestrationSlotId`, or `GraphNodeId`. Physical resolution remains in Agent Distribution / Execution.

### 4.3 SINGLE routing

`CoordinationExecutionMode.SINGLE` always executes through `MultiAgentCoordinationService`. Not through `DelegatedSubtaskService`, `ChildExecutionRunner`, or Nexus directly.

### 4.4 FAN_OUT routing

`CoordinationExecutionMode.FAN_OUT` always executes through `BoundedMultiAgentFanOutService`. Scheduling remains exclusively in frozen NPSC-5B / Nexus.

### 4.5 Typed binding

Runtime lease binding is resolved by `CoordinationContributionId` → `CoordinationContributionBinding`. Not by tuple index, zip position, or implicit order.

### 4.6 Referential integrity (H1 baseline)

Fail-closed on: missing binding, extra binding, duplicate binding id, unknown binding id. Reordered binding tuples are legal.

### 4.7 Order semantics

Result order follows **intent contribution order**. Binding order is **irrelevant**. `ordering ≠ identity`.

### 4.8 Decision optional

NPSC-5C/R1 operates without Decision System. Canonical legal path:

```text
Application / deterministic producer
  → CoordinationIntent
  → CoordinationIntentExecutor
  → NPSC
```

Decision is not a runtime foundation dependency.

### 4.9 Decision does not execute

Future Decision integration may produce typed semantic information only. Decision must never route to Nexus, GraphExecutor, specialist agents, or leases.

### 4.10 Authority

`CoordinationIntent` does not mint authority. Canonical authority remains on active Execution, principal, and `ParentExecutionAuthority`.

### 4.11 Budget

`CoordinationIntent` does not create planner, coordination, or fan-out shadow budget ledgers. Budget remains canonical Execution responsibility.

### 4.12 Errors

Expected contract errors: `CoordinationIntentContractError`. Unexpected programming errors propagate. Broad `except Exception` swallowing is forbidden.

### 4.13 No second runtime

No `CoordinationRuntime`, `PlannerRuntime`, or `MultiAgentPlanningRuntime`. NPSC-5C remains a thin semantic layer.

---

## 5. NPSC-5A dependency (frozen)

| Dependency | Usage |
| ---------- | ----- |
| `MultiAgentCoordinationService` | Sole SINGLE-mode execution owner |
| `CoordinationRequest` / `CoordinationDelegation` / `CoordinationResult` | Materialized from intent + binding |
| `CoordinationPolicy` | Optional per-contribution selection constraints |
| `TaskCapabilityResolutionRequest` | Capability requirement on contributions |

NPSC-5A frozen regression: **PASS** (see §8).

---

## 6. NPSC-5B dependency (frozen)

| Dependency | Usage |
| ---------- | ----- |
| `BoundedMultiAgentFanOutService` | Sole FAN_OUT-mode execution owner |
| `FanOutRequest` / `FanOutItem` / `FanOutResult` | Materialized from intent + binding |
| `effective_fan_out_max_concurrency` | Semantic concurrency preference resolution |
| `MAX_FAN_OUT_ITEMS` / `MAX_FAN_OUT_CONCURRENCY` | Platform caps |

NPSC-5B frozen regression: **PASS** (see §8).

---

## 7. Decision optionality and external dependency

```text
DEPENDENCY:
  Decision-owned typed semantic coordination artifact

OWNER:
  Decision System session

CONSUMER:
  NPSC-5C/R2

STATUS:
  BLOCKED ON EXTERNAL OWNER
```

NPSC-5C/R1 does **not** implement Decision integration. Required Decision semantics are documented in:

`docs/project/maintainers/architecture/NPSC_5C_DECISION_INTEGRATION_REQUIREMENT.md`

---

## 8. Test requalification

### NPSC-5C/R1

```bash
uv run pytest tests/unit/agent_distribution/test_coordination_intent.py -q
uv run pytest tests/unit/agent_distribution/test_coordination_intent_executor.py -q
uv run pytest tests/unit/runtime/architecture/test_npsc5c_coordination_intent_gate.py -q
```

### NPSC-5A frozen regression

```bash
uv run pytest tests/unit/agent_distribution/test_multi_agent_coordination.py -q
uv run pytest tests/unit/runtime/architecture/test_npsc5a_multi_agent_coordination_gate.py -q
uv run pytest tests/unit/runtime/architecture/test_npsc5a_coordination_delegation_e2e.py -q
```

### NPSC-5B frozen regression

```bash
uv run pytest tests/unit/runtime/architecture/test_npsc5b_final_production_fanout_fanin_qualification.py -q
uv run pytest tests/unit/runtime/architecture/test_npsc5b_bounded_multi_agent_fanout_gate.py -q
```

**Result:** all suites **PASS** at freeze certification.

---

## 9. Architecture gates

| Gate | Path |
| ---- | ---- |
| NPSC-5C static ownership / hygiene | `tests/unit/runtime/architecture/test_npsc5c_coordination_intent_gate.py` |
| NPSC-5C contract tests | `tests/unit/agent_distribution/test_coordination_intent.py` |
| NPSC-5C executor + binding tests | `tests/unit/agent_distribution/test_coordination_intent_executor.py` |

Gate proofs include: no Nexus / GraphExecutor / ChildExecutionRunner imports in NPSC-5C modules; SINGLE routes only through `MultiAgentCoordinationService`; FAN_OUT routes only through `BoundedMultiAgentFanOutService`; no physical agent identity on semantic intent types.

---

## 10. R2 entry conditions

NPSC-5C/R2 may start only when:

1. Decision owner publishes a public typed coordination semantic artifact.
2. Artifact is committed to `development`.
3. Decision owner qualifies its own contract.
4. This session re-reads the exact public contract.
5. No string/metadata parsing is required for projection.

### R2 stop conditions

R2 remains **BLOCKED** if the artifact:

- contains only free text or rationale;
- requires metadata parsing;
- selects physical agents;
- contains runtime topology;
- requires NPSC to modify Decision-owned code.

### Future handshake (not implemented in R1)

```text
Decision System
    ↓ public typed artifact
NPSC Decision projection adapter (future ownership: this session)
    ↓
CoordinationIntent
    ↓
CoordinationIntentExecutor
```

Decision System must **not** duplicate `CoordinationIntent` into its namespace. Consumer adapter projects Decision public artifact → `CoordinationIntent`.

---

## 11. Known exclusions (not in R1 freeze)

```text
Decision-backed planning
Decision → CoordinationIntent adapter
post-fan-in Decision synthesis
retry/recovery
checkpoint/resume
advanced cancellation
governance extensions
quorum/voting
recursive fan-out
replay/evidence
```

---

## 12. Enterprise definition of FROZEN

`FROZEN` means: public semantics stable, ownership stable, entrypoints stable, identity semantics stable, failure semantics stable, routing stable, tests enforce invariants, changes require explicit architecture revision.

`FROZEN` does **not** mean code can never change. Bugfixes remain possible through separate hardening tasks.

---

## 13. Related documents

| Document | Role |
| -------- | ---- |
| `docs/project/maintainers/architecture/NPSC_5_MULTI_AGENT_PRODUCTION_ARCHITECTURE.md` | NPSC-5 series hub |
| `docs/project/maintainers/architecture/NPSC_5C_DECISION_INTEGRATION_REQUIREMENT.md` | External Decision contract requirement |
| `docs/project/maintainers/qualification/NPSC_5A_CANONICAL_MULTI_AGENT_COORDINATION_CONTRACTS.md` | NPSC-5A frozen baseline |
| `docs/project/maintainers/qualification/NPSC_5B_FINAL_PRODUCTION_FANOUT_FANIN_QUALIFICATION.md` | NPSC-5B frozen baseline |

---

## 14. Next step

```text
THIS SESSION: WAITING FOR DECISION OWNER
```

Do not proceed to NPSC-5C/R2 adapter work until Decision owner delivers a qualified public typed contract.
