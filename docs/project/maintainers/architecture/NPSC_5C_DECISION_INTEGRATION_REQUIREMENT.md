# NPSC-5C — Decision Integration Requirement

**Status:** `BLOCKED ON EXTERNAL OWNER`

**Series:** NPSC-5C — Typed Coordination Intent + Deterministic Routing

**Branch:** `development`

**Consumer:** NPSC-5C/R2 (Agent Distribution session)

**Producer owner:** Decision System session

**Related:**

- [`NPSC_5C_R1_TYPED_COORDINATION_INTENT_FREEZE.md`](../qualification/NPSC_5C_R1_TYPED_COORDINATION_INTENT_FREEZE.md) (R1 `FROZEN / PASS`)
- [`NPSC_5_MULTI_AGENT_PRODUCTION_ARCHITECTURE.md`](NPSC_5_MULTI_AGENT_PRODUCTION_ARCHITECTURE.md)
- [`NPSC_5B_R3_NEXUS_FANOUT_CONTRACT_REQUIREMENT.md`](NPSC_5B_R3_NEXUS_FANOUT_CONTRACT_REQUIREMENT.md) *(precedent: neutral cross-system contract requirement)*

---

## 1. Dependency registration

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

NPSC-5C/R1 is **frozen and operational without Decision**. This document records the **minimum semantic capability** required before NPSC-5C/R2 adapter work may begin.

---

## 2. Consumer requirement (semantics only)

The consumer requires a Decision-owned **public typed artifact** that enables unambiguous expression of:

| Capability | Required semantics |
| ---------- | ------------------ |
| Coordination execution shape | `SINGLE` or `FAN_OUT` |
| Typed contributions | Contribution identity, capability requirement, semantic / typed payload reference |
| Optional concurrency preference | Semantic bounded concurrency hint (not scheduler topology) |

The producer owner chooses canonical class names, module placement, artifact shape, versioning, serialization, and Decision lineage integration.

---

## 3. Forbidden content in Decision-owned artifact

The artifact **must not** carry runtime ownership:

```text
agent_id
agent_instance_id
lease_id
ExecutionId
Nexus node reference
OrchestrationSlotId
scheduler
semaphore
GraphExecutor
```

Decision may express *"I need N independent contributions"* — not *"create N Nexus nodes"*, *"use semaphore=3"*, or *"use agent instance X"*.

---

## 4. Integration contract constraints

### 4.1 Typed only

NPSC-5C/R2 will **not** accept:

```text
free text
rationale parsing
JSON hidden in string
metadata["fanout"]
metadata["agents"]
```

### 4.2 Public and importable

The contract must be:

```text
public
typed
versionable / evolution-safe per Decision conventions
importable without private implementation dependency
```

NPSC must not import private Decision classes.

### 4.3 No CoordinationIntent duplication

Decision System must **not** copy `CoordinationIntent` into its own namespace. Decision exposes its own public typed semantic artifact under Decision ownership. The NPSC consumer adapter (future R2 work) performs:

```text
Decision public artifact → CoordinationIntent
```

### 4.4 Decision does not execute

Decision integration produces typed semantic information only. Decision must never route to:

```text
Nexus
GraphExecutor
specialist agent
lease acquisition
```

Execution remains exclusively on the frozen NPSC path:

```text
CoordinationIntent → CoordinationIntentExecutor → NPSC-5A / NPSC-5B
```

---

## 5. Future handshake contract

```text
Decision System
    ↓ public typed artifact
NPSC Decision projection adapter (NPSC-5C/R2 — this session)
    ↓
CoordinationIntent
    ↓
CoordinationIntentExecutor
    ├── SINGLE  → MultiAgentCoordinationService
    └── FAN_OUT → BoundedMultiAgentFanOutService
```

The adapter is **future ownership of the Agent Distribution session**. The artifact is **ownership of the Decision System session**.

No speculative adapter or placeholder is created at R1 freeze.

---

## 6. R2 entry conditions

NPSC-5C/R2 may be unblocked only when:

1. Decision owner publishes public typed coordination artifact.
2. Artifact is committed to `development`.
3. Decision owner qualifies its own contract.
4. Agent Distribution session re-reads exact public contract.
5. No string/metadata parsing is required.

### R2 remains BLOCKED if

| Condition | Result |
| --------- | ------ |
| Artifact is text-only or rationale-only | BLOCKED |
| Metadata parsing required | BLOCKED |
| Physical agent selection in artifact | BLOCKED |
| Runtime topology in artifact | BLOCKED |
| NPSC must modify Decision-owned code | BLOCKED |

---

## 7. Ownership split

| Concern | Owner |
| ------- | ----- |
| Public typed Decision semantic artifact | Decision System session |
| Artifact invariants, lineage, serialization, qualification | Decision System session |
| Thin projection adapter (artifact → `CoordinationIntent`) | Agent Distribution session (R2) |
| Adapter qualification | Agent Distribution session (R2) |
| Coordination execution | Frozen NPSC-5A / NPSC-5B (unchanged) |

---

## 8. Neutral requirement statement

> **Consumer requires semantic capability:** a public typed Decision artifact expressing coordination execution shape, typed contributions with identity and capability requirements, and optional semantic concurrency preference.
>
> **Producer owner chooses:** canonical Decision representation, naming, module, versioning, and serialization.

This preserves modular ownership and prevents NPSC from prescribing Decision implementation details.
