# Execution Identity Authority Model (EE-A2)

**Classification:** `MAINTAINER_CERTIFICATION`  
**Status:** `CERTIFIED` (audit EE-A2-H2 global freeze on `development`)  
**Audience:** Maintainers, enterprise qualification, architecture gates  

**Parent:** [`EXECUTION_ENGINE_OWNERSHIP_MODEL.md`](EXECUTION_ENGINE_OWNERSHIP_MODEL.md) (EE-A1)  
**Semantic types:** [`intergrax/contracts/execution_identity.py`](../../../intergrax/contracts/execution_identity.py)  
**Authority port:** [`intergrax/contracts/execution_identity_authority.py`](../../../intergrax/contracts/execution_identity_authority.py)  
**Runtime mint:** [`intergrax/runtime/execution/identity_authority.py`](../../../intergrax/runtime/execution/identity_authority.py)

---

## Global Freeze Statement

ExecutionIdentityAuthority is the only production authority
allowed to mint execution lifecycle identities.

No runtime component, adapter, provider,
scheduler or recovery mechanism may create
RunId, ExecutionId or AttemptId directly.

Re-verify this freeze against the current GitHub `development` branch before downstream certification; repository source is the sole authority of truth.

---

## Identity ownership

| Identity | Canonical owner | Notes |
| --- | --- | --- |
| **RunId** | `identity_authority.mint_root_execution_identity` / `mint_background_transport_identity`, admitted by **ExecutionRuntime** | Immutable for the run; survives retry |
| **ExecutionId** | Same authority (`mint_root_execution_identity`, `mint_child_execution_id`) | One root per admission; child segments get new ExecutionId under parent |
| **AttemptId** | Initial: root mint with Run; **retry:** `AttemptLifecycleService.transition_to_next_attempt` via `mint_retry_attempt_id` | Only durable lifecycle service creates retry Attempt |
| **TaskId** | Task plane (`Task` construction, background transport) | Work correlation; not a substitute for execution lifecycle |
| **EventId** | Evidence / observability contracts | Must not schedule execution |
| **SessionId / CorrelationId / RequestId** | Ingress and tracing | May **validate** into RunId at admission; must not bypass runtime mint for Attempt/Execution |

**Lifecycle owner:** `ExecutionRuntime` opens and closes execution.  
**Propagation owner:** `ExecutionBoundary` (`bind_active_execution_identity`).  
**Attempt transitions:** `AttemptLifecycleService` only.

---

## Lifecycle diagram

```text
Intent (Task / AgentRunRequest)
        |
        v
Admission (capacity, host port, background identity)
        |
        v
ExecutionRuntime  ---------------------------+
        |                                    |
        v                                    |
IdentityAuthority (identity_authority.py)    |
        |                                    |
        +-- mint_root_execution_identity     |
        +-- mint_child_execution_id          |
        +-- mint_retry_attempt_id (via       |
        |      AttemptLifecycleService)      |
        v                                    |
ExecutionBoundary.bind_active_*              |
        |                                    |
        v                                    |
Strategy / Nexus / Tools / Evidence          |
        |                                    |
        v                                    |
Terminal / checkpoint / recovery re-entry ---+
```

---

## Mint rules

1. **RunId, AttemptId (initial), ExecutionId (root)** for a production execution are minted only through `identity_authority` functions invoked from `ExecutionRuntime`, `ChildExecutionRunner`, or agent ingress that delegates to `mint_root_execution_identity` (ACP session).
2. **Retry AttemptId** is minted only inside `AttemptLifecycleService` using `mint_retry_attempt_id`.
3. **Child ExecutionId** is minted only via `mint_child_execution_id` during child admission.
4. Low-level `mint_*` in `execution_identity.py` are **primitives**; production code must not call them outside the authority module and exempt conformance harnesses.
5. **Persistence** stores identity fields; it never generates Run/Attempt/Execution identifiers.

---

## Mutation rules

Execution identity values are **immutable** once admitted:

| Operation | RunId | ExecutionId | AttemptId |
| --- | --- | --- | --- |
| Retry | preserved | preserved (re-bind may clear in-process execution until re-admission) | **new** via lifecycle |
| Resume from checkpoint | preserved | preserved | preserved (no lifecycle bump on resume alone) |
| Recover / fan-out partial | preserved | preserved per segment rules | lifecycle-owned transitions only |

Forbidden:

- `resume()` assigning a new **ExecutionId**
- `retry()` assigning a new **RunId**
- Checkpoint restore regenerating Run/Attempt/Execution

---

## Retry semantics

```text
RunId
 |
 ExecutionId
 |
 AttemptId-1  --(failure, eligible)-->  AttemptLifecycleService  -->  AttemptId-2
```

`ExecutionAttemptRetryService` decides eligibility; **only** `AttemptLifecycleService` mints the next Attempt.

---

## Resume semantics

Checkpoint resume **rehydrates** `run_id`, `attempt_id`, and root execution identity from durable checkpoint payload via `resolve_root_task_identity`. Resume does **not** call attempt lifecycle transition and does **not** mint new Run or Execution identifiers.

---

## Recovery semantics

Recovery plane (`LongRunningCoordinator`, `FanOutPartialRecoveryService`) re-enters through canonical execution admission with **existing** identity bindings. Partial recovery does not create parallel lifecycle owners.

---

## Provider rules

LLM and integration **providers** receive `ExecutionContext` / admission intents. They must not mint `RunId`, `AttemptId`, or `ExecutionId`. External-operation correlation may use `TaskId` or intent ids only.

---

## Forbidden patterns

```text
HTTP / MCP intake  -->  mint_run_id()           # blocked (NPSC-4.1)
Nexus              -->  mint_execution_id()     # blocked
Governance         -->  mint_attempt_id()       # blocked
CheckpointStore    -->  generate RunId          # blocked
Provider adapter   -->  create execution context with new ExecutionId  # blocked
Agent local retry  -->  mint AttemptId          # blocked (use AttemptLifecycleService)
```

---

## Architecture gates

| Gate | Scope |
| --- | --- |
| `test_execution_identity_single_authority_gate.py` | Runtime / Nexus / background mint |
| `test_npsc4_1_execution_boundary_hardening_gate.py` | Application intake |
| `test_npsc5e_r1_execution_retry_attempt_semantics.py` | Retry / attempt |
| `test_ee_a2_identity_authority_certification.py` | EE-A2 enterprise certification |
| `test_ee_a2_h2_identity_authority_global_freeze.py` | EE-A2-H2 global identity authority freeze |

---

## Verification statement

Re-verify this model against the current GitHub `development` branch before downstream certification; repository source is the sole authority of truth.
