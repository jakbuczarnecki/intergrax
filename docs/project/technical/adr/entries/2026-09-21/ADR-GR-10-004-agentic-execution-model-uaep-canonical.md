# ADR-GR-10-004 — Canonical AGENTIC execution model (UAEP vs ACP session)

| Field | Value |
| ----- | ----- |
| **Status** | Accepted |
| **Date** | 2026-09-21 |
| **Related** | [ADR-AGENT-002](../2026-06-11/ADR-AGENT-002.md) · [GOVERNED_EXECUTION.md](../../../../architecture/GOVERNED_EXECUTION.md) · GR-10-A1 / A1-R1 |

## Context

Production AGENTIC work enters through `HostTaskExecution` → `TaskBoundAgenticDelegate` → `AgentEngine`. Two runtime branches exist today:

- **P-UAEP:** `AgentEngine` → `UAEPExecutor` (canonical governance spine: inner guard, `RuntimeToolInvoker`, interrupt handler, StepKernel policy carriers).
- **P-ACP-SESSION:** `AgentEngine` → `IntergraxAgent.run` → `run_acp_session` when `acp.session.v1` (`AcpMetadataKey.SESSION_ENABLED`) is true.

`make_acp_checkpoint_task_enricher` sets `SESSION_ENABLED = True` whenever an `AgentCheckpointStore` is wired through `build_reliability_task_enricher`. Checkpoint persistence therefore **implicitly selects** the ACP session execution branch for harness hosts (legal, research, lab, etc.), while governed contractor and similar hosts without `agent_checkpoint_store` remain on UAEP.

Enterprise composition requires **orthogonal capabilities**: checkpointing is persistence; execution architecture must be explicit, contract-driven, and host-controlled.

## Problem

Coupling checkpoint store presence to `SESSION_ENABLED` conflates:

1. **Persistence / resume** (checkpoint store, compensation, idempotency metadata), with  
2. **Agent execution architecture** (UAEP vs ACP session loop).

`AgentEngine._execute_agent_impl` returns `governance=None` on the ACP branch, bypassing the UAEP governance spine that GR-10 qualified on P-UAEP. Treating both paths as equally canonical would require a second full governance proof matrix (GR-13-scale) for `run_acp_session`.

## Decision

**`UAEP_CANONICAL_ACP_EXPLICIT` (Model A).**

| Concern | Owner layer |
| ------- | ----------- |
| Canonical AGENTIC governance spine | Tier-1 `UAEPExecutor` + production tool invoker composition |
| Agent execution strategy selection | Host composition + explicit task/run metadata contract (not checkpoint enricher side effect) |
| Checkpoint persistence | Tier-2 persistence ports + task metadata wiring (`AgentCheckpointStore`, hooks) |
| ACP session loop | Tier-2 authoring runtime (`run_acp_session`, `AgentRuntime`) — **explicit opt-in** only |
| Session state / step loop | ACP session host inside authoring layer |
| Governance policy evaluation (canonical AGENTIC) | UAEP StepKernel + `PolicyEngine` on UAEP path |
| Tool authorization (canonical AGENTIC) | `RuntimeToolInvoker` composition on UAEP path |
| Continuation (canonical AGENTIC) | `ExecutionContinuationPort` on UAEP-qualified hosts |

**Checkpoint ↔ session coupling:** `make_acp_checkpoint_task_enricher` setting `SESSION_ENABLED` is **`DEPRECATED_MIGRATION_TARGET`**. It is **not** the long-term contract for execution mode. **GR-10-A3** implements runtime decoupling.

**ACP session role:** Authoring-oriented **session execution runtime** for `IntergraxAgent` (step loop, checkpoint hooks inside `run_acp_session`, declarative tool binding). It is **not** a second canonical AGENTIC governance spine for production qualification.

**Rejected alternative — `DUAL_CANONICAL_AGENTIC_EXECUTION` (Model B):** Would require formal execution-strategy selection and full GEP parity for ACP. Cost exceeds benefit given UAEP already owns qualified governance; dual spines increase policy drift and GR-13 surface without unique production semantics that UAEP cannot absorb with checkpoint ports on the UAEP path.

## Alternatives considered

1. **Model A — UAEP canonical, ACP explicit opt-in** — **Chosen.**
2. **Model B — Dual canonical paths** — Rejected (governance duplication, implicit enricher coupling, GR-10 closure blocked).
3. **Remove ACP session entirely** — Rejected; authoring and harness session runs still need `run_acp_session` behind explicit metadata.

## Consequences

### Positive

- Single canonical AGENTIC governance spine for GR-10 / GR-13.
- Checkpointing becomes a persistence capability independent of execution mode.
- Clear migration boundary (GR-10-A3) without ORCHESTRATION or UAEP rewrite in this task.

### Negative

- Harness hosts still hit ACP branch until A3 ships (documented migration).
- ACP checkpoint/resume logic must be re-homed or bridged onto UAEP during A3.

## Migration impact

**GR-10-A3 may change:** `acp_checkpoint_task_enricher`, `build_reliability_task_enricher` composition docs, host factories, `AgentEngine` selection inputs — **not** ORCHESTRATION topology (R9–R15).

**Backward compatibility:** Until A3, runtime behavior unchanged; hosts with checkpoint store still set `SESSION_ENABLED`. Qualification SSOT marks P-ACP as `ARCHITECTURAL_MIGRATION_REQUIRED`.

## Governance impact

Model B parity cost (not undertaken):

| GEP | Existing ACP owner | Parity with UAEP | Missing work if dual-canonical |
| --- | ------------------ | ---------------- | ------------------------------ |
| ROOT_EXECUTION_ADMISSION | Shared root launcher | Same | N/A |
| AGENT_DECISION / INNER | `run_acp_session` / StepKernel | Partial | Recertify inner guard on ACP tool path |
| PRE_MODEL / PRE_OUTPUT / POST_RUN | PolicyEngine in ACP loop | Partial | UAEP-equivalent GEP proofs |
| TOOL_PLAN / TOOL_INVOCATION | Declarative invoker in ACP metadata | Partial | Production `RuntimeToolInvoker` parity |
| MSE / HITL / Continuation | ACP session internals | Gap | Full MP-4R7 / GR-5 spine on ACP branch |
| Reliability | `AgentSessionReliability` in ACP | Partial | Unified with GR-7 host wiring on UAEP |
| Governance Evidence | ACP trace hooks | Gap | Per-GEP GR-8 facts on ACP branch |

## Layer ownership

- **Tier-1 runtime:** `AgentEngine` branch selection reads explicit metadata only (post-A3).
- **Tier-3 hosts:** Choose checkpoint store vs execution mode independently in composition.
- **Tier-2 agents:** `run_acp_session` remains authoring; not production canonical spine.

## Future abstraction (design only — not implemented in GR-10-A2)

If needed after A3: **`AgentExecutionStrategyPort`** (Tier-1 Nexus boundary) with inputs `(agent, RuntimeRequest, composition context)` → `AgentExecutionResult`, selection via host-wired strategy registry, lifecycle bound to single `Execution` attempt. **Owner:** runtime execution + host composition. **Not created in this ADR implementation slice.**

## Rollout boundaries

- **In scope A2:** ADR, qualification SSOT, architecture gates.
- **Out of scope A2:** Runtime migration, ORCHESTRATION GEP changes, GR-13 matrix implementation.

## Compliance

- Tier boundaries preserved (no agents → applications import violations).
- No hidden metadata as permanent execution-mode contract.
- Linked SSOT: `tests/qualification/governance/strategy/catalog.py` (`GR10_A2_*`).
