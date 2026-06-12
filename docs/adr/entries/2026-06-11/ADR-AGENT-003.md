# ADR-AGENT-003: Agent Step Loop (`on_next_step`) and Dual Observability Planes

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-06-10 |
| **Deciders** | Platform architecture (Harness AI) |
| **Related** | [ADR-AGENT-001](entries/2026-06-11/ADR-AGENT-001.md) · [ADR-AGENT-002](entries/2026-06-11/ADR-AGENT-002.md) · [`architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md`](../../architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md) §31–§36 |

## Context

Authors need:

1. A **single session API** — `agent.run(AgentRunRequest)` returning full trace (steps, tools, RAG, LLM, decisions, terminal reason).
2. A **single step API** — `on_next_step(step_ctx)` overridden in subclasses; harness wraps observability, policy, gateways.
3. **Per-step configuration deltas** — change tools, skills, RAG scope, LLM model between steps.
4. **Dual observability** — application orchestration journal (which agents, why, I/O) separate from agent execution journal (internal steps).

UAEP already implements an internal step loop (`run_step`, `decide_after_step`) but exposes it to authors inconsistently. Production requires typed **`StepOutcome`**, **`AgentRunTrace`**, and **`ApplicationRunSummary`**.

## Decision

**Adopted:**

1. **Author-facing step hook:** `async def on_next_step(self, step_ctx: AgentStepContext) -> StepOutcome` — primary domain extension point (alongside optional ACP hooks that may delegate to it).
2. **Harness-only executor:** split into **`AgentRuntime.advance_step`** (iteration glue) and **`HarnessKernel.execute_step`** (deterministic L1 primitive — policy, trace, gateways, state merge); **not overridable** by Tier-2 authors; **MUST NOT plan** agent reasoning (§38).
3. **`agent.run()`** runs the **agent decision loop**: merge environment → `advance_step` until `StepOutcome.is_terminal`.
4. **Implementation maps to UAEP** — kernel wraps existing `UAEPExecutor` step path; no second runtime engine. **`NexusLoop`** remains Task/graph orchestration only — not agent plan brain.
5. **Dual observability planes** — §31: Application (Nexus/graph) + Agent (`AgentRunTrace` on result).
6. **Per-step LLM routing** via `StepLLMRouter` on `AgentStepContext` — author selects model within host `LLMProfile` allowlist; policy enforced in STRICT mode.
7. **Shared state** via `SharedContextView` only — agents do not read raw Nexus OS state (§34).

**Rejected:**

- Replacing Nexus with repeated external `agent.run()` orchestration in application code for multi-agent prod flows.
- Author override of `execute_next_step` or direct emission to external observability sinks.

## Consequences

### Positive

- Clear author model: **`run` = session**, **`on_next_step` = one reasoning/action iteration**.
- Full trace returned to application for debugging and eval.
- Supports super-agent (plan in state) and multi-agent graph with same agent class.

### Negative

- Naming migration from `run_step` / `execute_next_step` / UAEP in docs and scaffold.
- New contract types and plan rows (ACP-STEP-*, ACP-OBS-*, ACP-STEP-2b HarnessKernel).

## Compliance

- ADR-AGENT-001 preserved (Nexus remains Agent OS).
- ADR-AGENT-002 preserved (`run()` facade).
- Observability aligns with [`OBSERVABILITY.md`](../../architecture/OBSERVABILITY.md) §1.2.

## Implementation notes

- Plan: **ACP-STEP-1..3**, **ACP-STEP-2b**, **ACP-OBS-1..2**, **ACP-LLM-1**, **ACP-STATE-1**, **ACP-CON-1..3,6,7**
- Canon: architecture §31–§38
