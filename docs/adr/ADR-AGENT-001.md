# ADR-AGENT-001: Agent Cognitive Patterns as Tier-2 Library — Nexus Remains Agent OS

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-06-10 |
| **Deciders** | Platform architecture (Harness AI) |
| **Related** | [`architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md`](../architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md) §21–§30 · [ADR-AGENT-002](ADR-AGENT-002.md) · [`plan/AGENT_CONTRACTS_AND_ASSEMBLY.md`](../plan/AGENT_CONTRACTS_AND_ASSEMBLY.md) Phase **ACP** · [`UNIFIED_EXECUTION_RUNTIME.md`](../architecture/UNIFIED_EXECUTION_RUNTIME.md) §42.5 |

## Context

Tier-2 agent authoring in Intergrax is UAEP-first (`get_steps`, `run_step`, `decide_after_step`) but lacks a **first-class cognitive pattern library** (reflex, ReAct, plan-execute, decomposition, reflection). Authors reimplement these loops inside `run_step` or fall back to the legacy `AgentEngine` pipeline path, which creates:

- inconsistent observability and decision semantics,
- confusion about what belongs in Nexus vs the agent class,
- proposals to collapse Tier-1 Nexus orchestration into a “fat” agent base class.

Production harness requirements (multi-agent graphs, HITL, policy, checkpoints, merge) MUST remain in Tier-1. Agent authors need **fast, pattern-based DX** without bypassing governance.

## Decision

**Adopted:**

1. **Nexus stays the Agent OS** — global task lifecycle, multi-agent graph, policy, HITL, retries, merge. No relocation of Nexus responsibilities into Tier-2 agent classes.
2. **Introduce Agent Cognitive Architecture (ACP)** as a **Tier-2 authoring library** under `intergrax/agents/authoring/patterns/`, built on existing UAEP and `RuntimeExecutionContext`.
3. **Cognitive patterns** (`ReflexAgent`, `ReActAgent`, `PlanExecuteAgent`, `DecompositionAgent`, `ReflectionAgent`) implement domain hooks (`perceive`, `reason`, `act`, `evaluate`) **inside** `run_step` / `decide_after_step` — not a parallel execution engine.
4. **Configuration split preserved:** governance and environment profiles remain Tier-3 `ApplicationEnvironmentProfile`; agents declare contract + cognitive pattern + domain logic; `build_context` consumes injected profile metadata from the host.
5. **Legacy `AgentEngine` path** is deprecated for new agents; Phase **ACP-LEG** tracks migration to UAEP-only.
6. **Author-facing `run()` facade** — see [ADR-AGENT-002](ADR-AGENT-002.md); complements this ADR without moving Nexus into agents.

**Rejected:**

| Option | Reason |
|--------|--------|
| Nexus reduced to “step executor only” | Loses multi-agent orchestration, HITL, graph scheduling — core Harness product |
| Agent base class absorbs Harness (memory, policy, metrics) | Duplicates Tier-0/Tier-1; breaks tier boundaries and audit gates |
| All configuration inside agent Python class | Prevents same agent in lab vs prod profiles; bypasses `PolicyEngine` |
| New universal Tier-0 “agent runtime” parallel to UAEP | Violates §5.2 no-redundancy principle |

## Consequences

### Positive

- Clear mental model: **Harness orchestrates, agents decide domain steps, patterns accelerate authoring**.
- One observability path: all patterns emit UAEP `STEP_*`, `DECISION_EMITTED`, tool events.
- Aligns with IDEAL `Harness → Runtime → Agents → Applications`.
- Enables scaffold `--pattern react|decomposition|…` and reference harness agents per pattern.

### Negative

- Authors must learn **three cognition planes** (Nexus graph / UAEP steps / tool loop) — documented in §21.4.
- Pattern library maintenance cost (5 base classes + tests + CVL/ReAct cross-domain sync).
- Short-term coexistence of UAEP patterns and legacy `AgentEngine` until ACP-LEG completes.

## Compliance

- Tier boundaries preserved: patterns live in `intergrax/agents/` (framework), not `agents/<business>/`.
- `intergrax/` MUST NOT import `agents/` or `applications/`.
- Tool/RAG/memory access ONLY via `RuntimeExecutionContext` gateways (§42.12, §42.41).
- Significant architecture canon updated in `AGENT_CONTRACTS_AND_ASSEMBLY.md` §21–§28 before implementation.

## Implementation notes

- **Canon:** `docs/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` §21–§28
- **Plan:** Phase **ACP** in `docs/plan/AGENT_CONTRACTS_AND_ASSEMBLY.md`
- **Audit:** `docs/guides/audit/AGENT_CONTRACTS_AND_ASSEMBLY.md` (ACP dimensions)
- **Verification (post-implementation):**

```bash
uv run pytest tests/unit/agents/authoring/patterns/ -q
uv run pytest tests/acceptance/agent_os -m agent_os -q
python scripts/check_agents_vendor_imports.py
uv run pytest -m gate -q
```
