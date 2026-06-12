# ADR-AGENT-002: Author-Facing `Agent.run()` Facade Over UAEP

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-06-10 |
| **Deciders** | Platform architecture (Harness AI) |
| **Related** | [ADR-AGENT-001](ADR-AGENT-001.md) · [`architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md`](../architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md) §29–§30 · [`plan/AGENT_CONTRACTS_AND_ASSEMBLY.md`](../plan/AGENT_CONTRACTS_AND_ASSEMBLY.md) Phase **ACP** |

## Context

Tier-2 authors need a **single, obvious entry point** (`agent.run(request)`) with incremental state, environment-injected configuration, and harness services (trace, policy, tools, memory) handled by the framework.

At the same time, production hosts require **Nexus** (`Task → NexusLoop`) for multi-agent graphs, HITL, checkpoints, and global governance.

Today:

- `Agent.run()` exists but delegates opaquely to `AgentEngine`.
- Authors also see UAEP (`get_steps`, `run_step`), legacy `AgentEngine.run`, and `on_next_step` bridges.
- Applications pass profile slices via Tier-3 wiring, but per-agent binding (memory namespace, tool allowlist, RAG backend) is not documented as a cohesive author model.

ADR-AGENT-001 rejected **removing Nexus** or **absorbing Agent OS into agent classes**. This ADR adds the **author-facing facade** without reversing ADR-AGENT-001.

## Decision

**Adopted — Option III (hybrid):**

1. **`AgentRunRequest` / `AgentRunResult`** are the canonical author I/O contracts (planned types; map from/to `RuntimeRequest` / `AgentExecutionResult` until implemented).
2. **`IntergraxAgent.run(request)`** (and ACP subclasses) is the **primary author API** — tracing, policy, errors, UAEP step boundaries, and ACP loops run inside the base implementation.
3. **UAEP remains the internal execution protocol** — `get_steps` / `run_step` / `decide_after_step` are framework-wired; authors override **hooks** (`perceive`, `reason`, `act`, `evaluate`) or `@step` methods, not the harness loop.
4. **Nexus remains the application entry** for `Task` lifecycle and multi-agent orchestration; graph nodes invoke the **same** agent `run` / UAEP path.
5. **Per-agent resource binding** — memory namespaces, tool/skill allowlists, RAG/knowledge backends — is declared on `AgentContract` + `AgentBinding` and **materialized at run time** from Tier-3 profile + request metadata overrides (§30).
6. **Legacy author paths deprecated:** direct `AgentEngine.run` from Tier-2, `on_next_step` bridge (ACP-LEG).

**Rejected:**

| Option | Reason |
|--------|--------|
| Remove Nexus; apps call only `agent.run()` | Loses graph, HITL, merge; orchestration duplicates per app |
| Expose only UAEP to authors | Poor DX; dual mental model persists |
| All config hardcoded in agent subclass | Breaks lab/prod parity and tenant isolation |

## Consequences

### Positive

- One mental model for authors: **subclass → hooks → `run()`**.
- Same agent code in pytest, lab HTTP, and Nexus graph nodes.
- Environment can inject external parameters per run without editing agent source.
- Per-agent memory/tools/knowledge remain policy-bound through harness gateways.

### Negative

- Two entry postures remain (direct `run` vs `Task`) — documented explicitly in §29.3.
- Implementation work: `AgentRunRequest`/`Result`, profile merge, ACP-DX plan rows.

## Compliance

- ADR-AGENT-001 preserved: Nexus not moved into Tier-2 business packages.
- Tier boundaries: agents use `ctx.invoke_tool`, `memory_view` — no vendor SDKs.
- Documentation updated before ACP-DX code (ACP-DOC.4).

## Implementation notes

- Plan rows: **ACP-DX-1** … **ACP-DX-5** in [`plan/AGENT_CONTRACTS_AND_ASSEMBLY.md`](../plan/AGENT_CONTRACTS_AND_ASSEMBLY.md)
- Canon: architecture §29–§30
- Verification (post-implementation):

```bash
uv run pytest tests/unit/agents/authoring/ -q
uv run pytest tests/acceptance/agent_os -m agent_os -q
```
