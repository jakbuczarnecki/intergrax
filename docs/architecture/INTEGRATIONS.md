# Integrations

**Status:** Canonical architecture (domain pair 1:1)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Plan (1:1):** [`plan/INTEGRATIONS.md`](../plan/INTEGRATIONS.md)  
**Target:** [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../guides/IDEAL_HARNESS_AI_ARCHITECTURE.md)  
**Audit layers:** 13–14  
**Audit instruction:** [`audit/INTEGRATIONS.md`](../audit/INTEGRATIONS.md)  
---

## Cursor read scope (token budget)

**Do not read this entire file in one session** (INTEGRATIONS canon).

- **Implement / audit default:** IntegrationLayer contract + wiring + checklists (hub). Provider catalog: [`satellites/INTEGRATIONS_provider_catalog.md`](satellites/INTEGRATIONS_provider_catalog.md).
- **Use** table of contents below — `Read` with offset/limit per §.
- **Plan hub:** [`plan/INTEGRATIONS.md`](../plan/INTEGRATIONS.md) (scoped §6 only).
- **Audit slice:** [`guides/audit_slices/INTEGRATIONS.md`](../guides/audit_slices/INTEGRATIONS.md).
- **Max reads:** at most **one** file >5k tokens per session unless RESUME cites more.

---


## Architecture satellites (read on demand)

Large § blocks moved out of the architecture hub to reduce Cursor context use.
Load **only** the satellite matching your task or cited §.

| Satellite | Contents |
|-----------|----------|
| [`satellites/INTEGRATIONS_provider_catalog.md`](satellites/INTEGRATIONS_provider_catalog.md) | provider catalog |

> **Cursor context budget:** read hub read-scope block + **at most one** satellite per session.
## Allowed integration responsibilities

Integrations **MAY**:

- wrap vendor SDKs or protocols,
- normalize request/response transport details,
- manage backend-specific authentication handoff,
- expose typed low-level operations to tools/platform services,
- translate backend errors into platform error types,
- support health checks,
- support capability discovery where appropriate,
- provide low-level clients for platform-owned services,
- handle retry only when it is backend/protocol-level and does not conflict with runtime retry policy.

Integration retries are **R0 — Backend/protocol** layer — [`RELIABILITY_FAILURE_AND_HITL.md`](RELIABILITY_FAILURE_AND_HITL.md#retry-layers). Must not duplicate R1–R4 retries or hide semantic failures from the Attempt Ledger.

---

## Disallowed integration responsibilities

Integrations **MUST NOT**:

- be invoked directly by agents,
- be invoked directly by Nexus graph nodes as side effects,
- decide which agent should run,
- manage global task lifecycle,
- own orchestration loops,
- own HITL approval,
- own business/product decisions,
- own prompt construction,
- own LLM calls unless the integration itself is explicitly an LLM provider adapter under the LLM adapter layer,
- write agent memory directly,
- emit private trace pipelines outside the observability spine,
- bypass ToolRuntime for agent-invokable side effects,
- implement product-specific workflows.

---

## Integration access paths

Correct access paths for integration use:

### Agent-invokable side effects

```text
Agent -> Tool / Skill -> ToolRuntime -> Integration
```

### Application intake / external surface

```text
External system -> Integration adapter -> Tier-3 intake surface -> UnifiedTaskRunner.run_task()
```

### Platform service backend

```text
Platform service -> Integration adapter
```

**Examples:**

- RAG service may use a vector database integration.
- Memory service may use a database integration.
- Observability sink may use OTEL/Sentry/log integration.
- ToolRuntime may use Slack/Google/GitHub integration through a tool.

---

## Slack / Teams / collaboration adapters

Intergrax supports Slack and Teams as **interaction surfaces** — examples of collaboration adapters, not the definition of the integration layer.

Slack and Teams adapters **may** normalize external messages into tasks and send approved outputs back, but they **must not** own runtime orchestration.

They may provide:

- task intake
- notifications
- approval requests
- progress updates
- final responses
- interactive buttons
- user context
- channel context

**Correct:**

```text
Slack event -> integration adapter -> application intake -> UnifiedTaskRunner.run_task()
    -> Nexus Runtime -> Agent execution -> Nexus final result -> integration adapter sends response
```

**Incorrect:**

```text
Slack bot -> direct agent call -> private memory -> direct tool execution
Slack bot contains orchestration logic
Slack bot directly manages agents
Slack bot stores global task state
```

---

## Cursor review checklist

Before adding or modifying an integration, Cursor must verify:

- [ ] Is this truly an integration, not a tool, skill, agent or application?
- [ ] Is the integration backend/vendor-facing rather than agent-facing?
- [ ] Are side effects exposed to agents only through ToolRuntime?
- [ ] Are secrets handled through approved config/policy mechanisms?
- [ ] Does the integration avoid orchestration, HITL and product workflow ownership?
- [ ] Are backend errors normalized?
- [ ] Is observability routed through the platform spine ([`OBSERVABILITY.md`](OBSERVABILITY.md#observability-event-spine), [event ownership rules](OBSERVABILITY.md#event-ownership-rules))?
- [ ] Is retry limited to protocol/backend concerns and compatible with runtime retry?
- [ ] Is the integration wired through Tier-3 profile/config where required?
- [ ] Are maturity claims expressed through [`guides/MATURITY_TAXONOMY.md`](../guides/MATURITY_TAXONOMY.md)?

---

## Adapter implementation checklist

Before implementing a new adapter, answer:

```text
1. What external system does it connect to?
2. What operations does it expose?
3. What permissions are required?
4. Is it read-only or write-capable?
5. What are risk levels?
6. What errors can happen?
7. What timeout/retry policy is needed?
8. What data should be logged?
9. What data must be protected?
10. Which tools or platform services may use it (not agents directly)?
```

Adapters should be generic and reusable.

---
