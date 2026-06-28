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
## Platform integration contract (Tier-1 runtime)

**Code:** `intergrax/runtime/integrations/contracts.py` · **Task:** INTEGRATIONS-1A  
**Observability vendor category:** `intergrax/runtime/integrations/observability.py` · **Task:** INTEGRATIONS-1B

All future integration categories derive from or align with the generic **`PlatformIntegrationContract`**. Vendor adapters (Langfuse, Arize, Phoenix, Elasticsearch, OTLP backends, and future custom backends) are **integrations**, not isolated ad-hoc exporters or one-off SDK wrappers.

### Core types

| Type | Role |
|------|------|
| **`PlatformIntegrationContract`** | Generic typed base contract |
| **`PlatformIntegrationConfig`** | Explicit opt-in config (`enabled=false` by default) |
| **`PlatformIntegrationKind`** | Integration category discriminator |
| **`PlatformIntegrationCapability`** | Declared capability tokens |
| **`PlatformIntegrationSecurityPosture`** | Default-safe exposure rules |
| **`PlatformIntegrationHealth`** | Lightweight health/check snapshot |

### Provider identity vs integration kind

**`provider_id`** identifies the shared vendor or backend (for example `elasticsearch`). **`integration_kind`** identifies the category (for example `observability_vendor`, `vector_store`, `search`). They are **separate**:

- Same **`provider_id`** may appear in many categories.
- Each category gets its **own integration class** and **`integration_id`** (`{provider_id}:{integration_kind}`).
- **Do not** build one multi-category “monster” class that inherits unrelated category contracts.

**Example (Elasticsearch):**

| Integration class (future) | `provider_id` | `integration_kind` |
|---------------------------|---------------|-------------------|
| `ElasticsearchObservabilityIntegration` | `elasticsearch` | `observability_vendor` |
| `ElasticsearchVectorStoreIntegration` | `elasticsearch` | `vector_store` |
| `ElasticsearchSearchIntegration` | `elasticsearch` | `search` |

Category-specific contracts (for example **`ObservabilityVendorIntegrationContract`**, **`VectorStoreIntegrationContract`**) **derive from** **`PlatformIntegrationContract`** — they do not replace it.

### Observability vendor integration category (INTEGRATIONS-1B)

**Code:** `intergrax/runtime/integrations/observability.py`

Observability backends (Langfuse, Arize, Phoenix, Elasticsearch, OTLP-oriented vendors, custom sinks) are **observability vendor integrations** — not ad-hoc exporters or one-off SDK wrappers. Concrete adapters must subclass **`ObservabilityVendorIntegrationContract`**, which **derives from** **`PlatformIntegrationContract`**.

| Type | Role |
|------|------|
| **`ObservabilityVendorIntegrationContract`** | Category base contract; satisfies **`ObservabilityExporter`** via `export()` |
| **`ObservabilityVendorIntegrationConfig`** | Typed opt-in config (`enabled=false` by default) |
| **`ObservabilityVendorSignal`** | Declared signal families (`events`, `logs`, `traces`, `metrics`, `llm_events`) |
| **`ObservabilityVendorPayload`** | Vendor-neutral payload mapped from policy-sanitized envelopes |
| **`ObservabilityVendorMappingResult`** | Envelope → payload mapping result |

**Input boundary:** vendor integrations accept only **`ObservabilityExportEnvelope`** records that have already passed **`ObservabilityExportPolicy`** / **`try_export_observability_envelope`**. They consume **`sanitized_application_attributes`** only — never raw **`application_attributes`**, never **`RuntimeEvent`** raw payloads, and never bypass export policy.

**Why not ad-hoc exporters:** OTLP/JSONL helpers remain transport-oriented exporters until aligned; future Langfuse/Arize/Phoenix/Elasticsearch adapters share identity, capabilities, health, failure isolation, and mapping through this contract — one integration class per category, not scattered SDK calls from runtime hot paths.

**Provider vs category (unchanged rule):** the same **`provider_id`** (for example `elasticsearch`) may appear in multiple categories through **separate** integration classes — never one multi-category class or multiple inheritance across unrelated categories:

| Integration class (future) | `provider_id` | `integration_kind` |
|---------------------------|---------------|-------------------|
| `ElasticsearchObservabilityIntegration` | `elasticsearch` | `observability_vendor` |
| `ElasticsearchVectorStoreIntegration` | `elasticsearch` | `vector_store` |
| `ElasticsearchSearchIntegration` | `elasticsearch` | `search` |

### Default safety and opt-in rules

- Integrations are **explicit opt-in** — disabled unless operator/config enables them.
- **`PlatformIntegrationSecurityPosture`** defaults: no secret exposure, no raw payload exposure, sanitized diagnostics.
- **`public_view()`** on contract/config must remain safe for logs and operator UIs.
- Integrations declare **`expects_failure_isolation=true`** — backend/export failures must not fail product/runtime runs (aligned with observability export policy).
- Tier-0 catalog integrations (`intergrax/integrations/`) remain separate; runtime category contracts compose platform behavior without duplicating the slug catalog.

---

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

## Phase INT-P8 — Dynamic Integration Selection & Agent Workspace Gateways (Planned)

**Status:** Architecture & plan only — **not shipped**  
**Plan (1:1):** [`plan/INTEGRATIONS.md`](../plan/INTEGRATIONS.md) — Phase INT-P8  
**Catalog (planned slugs):** [`satellites/INTEGRATIONS_provider_catalog.md`](satellites/INTEGRATIONS_provider_catalog.md) — §INT-P8 planned categories

**Purpose:** Extend the mature integration catalog (**194 shipped slugs** per `layout.py` · `SLUG_CATEGORY`, Full Harness LC **Done**) with **selection metadata**, **gateway-style connectors**, and **agent workspace backends** — without expanding the vendor catalog for its own sake.

### Why INT-P8 (product value, not catalog padding)

| Mechanism | Need |
|-----------|------|
| Dynamic selection metadata + selection engine | Harness chooses provider by capability, risk, cost, locality, and task intent — not slug alone |
| `tool_protocol_gateway` / `mcp` | Controlled access to MCP ecosystem tools/resources through ToolRuntime |
| `api_connector` / `openapi_http` | Universal REST attachment without one provider per vendor |
| `workspace_store` / `local_workspace` | Policy-scoped agent working directory (distinct from `filesystem` object storage) |
| `code_repository` / `local_git` | Local repo analysis and controlled patches/commits without GitHub/GitLab dependency |
| `code_intelligence` / `sourcegraph` | Enterprise cross-repo code search (GitHub issue tracker ≠ code intelligence) |
| Tier-3 presets (INT-P8.9) | Composable stacks for local workspace, coding agents, enterprise API, MCP gateway |

### Architectural boundaries (unchanged)

INT-P8 **does not** change tier boundaries or access paths:

```text
Agent -> Tool / Skill -> ToolRuntime -> Integration
```

- Integrations remain **backend/vendor-facing**; agents **MUST NOT** invoke integrations directly.
- MCP tools, OpenAPI write methods, workspace writes, git commits/patches — **all** side effects through **ToolRuntime** with policy/approval gates.
- INT-P8 **MUST NOT** add LLM providers, vector DBs, observability vendors, browser automation, or project-management SaaS without a product driver.

### Invariants preserved by INT-P8

| Invariant | INT-P8 enforcement |
|-----------|-------------------|
| No direct agent → integration | MCP/OpenAPI/workspace/git exposed only via catalog tools + ToolRuntime |
| HITL on destructive / external write | `requires_human_approval`, `side_effect_level`, unsafe HTTP methods blocked by default |
| Explainable integration choice | Selection engine returns reason, ranked candidates, trace/diagnostic payload |
| Safe refusal | Engine may refuse when no safe integration matches constraints |
| Audit trail | All side effects logged through existing ToolRuntime / observability spine |
| Catalog honesty | Planned categories/slugs documented as **Planned** — not registered in `layout.py` until implementation PRs |

### Planned new categories (summary)

| Category | First provider (planned) | Role |
|----------|-------------------------|------|
| `tool_protocol_gateway` | `mcp` | MCP server discovery, tool/resource listing, schema fetch, gated invocation |
| `api_connector` | `openapi_http` | OpenAPI-driven REST connector with schema validation and method risk classification |
| `workspace_store` | `local_workspace` | Root-scoped local workspace with path policy and gated writes |
| `code_repository` | `local_git` | Local Git read ops + approval-gated patch/commit (backend); Wave 2 `git.*` tools read-only only |
| `code_intelligence` | `sourcegraph` (optional later: `github_code`) | Read-only enterprise code search |

Selection metadata fields (INT-P8.1): `capabilities`, `operations`, `read_write`, `auth_type`, `required_scopes`, `data_sensitivity`, `latency_class`, `cost_class`, `locality`, `deterministic`, `side_effect_level`, `supported_task_intents`, `suitable_agent_types`, `supports_dry_run`, `supports_rollback`, `requires_human_approval`, `rate_limit_class`, `testability`, `selection_hints`, `risk`.

### Product mapping (INT-P8 consumers)

| Product / agent class | INT-P8 mechanisms |
|----------------------|-------------------|
| Local Knowledge Workspace | `local_workspace_stack` — workspace + git + parser + vector |
| Dispute Simulation Workspace | workspace + document parser + local vector |
| Research agents | `openapi_http`, search/RAG slots (existing), selection metadata |
| Coding agents | `coding_agent_stack` — git + workspace + code intelligence + security scanner |
| Automation agents | `openapi_http`, MCP gateway, enterprise API stack |
| Enterprise assistant | `enterprise_api_stack`, identity, secrets, observability |
| Repo architecture audit agents | `local_git`, `sourcegraph`, semgrep (existing) |
| Document intelligence agents | workspace + document parser (existing) |

### Explicit non-goals (INT-P8)

- No new LLM, vector DB, observability/eval, or browser automation providers
- No new project-management SaaS without product
- No LangChain/LlamaIndex/Zapier/Make.com integrations in first wave
- No Git **push** in first wave
- No direct agent invocation of MCP tools or OpenAPI write methods without ToolRuntime approval

**Implementation:** deferred to Phase INT-P8 tasks in [`plan/INTEGRATIONS.md`](../plan/INTEGRATIONS.md) — architecture update first; no runtime catalog changes until per-task PRs.

---
