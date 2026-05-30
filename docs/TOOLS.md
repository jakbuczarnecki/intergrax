# Intergrax Tool Library

**Last updated:** 2026-05-30

The **Tool Library** (`intergrax/tools/`) is Intergrax’s modular catalog of **LLM-facing, agent-invokable capabilities**. Tools sit between agents and the [Integration Library](INTEGRATIONS.md): they expose semantic operations (JSON schemas, descriptions, risk metadata) while composing integration contracts and platform modules underneath.

**Related docs:**

| Document | Purpose |
|----------|---------|
| [intergrax_runtime_architecture.md](intergrax_runtime_architecture.md) §7.1.6–§7.1.7, §22 | Architecture canon — Tool Library, unified tool model |
| [INTERGRAX_IMPLEMENTATION_PLAN.md](INTERGRAX_IMPLEMENTATION_PLAN.md) Phase O | Phase status, backlog, delivery workflow |
| [INTEGRATIONS.md](INTEGRATIONS.md) | Backend adapters tools compose (not called directly by agents) |
| [AGENT_CREATION_GUIDE.md](AGENT_CREATION_GUIDE.md) Appendix E | How agents declare `allowed_tools` vs applications wire backends |

---

## Design principles

| Principle | What it means |
|-----------|---------------|
| **LLM-first contracts** | Every tool has `tool_id`, `description`, Pydantic `input_schema` / `output_schema` — optimized for model tool selection and MCP export. |
| **Compose integrations** | Handlers call `IssueTracker`, `SearchProvider`, RAG managers, etc. — never vendor SDKs. |
| **Single execution path** | All invocations route through `ToolRuntime` → `RuntimeToolInvoker` (trace, policy, idempotency). |
| **Explicit registration** | Tier-3 passes `ToolProvider` modules + `ToolWiringContext`; no runtime discovery from agent code. |
| **Unified model** | Platform capabilities (RAG, web search, Jira, sandbox) are **tools** — not parallel boolean flags (§7.1.7). |
| **Dual export** | Same `ToolContract` → OpenAI function schema, MCP tool, and `ToolRequest.tool_name`. |

---

## Three-layer stack

```text
Tier-2  Agent (allowed_tools, ToolRequest)
        │
        ▼
Tier-0  Tool Library (rag.retrieve, jira.search_tasks, …)
        │
        ▼
Tier-0  Integration Library (IssueTracker, SearchProvider, VectorStore, …)
```

**Agents declare tool_ids.** **Applications enable tools** via `ToolProfile` and inject integrations via `ToolWiringContext`. **Integrations** remain vendor-swappable without agent changes.

---

## How wiring works (Phase O.2)

```text
Tier-3 application (tool_wiring.py)
        │
        ├── IntegrationProfile.resolve()  ──►  ToolWiringContext.from_integration_profile()
        │
        ▼
ToolProfile(enabled=[...], enabled_bundles=[...])
        │
        ▼
register_default_tools()  ──►  build_registry_from_profile(profile, ctx)
        │
        ▼
ToolRegistry  ──►  RuntimeToolInvoker  ──►  Agent / ToolsAgent / MCP
```

**Example — enable tools from catalog profile:**

```python
from intergrax.tools.registry import (
    ToolProfile,
    ToolWiringContext,
    build_registry_from_profile,
    register_default_tools,
)
from intergrax.integrations import IntegrationProfile, register_default_integrations

register_default_integrations()
register_default_tools()

profile = IntegrationProfile(issue_tracker=IntegrationSlug.JIRA)
ctx = ToolWiringContext.from_integration_profile(profile)

registry = build_registry_from_profile(
    ToolProfile(enabled_bundles=["jira"]),  # Phase O.4
    ctx=ctx,
)
```

---

## Tool engine (implemented today)

These components exist in the repository **before** the full provider catalog ships:

| Component | Path | Status |
|-----------|------|--------|
| `ToolContract` | `intergrax/tools/core/contracts.py` | **Done** — `ToolRiskLevel`, `ToolRetryPolicy`, metadata; invoker enforces timeout/retry |
| `ToolRegistry` | `intergrax/tools/registry/runtime.py` | **Done** |
| `ToolHandler` / `ToolExecutor` | `intergrax/tools/tool_executor.py` | **Done** |
| `ToolExecutionRequest` / `ToolExecutionResult` | `intergrax/tools/execution_models.py` | **Done** |
| `ToolProvider` protocol | `intergrax/tools/core/provider.py` | **Done** — accepts optional `ToolWiringContext` |
| `ToolCatalog` / `ToolProfile` / `ToolWiringContext` | `intergrax/tools/registry/` | **Done** — Phase O.2 |
| `register_default_tools()` / `build_registry_from_profile()` | `intergrax/tools/registry/bootstrap.py`, `factory.py` | **Done** |
| `RuntimeToolInvoker` | `intergrax/runtime/nexus/tools/invoker.py` | **Done** — validation, trace, error mapping |
| `RuntimeToolGateway` | `intergrax/runtime/nexus/tools/tool_gateway.py` | **Done** — UAEP / §42.12 entry |
| `ToolsAgent` (LLM planner) | `intergrax/tools/tools_agent.py` | **Done** — OpenAI schema from registry |
| `ToolAccessPolicy` | `intergrax/runtime/nexus/tools/tool_access_policy.py` | **Done** |
| Legacy `ToolBase` | `intergrax/tools/tools_base.py` | **Deprecated** — migrate to `ToolContract` (Phase O.2) |

---

## Catalog tools

Status legend: **Done** = registered handler in catalog; **Engine only** = invoked outside catalog path; **Planned** = Phase O backlog.

### Context & retrieval

| tool_id | Status | Description | Composes |
|---------|--------|-------------|----------|
| `rag.retrieve` | **Done** | Retrieve documents from vector index for prompt context | `vectorstore_manager` + `embedding_manager` via `ToolWiringContext` |
| `websearch.query` | **Done** | Run web search and return normalized snippets | `websearch_executor` or `SearchProvider` |

> **Transitional:** Today these run via legacy `RagStep` / `WebsearchStep` and plan flags `use_rag` / `use_websearch`. Phase O.5 unifies them as catalog tools (§7.1.7).

### Execution & sandbox

| tool_id | Status | Description | Composes |
|---------|--------|-------------|----------|
| `sandbox.exec` | **Engine only** | Execute allowlisted operation in runtime sandbox | `intergrax/runtime/sandbox/` |

### Issue tracker (Jira)

| tool_id | Status | Description | Composes |
|---------|--------|-------------|----------|
| `jira.get_issue` | Planned (O.4) | Fetch single issue by key | `IssueTracker` (`jira` integration) |
| `jira.add_comment` | Planned (O.4) | Add comment to issue | `IssueTracker` |
| `jira.search_tasks` | Planned (O.4) | Search issues by project, status, assignee (builds JQL internally) | `IssueTracker` |

### Wiki / knowledge

| tool_id | Status | Description | Composes |
|---------|--------|-------------|----------|
| `confluence.get_page` | Planned (O.6) | Fetch wiki page content | `WikiKnowledge` |
| `confluence.search_pages` | Planned (O.6) | Search internal documentation | `WikiKnowledge` |

### Notifications (side-effect tools)

| tool_id | Status | Description | Composes |
|---------|--------|-------------|----------|
| `notify.send` | Planned (O.6) | Send outbound notification message | `NotificationChannel` |

### Observability

| tool_id | Status | Description | Composes |
|---------|--------|-------------|----------|
| `metrics.query_instant` | Planned (O.6) | Instant metrics query | `ObservabilityBackend` (`prometheus`) |
| `logs.search` | Planned (O.6) | Search log index | `ObservabilityBackend` (`elasticsearch`) |

---

## Tool metadata (contract — Phase O.1 Done)

| Field | Purpose | Status |
|-------|---------|--------|
| `tool_id` | Stable registry key and `ToolRequest.tool_name` | **Done** |
| `name` | Human-readable label | **Done** |
| `description` | LLM tool-selection text (required) | **Done** |
| `description_short` | Optional compact variant for large catalogs | **Done** |
| `input_schema` / `output_schema` | Pydantic models → JSON Schema export | **Done** |
| `risk_level` | `ToolRiskLevel`: LOW \| MEDIUM \| HIGH \| CRITICAL | **Done** |
| `side_effects` | Whether invocation mutates external state | **Done** |
| `injects_context` | When true, Nexus merges output into LLM prompt (§22.1) | **Done** (contract; injection wiring Phase O.3+) |
| `timeout_ms` | Runtime-enforced ceiling via `RuntimeToolInvoker` | **Done** |
| `retry_policy` | `ToolRetryPolicy` — runtime-managed retries | **Done** |
| `error_mapping` | Exception type → `RuntimeErrorCode` | **Done** |
| `category` / `tags` | Filtering for large tool sets and MCP grouping | **Done** |

---

## Unified tool model vs legacy flags

| Legacy (deprecated) | Target |
|---------------------|--------|
| `ToolInvocationPlan.use_rag` | Invoke `rag.retrieve` or list in plan `tool_ids` |
| `ToolInvocationPlan.use_websearch` | Invoke `websearch.query` |
| `ToolInvocationPlan.use_tools` | Explicit `tool_ids` + ToolsAgent over registry |
| `LegalToolPlan.use_rag` / `use_websearch` | `tools: ["rag.retrieve", "websearch.query"]` |

**Rule:** No new platform capability flags — ship as catalog tools. See §7.1.7 and Phase O.5.

---

## MCP export

Each application MAY mount catalog tools on MCP (`applications/<app>/mcp/server.py`). Target flow:

```text
ToolCatalog.list_enabled(ToolProfile)
    → exporters.to_mcp_tools(contracts)
    → FastMCP server (alongside list_agents, run_agent)
```

Same `tool_id` values as `AgentContract.allowed_tools` and OpenAI function names.

---

## Full tool index (planned catalog)

Alphabetical reference — all target first-party tools.

| tool_id | Category | Status | Integration / module |
|---------|----------|--------|----------------------|
| `confluence.get_page` | wiki | Planned | `confluence` |
| `confluence.search_pages` | wiki | Planned | `confluence` |
| `jira.add_comment` | issue_tracker | Planned | `jira` |
| `jira.get_issue` | issue_tracker | Planned | `jira` |
| `jira.search_tasks` | issue_tracker | Planned | `jira` |
| `logs.search` | observability | Planned | `elasticsearch` |
| `metrics.query_instant` | observability | Planned | `prometheus` |
| `notify.send` | notification | Planned | `notification_channel` slug |
| `rag.retrieve` | retrieval | **Done** | `vectorstore_manager`, `embedding_manager` | [USAGE](../intergrax/tools/providers/rag/USAGE.md) |
| `sandbox.exec` | sandbox | Engine only | `runtime/sandbox/` |
| `websearch.query` | retrieval | **Done** | `websearch_executor`, `search_provider` | [USAGE](../intergrax/tools/providers/websearch/USAGE.md) |

---

## Adding a new tool

1. Add handler under `intergrax/tools/providers/<domain>/` (contracts, handlers, bundle).
2. Compose existing integration contracts — add integration provider first if missing.
3. Register in `register_default_tools()` (Phase O.2).
4. Add unit tests under `tests/unit/tools/providers/<domain>/`.
5. Add `providers/<domain>/USAGE.md` (English).
6. Update this catalog and Phase O tracker in the implementation plan.
7. Wire in one Tier-3 application via `ToolProfile` + `ToolWiringContext`.

Delivery checklist: [INTERGRAX_IMPLEMENTATION_PLAN.md](INTERGRAX_IMPLEMENTATION_PLAN.md) — Phase O.4 workflow.

---

## Tests

Tool runtime regression (existing):

```bash
uv run pytest tests/unit/runtime/tools/ tests/unit/tools/ -q
```

Catalog conformance (Phase O.6+):

```bash
uv run pytest tests/unit/tools/providers/ -q
```
