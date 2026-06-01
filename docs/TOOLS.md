# Intergrax Tool Library

**Last updated:** 2026-05-30

The **Tool Library** (`intergrax/tools/`) is Intergrax’s modular catalog of **LLM-facing, agent-invokable capabilities**. Tools sit between agents and the [Integration Library](INTEGRATIONS.md): they expose semantic operations (JSON schemas, descriptions, risk metadata) while composing integration contracts and platform modules underneath.

**Related docs:**

| Document | Purpose |
|----------|---------|
| Phase **M-RAG** | [INTERGRAX_IMPLEMENTATION_PLAN.md](INTERGRAX_IMPLEMENTATION_PLAN.md) — RAG engine phases M-RAG.1–M-RAG.17 |
| RAG stack canon | [intergrax_runtime_architecture.md](intergrax_runtime_architecture.md) — Tier-0 retrieval architecture |
| [intergrax/tools/USAGE.md](../intergrax/tools/USAGE.md) | **Operational guide** — wire tools in Tier-3 apps and invoke from agents |
| [intergrax_runtime_architecture.md](intergrax_runtime_architecture.md) §7.1.6–§7.1.7, §22 | Architecture canon — Tool Library, unified tool model |
| [INTERGRAX_IMPLEMENTATION_PLAN.md](INTERGRAX_IMPLEMENTATION_PLAN.md) Phase O | Phase status, backlog, delivery workflow |
| [INTEGRATIONS.md](INTEGRATIONS.md) | **99** backend adapters tools compose (not called directly by agents) |
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
    ToolProfile(enabled_bundles=["jira"]),
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
| Legacy `ToolBase` | `intergrax/tools/tools_base.py` | **Deprecated** — use `ToolContract` (Phase O.7 Done) |

---

## Catalog tools

Status legend: **Done** = registered handler in catalog.

### Context & retrieval

| tool_id | Status | Description | Composes |
|---------|--------|-------------|----------|
| `rag.retrieve` | **Done** | Hybrid retrieval + optional rerank via `RetrievalService` / `RagProfile` | `vectorstore_manager`, `embedding_manager`, optional `retrieval_service` |
| `rag.ingest_document` | **Done** | `IngestPipeline`: parse (catalog/handler registry) → chunk (strategy id) → embed → index | Same managers + optional `contextual_enricher` |
| `websearch.query` | **Done** | Run web search and return normalized snippets | `websearch_executor` or `SearchProvider` |
| `websearch.read_url` | **Done** | Fetch a URL and return extracted title + plain text | `websearch` page fetch pipeline |
| `websearch.fetch_batch` | **Done** | Fetch multiple URLs and return combined context | `websearch` page fetch pipeline |
| `rag.list_collections` | **Done** | List vector index collection names | `vectorstore_manager` |

**Catalog providers:** Phase O complete — all first-party tools registered; applications wire via `host/tool_wiring.py`.

**Ready-to-use hosts:** `lab_application`, `legal_application`, `research_application`, `poc_template_application` — see [`intergrax/tools/USAGE.md`](../intergrax/tools/USAGE.md).

**Product env flags:** `LEGAL_ENABLE_RAG` / `LEGAL_ENABLE_RAG_INGEST`, `RESEARCH_ENABLE_RAG` / `RESEARCH_ENABLE_RAG_INGEST` — wire vectorstore + embedding managers in `host/tool_wiring.py`.

### Execution & sandbox

| tool_id | Status | Description | Composes |
|---------|--------|-------------|----------|
| `sandbox.exec` | **Done** | Execute allowlisted operation in runtime sandbox | `sandbox_session` via `ToolWiringContext` |

### Issue tracker (Jira)

| tool_id | Status | Description | Composes |
|---------|--------|-------------|----------|
| `jira.get_issue` | **Done** | Fetch single issue by key | `IssueTracker` (`jira` integration) |
| `jira.add_comment` | **Done** | Add comment to issue | `IssueTracker` |
| `jira.search_tasks` | **Done** | Search issues by project, status, assignee (builds JQL internally) | `IssueTracker` |

### Wiki / knowledge

| tool_id | Status | Description | Composes |
|---------|--------|-------------|----------|
| `confluence.get_page` | **Done** | Fetch wiki page content | `WikiKnowledge` |
| `confluence.search_pages` | **Done** | Search internal documentation | `WikiKnowledge` |
| `confluence.search` | **Done** | Alias of `confluence.search_pages` (shorter tool_id for LLM catalogs) | `WikiKnowledge` |

### Notifications (side-effect tools)

| tool_id | Status | Description | Composes |
|---------|--------|-------------|----------|
| `notify.send` | **Done** | Send outbound notification message | `NotificationChannel` |

### Observability

| tool_id | Status | Description | Composes |
|---------|--------|-------------|----------|
| `metrics.query_instant` | observability | **Done** | Instant metrics query | `ObservabilityBackend` |
| `logs.search` | observability | **Done** | Search log index | `ObservabilityBackend` (`elasticsearch`, `opensearch`) |
| `observability.query_traces` | observability | **Done** | Query LLM/agent traces | `ObservabilityBackend` (`langfuse`, `langsmith`, …) |
| `errors.capture` | observability | **Done** | Report error events | `ObservabilityBackend` (`sentry`) |
| `braintrust.log_eval` | observability | **Done** | Log eval score | `ObservabilityBackend` (`braintrust`, role `eval`) |
| `gitlab.create_issue` | issue_tracker | **Done** | Create GitLab issue | `IssueTracker` (`gitlab`) |
| `pagerduty.trigger_incident` | notification | **Done** | Trigger on-call incident | `NotificationChannel` (`pagerduty`) |

### Composite observability (Phase M.10)

Harness lab uses **one primary** `observability_backend` (Sentry) plus **additional slugs** in `IntegrationProfile.options` (LangSmith). `ToolWiringContext.from_integration_profile()` builds `observability_backends`; each tool picks a backend by role (`errors`, `traces`, `logs`, `eval`) via `resolve_observability_backend()`. See [observability USAGE](../intergrax/tools/providers/observability/USAGE.md).

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
| `injects_context` | When true, Nexus merges output into LLM prompt (§22.1) | **Done** — catalog shim in `catalog_context.py` |
| `timeout_ms` | Runtime-enforced ceiling via `RuntimeToolInvoker` | **Done** |
| `retry_policy` | `ToolRetryPolicy` — runtime-managed retries | **Done** |
| `error_mapping` | Exception type → `RuntimeErrorCode` | **Done** |
| `category` / `tags` | Filtering for large tool sets and MCP grouping | **Done** |

---

## Unified tool model vs legacy flags

| Legacy (deprecated) | Target (canonical) |
|---------------------|--------|
| `ToolInvocationPlan.use_rag` | `tool_ids=["rag.retrieve"]` |
| `ToolInvocationPlan.use_websearch` | `tool_ids=["websearch.query"]` |
| `ToolInvocationPlan.use_tools` | `use_tools=True` (ToolsAgent planner over registry) |
| `LegalToolPlan.use_rag` / `use_websearch` | `tool_ids` + legacy booleans (auto-synced) |

**Rule:** No new platform capability flags — ship as catalog tools. Legacy booleans emit deprecation trace when used without explicit `tool_ids`. See §7.1.7 and Phase O.5 (**Done**).

---

## MCP export

Each application MAY mount catalog tools on MCP (`applications/<app>/mcp/server.py`). Implemented flow (Phase O.6):

```text
ToolRegistry (from wire_*_tools)
    → exporters.to_mcp_tools(contracts)
    → mount_catalog_tools_on_mcp (list_catalog_tools, describe_catalog_tool)
    → FastMCP server (alongside list_agents, run_agent)
```

OpenAI export: `intergrax.tools.exporters.to_openai_tools(registry)` — used by `ToolsAgent`.

---

## Full tool index

Alphabetical reference — all first-party catalog tools (Phase O complete).

| tool_id | Category | Status | Integration / module |
|---------|----------|--------|----------------------|
| `confluence.get_page` | wiki | **Done** | `confluence` | [USAGE](../intergrax/tools/providers/confluence/USAGE.md) |
| `confluence.search_pages` | wiki | **Done** | `confluence` | [USAGE](../intergrax/tools/providers/confluence/USAGE.md) |
| `confluence.search` | wiki | **Done** | `confluence` (alias) | [USAGE](../intergrax/tools/providers/confluence/USAGE.md) |
| `jira.add_comment` | issue_tracker | **Done** | `jira` | [USAGE](../intergrax/tools/providers/jira/USAGE.md) |
| `jira.get_issue` | issue_tracker | **Done** | `jira` | [USAGE](../intergrax/tools/providers/jira/USAGE.md) |
| `jira.search_tasks` | issue_tracker | **Done** | `jira` | [USAGE](../intergrax/tools/providers/jira/USAGE.md) |
| `braintrust.log_eval` | observability | **Done** | `braintrust` | [USAGE](../intergrax/tools/providers/braintrust/USAGE.md) |
| `errors.capture` | observability | **Done** | `sentry` | [USAGE](../intergrax/tools/providers/observability/USAGE.md) |
| `gitlab.create_issue` | issue_tracker | **Done** | `gitlab` | [USAGE](../intergrax/tools/providers/gitlab/USAGE.md) |
| `pagerduty.trigger_incident` | notification | **Done** | `pagerduty` | [USAGE](../intergrax/tools/providers/pagerduty/USAGE.md) |
| `logs.search` | observability | **Done** | `elasticsearch` | [USAGE](../intergrax/tools/providers/observability/USAGE.md) |
| `metrics.query_instant` | observability | **Done** | `prometheus` | [USAGE](../intergrax/tools/providers/observability/USAGE.md) |
| `observability.query_traces` | observability | **Done** | `langfuse` / observability slug | [USAGE](../intergrax/tools/providers/observability/USAGE.md) |
| `notify.send` | notification | **Done** | `notification_channel` slug | [USAGE](../intergrax/tools/providers/notify/USAGE.md) |
| `rag.retrieve` | retrieval | **Done** | `vectorstore_manager`, `embedding_manager` | [USAGE](../intergrax/tools/providers/rag/USAGE.md) |
| `rag.ingest_document` | retrieval | **Done** | `vectorstore_manager`, `embedding_manager` | [USAGE](../intergrax/tools/providers/rag/USAGE.md) |
| `rag.list_collections` | retrieval | **Done** | `vectorstore_manager` | [USAGE](../intergrax/tools/providers/rag/USAGE.md) |
| `sandbox.exec` | sandbox | **Done** | `sandbox_session` | [USAGE](../intergrax/tools/providers/sandbox/USAGE.md) |
| `websearch.query` | retrieval | **Done** | `websearch_executor`, `search_provider` | [USAGE](../intergrax/tools/providers/websearch/USAGE.md) |
| `websearch.read_url` | retrieval | **Done** | page fetch + text extraction | [USAGE](../intergrax/tools/providers/websearch/USAGE.md) |
| `websearch.fetch_batch` | retrieval | **Done** | batch page fetch | [USAGE](../intergrax/tools/providers/websearch/USAGE.md) |

---

## Adding a new tool

1. Add handler under `intergrax/tools/providers/<domain>/` — subclass `ServiceToolHandler` (or `WiringContextToolHandler` for custom logic); put business logic in `service.py`.
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

Catalog conformance (Phase O.4+):

```bash
uv run pytest tests/unit/tools/providers/ -q
```
