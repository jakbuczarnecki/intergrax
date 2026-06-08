# Tool Runtime and Unified Tool Model

**Status:** Canonical architecture (decomposed from platform canon)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Target reference:** [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../IDEAL_HARNESS_AI_ARCHITECTURE.md)

---

# 22. Tool Runtime

Tools are callable operations exposed to Nexus, agents, and MCP clients. They are the **only** LLM-facing execution surface for platform capabilities ([§7](PLATFORM_FOUNDATION.md).1.6, [§7](PLATFORM_FOUNDATION.md).1.7).

Examples:

- `rag.retrieve` — vector / hybrid retrieval (replaces legacy `use_rag`)
- `websearch.query` — web research (replaces legacy `use_websearch`)
- `jira.search_tasks` — issue search with LLM-friendly parameters
- `sandbox.exec` — isolated script execution
- send notification, read/write artifact, query database, browser action

Tools must have:

- `tool_id` (stable registry key)
- `name` and `description` (optimized for LLM tool selection)
- `input_schema` and `output_schema` (Pydantic → JSON Schema)
- `risk_level` and `side_effects`
- `timeout_ms` and `retry_policy` (runtime-enforced)
- optional `injects_context` — when true, Nexus merges output into LLM prompt context (retrieval tools)

Tools MUST be registered in **`ToolRegistry`** via the Tool Library catalog (`intergrax/tools/providers/`, [§7](PLATFORM_FOUNDATION.md).1.6).

All agent and Nexus tool invocation MUST route through **`ToolRuntime`** with policy enforcement ([§42](UNIFIED_EXECUTION_RUNTIME.md).12, [§42](UNIFIED_EXECUTION_RUNTIME.md).36). Direct integration adapter calls from agents are forbidden ([§42](UNIFIED_EXECUTION_RUNTIME.md).41).

**Canonical modules:**

| Concern | Location |
|---------|----------|
| Tool contract + handler protocol | `intergrax/tools/core/`, `intergrax/tools/tool_executor.py` |
| Catalog + providers | `intergrax/tools/providers/` (Phase O) |
| Registry | `intergrax/tools/registry/` |
| LLM planner | `intergrax/tools/tools_agent.py` |
| Runtime enforcement | `intergrax/runtime/nexus/tools/` (`RuntimeToolInvoker`, `RuntimeToolGateway`) |

**Catalog index:** [`TOOLS.md`](TOOLS.md) — first-party catalog (11 `tool_id`s: retrieval, Jira, Confluence, notify, observability, sandbox) registered via `register_default_tools()` (Phase O.4, 2026-05-30).

## 22.1 Context-Injection Tools

Some tools exist primarily to ** enrich the LLM prompt** rather than to perform irreversible side effects (e.g. `rag.retrieve`, `websearch.query`).

When `ToolContract.injects_context = true`:

1. Runtime invokes the tool through the same `RuntimeToolInvoker` path.
2. Nexus merges a bounded preview of the tool output into `state.tools_context_parts` / message assembly (replacing implicit `RagStep` / `WebsearchStep` injection).
3. Trace records both `tool_invocation_*` and context-injection diagnostics.

Side-effect tools (`injects_context = false`) return results to the agent loop only — they do not auto-inject into the main LLM prompt unless the agent explicitly uses the output.

## 22.2 Legacy Pipeline Flags (Deprecated)

Phase O.5 migration is **Done**. Nexus MAY still accept `ToolInvocationPlan(use_rag=…, use_websearch=…, use_tools=…)` as **deprecated aliases** that map to catalog tool_ids:

```text
use_rag=True        → rag.retrieve
use_websearch=True  → websearch.query
use_tools=True      → ToolsAgent planner over ToolRegistry
```

New code MUST use explicit tool_ids and `ToolRequest` — not boolean plan flags.

---

