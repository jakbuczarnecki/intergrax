# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""End-to-end Tool Library usage for applications and agents."""

## Quick start (Tier-3 application)

```python
from intergrax.applications._shared.tool_wiring import build_application_tool_wiring
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.tools.registry.profile import ToolProfile
from intergrax.tools.registry.bootstrap import register_default_tools

register_default_tools()
tool_wiring = build_application_tool_wiring(
    ToolProfile(enabled=["rag.retrieve", "websearch.query"]),
    vectorstore_manager=vectorstore_manager,
    embedding_manager=embedding_manager,
    websearch_executor=websearch_executor,
)

ctx = ApplicationBuildContext.for_manifest(
    manifest,
    settings=settings,
    tool_profile=tool_wiring.profile,
    tool_wiring_context=tool_wiring.wiring_context,
)
registry = build_application_registry(manifest, ctx, builders=builders)
```

Pass `tool_profile` and `tool_wiring_context` into agent config → `RuntimeConfig` → `RuntimeContext.build()` registers catalog tools on `tool_invoker`.

## Reference hosts

| Application | `host/tool_wiring.py` | Default tools |
|-------------|----------------------|---------------|
| `lab_application` | `wire_lab_tools()` | `rag.retrieve`, `websearch.query`, `sandbox.exec` |
| `legal_application` | `wire_legal_tools()` | env-driven (`LEGAL_ENABLE_RAG`, …) |
| `research_application` | `wire_research_tools()` | `websearch.query` when enabled |
| `poc_template_application` | `wire_poc_template_tools()` | lab-like defaults |

## Agent contract

```python
AgentContract(
    allowed_tools=["rag.retrieve", "websearch.query", "jira.search_tasks"],
)
```

`ToolAccessPolicy` enforces the allow-list before `RuntimeToolInvoker` runs.

## Invoke from agent step (UAEP)

```python
from intergrax.contracts.tool_request import ToolRequest

response = await ctx.invoke_tool(
    ToolRequest(
        tool_name="websearch.query",
        agent_id=ctx.agent_id,
        step_id=step.step_id,
        input={"query": "Intergrax agent runtime", "limit": 5},
    )
)
```

## Nexus pipeline (RAG / websearch)

Plans use canonical `tool_ids` (legacy booleans still map automatically):

```python
LegalToolPlan(tool_ids=["rag.retrieve"], use_tools=False, ...)
# or legacy: use_rag=True → auto-adds rag.retrieve
```

`RagStep` / `WebsearchStep` delegate to catalog handlers when `tool_invoker` is configured.

## MCP export

```python
from intergrax.applications._shared.mcp_catalog_tools import mount_catalog_tools_on_mcp

mount_catalog_tools_on_mcp(mcp, tool_wiring.registry)
```

## Full catalog

See [docs/TOOLS.md](../../docs/TOOLS.md).
