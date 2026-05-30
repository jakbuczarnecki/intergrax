# Research Application (prototype)

Thin execution environment for the research → summarize multi-agent pipeline.

**Build & deploy:** [`BUILD_AND_DEPLOY.md`](BUILD_AND_DEPLOY.md)

```bash
cp applications/research_application/.env.example applications/research_application/.env
uv run uvicorn research_application.host.main:app --host 0.0.0.0 --port 8010
```

POST `/v1/research/run` with JSON body `{ "message": "your research question" }`.

**Docker:**

```bash
applications/research_application/docker/build-docker.sh
# Windows: applications\research_application\docker\build-docker.bat
```

**MCP:** FastMCP at `/mcp` — `list_agents`, `run_agent`, `run_research_pipeline`, `list_catalog_tools`, `describe_catalog_tool` (`RESEARCH_INCLUDE_MCP`).

**Tools:** `websearch.query` enabled by default (`RESEARCH_ENABLE_WEBSEARCH`). Wiring: `host/tool_wiring.py` → `ApplicationBuildContext` → `ResearchAgent` `RuntimeConfig`. See [`intergrax/tools/USAGE.md`](../../intergrax/tools/USAGE.md).

## Tests

```bash
uv run pytest applications/research_application/research_application_tests -q
```
