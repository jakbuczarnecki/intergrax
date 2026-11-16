# ADR-MCP-05 — Compatibility facades and `mcp-server-definition.v1`

| Field | Value |
| --- | --- |
| **Document ID** | ADR-MCP-05 |
| **Status** | **Proposed** (pending OD-MCP-05) |
| **Related OD** | OD-MCP-05 |

## Context

Eight applications and `build_nexus_mcp_server` encode today’s behavior. COMPAT-X requires versioned public surfaces. Historical PASS does not cover a new MCP definition contract.

## Decision (proposed)

1. Public contract version: **`mcp-server-definition.v1`** with schema fingerprint of exposed tool schemas.
2. Existing `build_nexus_mcp_server` and `applications/*/mcp/server.py` remain **compatibility facades** over the engine until migration guide retires them.
3. **No silent removal** of `list_agents`, `run_agent`, or catalog triad without COMPAT bump and EVOL record.
4. `research_application` `run_research_pipeline` becomes an explicit **binding**, not a one-off decorator path.

## Alternatives

| Option | Outcome |
| --- | --- |
| A. Big-bang delete legacy builders in P2 | **Rejected** — breaks eight apps |
| B. Facade + v1 contract + inventory | **Recommended** |

## Consequences

- MCP-P4-03 fingerprint; MCP-P4-05 COMPAT inventory entry.
- Regressions: `test_reference_apps_mcp`, historical MCP unit tests remain mandatory in P7.
