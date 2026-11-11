# ADR-MCP-02 — MCP capability scope (Tools mandatory; Resources/Prompts out of v1)

| Field | Value |
| --- | --- |
| **Document ID** | ADR-MCP-02 |
| **Status** | **Proposed** (pending OD-MCP-02) |
| **Related OD** | OD-MCP-02 |

## Context

MCP protocol defines Tools, Resources, and Prompts. The closed-world inventory shows **no** `@mcp.resource` or `@mcp.prompt` under `intergrax/`. Skill-X owns agent behavior influence; adding Prompts would duplicate that authority.

## Decision (proposed)

| MCP capability | v1 classification |
| --- | --- |
| Tools | **MANDATORY** — via `ToolContract` + explicit host tools |
| Resources | **UNSUPPORTED_BY_DESIGN** |
| Prompts | **UNSUPPORTED_BY_DESIGN** |
| sampling / roots extensions | **DEFERRED_WITH_JUSTIFICATION** |

Clients MUST receive honest capability advertisement: **tools only** on the streamable HTTP surface for v1.

## Alternatives

| Option | Outcome |
| --- | --- |
| A. Implement Resources with ad-hoc document store | **Rejected** — second content authority, out of campaign scope |
| B. Expose Prompts for agent instructions | **Rejected** — collides with SKILL-X |
| C. Tools only v1 | **Recommended** |

## Consequences

- L3 protocol matrix (P7) tests only tool methods; Resources/Prompts listed as unsupported.
- No second RAG or prompt registry in MCP engine.
