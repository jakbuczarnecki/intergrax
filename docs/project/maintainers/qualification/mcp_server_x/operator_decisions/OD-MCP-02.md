# OD-MCP-02 — MCP capability scope v1

| Field | Value |
| --- | --- |
| **Decision ID** | OD-MCP-02 |
| **Status** | **APPROVED — Option A** |
| **ADR** | [`ADR_MCP_02_CAPABILITY_SCOPE_TOOLS_ONLY.md`](../../../architecture/ADR_MCP_02_CAPABILITY_SCOPE_TOOLS_ONLY.md) |

## Question

Which MCP protocol capabilities does v1 expose?

## Recommended option (A)

| Capability | v1 |
| --- | --- |
| Tools | MANDATORY |
| Resources | UNSUPPORTED_BY_DESIGN |
| Prompts | UNSUPPORTED_BY_DESIGN |

## Rejected

- Resources/Prompts in v1 — second content/skill authority.

## Operator record

Approved: operator @ 2026-11-11T11:24:00+01:00

**Selected variant:** Option A — Tools MANDATORY; Resources and Prompts UNSUPPORTED_BY_DESIGN for MCP-SERVER-X v1. ADR-MCP-02 remains **Proposed**.
