# OD-MCP-04 — Declarative definition and dynamic exposure

| Field | Value |
| --- | --- |
| **Decision ID** | OD-MCP-04 |
| **Status** | **PENDING** |
| **ADR** | [`ADR_MCP_04_DECLARATIVE_DEFINITION_AND_EXPOSURE.md`](../../../architecture/ADR_MCP_04_DECLARATIVE_DEFINITION_AND_EXPOSURE.md) |

## Question

How are server definitions authored and how may exposure change at runtime?

## Recommended option (A)

- Typed definition object + thin programmatic builder (same semantics).
- Exposure changes via CONFIG-X recomposition / remount — **not** ToolRegistry mutation.
- No shadow registry in MCP engine.

## Operator record

_Decision status: **PENDING**. Agent MUST NOT write `Approved`._
