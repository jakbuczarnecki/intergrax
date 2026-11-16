# OD-MCP-04 — Declarative definition and dynamic exposure

| Field | Value |
| --- | --- |
| **Decision ID** | OD-MCP-04 |
| **Status** | **APPROVED — Option A** |
| **ADR** | [`ADR_MCP_04_DECLARATIVE_DEFINITION_AND_EXPOSURE.md`](../../../architecture/ADR_MCP_04_DECLARATIVE_DEFINITION_AND_EXPOSURE.md) |

## Question

How are server definitions authored and how may exposure change at runtime?

## Recommended option (A)

- Typed definition object + thin programmatic builder (same semantics).
- Exposure changes via CONFIG-X recomposition / remount — **not** ToolRegistry mutation.
- No shadow registry in MCP engine.

## Operator record

Approved: operator @ 2026-11-11T11:24:00+01:00

**Selected variant:** Option A — typed definition + thin programmatic builder; exposure via CONFIG-X recomposition; no shadow registry in engine. ADR-MCP-04 remains **Proposed**.
