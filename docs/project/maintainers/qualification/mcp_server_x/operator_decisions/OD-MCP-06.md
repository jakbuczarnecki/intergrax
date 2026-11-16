# OD-MCP-06 — Identity and trust boundary

| Field | Value |
| --- | --- |
| **Decision ID** | OD-MCP-06 |
| **Status** | **APPROVED — Option A** |
| **ADR** | [`ADR_MCP_06_TRUST_BOUNDARY_HARNESS_PRINCIPAL.md`](../../../architecture/ADR_MCP_06_TRUST_BOUNDARY_HARNESS_PRINCIPAL.md) |

## Question

What identity plane governs MCP tool invocation?

## Recommended option (A)

- Harness HTTP principal SSOT; client tenant/user fields are assertions only.
- Fail-closed when auth profile requires principal.
- Local dev bypass allowed only in dev profile with P5 proof that production profile excludes bypass.

## Rejected

- Separate MCP RBAC or credential store.

## Operator record

Approved: operator @ 2026-11-11T11:24:00+01:00

**Selected variant:** Option A — harness HTTP principal SSOT; MCP tenant/user fields are assertions; fail-closed when profile requires principal. ADR-MCP-06 remains **Proposed**.
