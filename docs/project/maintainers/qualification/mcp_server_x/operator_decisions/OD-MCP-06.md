# OD-MCP-06 — Identity and trust boundary

| Field | Value |
| --- | --- |
| **Decision ID** | OD-MCP-06 |
| **Status** | **PENDING** |
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

_Decision status: **PENDING**. Agent MUST NOT write `Approved`._
