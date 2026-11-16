# ADR-MCP-06 — Trust boundary: harness principal and tenant assertions

| Field | Value |
| --- | --- |
| **Document ID** | ADR-MCP-06 |
| **Status** | **Proposed** (pending OD-MCP-06) |
| **Related OD** | OD-MCP-06 |

## Context

`resolve_mcp_run_agent_identity` binds MCP calls to harness HTTP principal. Client-supplied `tenant_id` / user fields are **assertions** rejected on conflict with verified identity. Local dev bypass may default tenant — production profile must be fail-closed when auth is required.

## Decision (proposed)

1. **Same identity plane** as HTTP host harness — no separate MCP permission schema.
2. Conflicting client assertion vs principal → error **before** Execution admission.
3. Cross-tenant negative tests are P5/P7 gates (FRZ-TEN/SEC incremental evidence only — no PASS promotion in engine PRs).
4. MCP engine does not mint principals or store API keys.

## Alternatives

| Option | Outcome |
| --- | --- |
| A. MCP-specific RBAC table | **Rejected** — splits TENANT-X / harness model |
| B. Harness principal only | **Recommended** |

## Consequences

- Reuse `mcp_harness_principal.py`; extend tests only if engine changes admission order.
- REQ-MCP-SEC-01 gap addressed in P5, not P1.
