# MCP-SERVER-X RC — contract matrix v1.1 (supplement)

| Field | Value |
| --- | --- |
| **Stage** | MCP-SERVER-X corrective campaign (MCP-RC), additive to the closed P7/P8 records |
| **Status** | WORKING DRAFT — not certified; no audit, no certification, no production claim |
| **Decisions** | `OD-MCP-RC-01` (D-4 versioning, D-5 invocation id, D-6 capabilities), `OD-MCP-RC-02` (deadline trigger) |
| **Relation to P7/P8 records** | `MCP_SERVER_X_CONTRACT_AND_PROTOCOL_MATRIX.md` and all P8 records are unchanged; this file supplements them |
| **Definition contract** | `mcp-server-definition.v1` is unchanged (no schema change). The **wire contract** moves to v1.1 (identity default, error codes, bounds) |

## Error codes (v1.1; 15 members of `McpServerDefinitionErrorCode`)

The first seven codes are the P7 set (see the P7 matrix). The codes below are added in v1.1; the wire shape is `"<MCP_CODE>: <detail>"` on a tool error (`isError=true`) and the legacy string stays in the detail.

| Code class | Meaning | Legacy strings mapped to it |
| --- | --- | --- |
| `MCP_PRINCIPAL_REQUIRED` | No verified harness principal (now **raised**, was declared only) | `unauthorized`, `forbidden`, `missing_caller_identity` |
| `MCP_TENANT_ASSERTION_CONFLICT` | Asserted tenant/user differs from the principal, or caller tenant differs from the binding (now **raised**) | `identity_conflict`, `wrong_tenant` |
| `MCP_RESULT_BOUND_EXCEEDED` | Result over the serialized byte ceiling on **any** result-producing path (now **raised**, `isError=true`) | `mcp_payload_byte_limit_exceeded` |
| `MCP_EXECUTION_BINDING_INVALID` | Catalog invoke without a valid active execution binding | `invalid_task_id`, `invalid_run_id`, `missing_execution_binding`, `stale_run_binding`, `stale_step_binding` |
| `MCP_CALLER_NOT_AUTHORIZED` | Verified caller does not own the execution | `unauthorized_execution_caller` |
| `MCP_CATALOG_TOOL_REFUSED` | Tool unknown or disabled | `unknown_tool`, `tool_disabled` |
| `MCP_TOOL_INPUT_INVALID` | Tool input or contract invalid | `invalid_tool_input`, `invalid_tool_contract` |
| `MCP_EXECUTION_REFUSED` | Execution terminal, cancelled or not admitted | `execution_terminal`, `execution_cancelled`, `execution_admission_denied` |
| `MCP_INTERNAL_ERROR` | Internal failure; raw exception text is never echoed | `internal_error`, any unmapped string |
| `MCP_DEADLINE_EXCEEDED` | MCP call bound (`max_call_seconds`) reached; outcome may be unknown | — |
| `MCP_CONCURRENT_CALL_LIMIT_EXCEEDED` | Admission limit reached | `mcp_concurrent_call_limit_exceeded` |

Result-returning domain errors of the catalog bridge (for example `execution_cancelled`) keep their `error` key and additionally carry `mcp_code`.

## Wire behaviour changes in v1.1

| Area | v1 (P7/P8) | v1.1 |
| --- | --- | --- |
| Identity default | A host without Harness Auth trusted client-asserted tenant/user (R-P8-01) | `HARNESS_BOUND` fail-closed; `LOCAL_DEV_EXPLICIT` is a code-level opt-in, admitted only in process or from loopback clients, tasks labelled `mcp_identity_assertion=unverified` |
| Result bound | Custom tools unbounded; `run_agent` overflow returned as a normal result | One bounded-result choke point; overflow is always `MCP_RESULT_BOUND_EXCEEDED` with `isError=true` |
| Custom tool failure | Raw exception text on the wire | `MCP_INTERNAL_ERROR` with the exception class only |

## Migration note (facades and hosts)

1. A host composed without effective Harness Auth must either wire harness auth (the MCP layer now resolves it at request time, so identity wired after coupling is honoured) or pass `identity_mode=McpIdentityMode.LOCAL_DEV_EXPLICIT` explicitly, in code.
2. All eight application facades take an additive `identity_mode` keyword (default `HARNESS_BOUND`).
3. Clients that matched legacy strings such as `identity_conflict:` at the start of the message must match `MCP_TENANT_ASSERTION_CONFLICT` (the legacy text is still inside the detail).
