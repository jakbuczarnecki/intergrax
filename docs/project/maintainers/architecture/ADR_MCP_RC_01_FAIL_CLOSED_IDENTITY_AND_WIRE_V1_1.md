# ADR-MCP-RC-01 — Fail-closed MCP identity and wire contract v1.1

| Field | Value |
| --- | --- |
| **Status** | PROPOSED (operator decision `OD-MCP-RC-01` D-4 selects versioned v1.1 with this ADR; not yet reviewed by an independent auditor) |
| **Stage** | MCP-SERVER-X corrective campaign (MCP-RC) |
| **Relates to** | R-P8-01, R-P7-01..06, `OD-MCP-RC-01`, `OD-MCP-RC-02`, `MCP_SERVER_X_RC_CONTRACT_MATRIX.md` |

## Context

At the MCP-SERVER-X P8 closure a host mounted without effective Harness Auth trusted the tenant and user that the client asserted (R-P8-01, F-P8-02). Code review for the corrective campaign found the surface wider than recorded: five application facades never wired harness auth on `/mcp`; three wired it after MCP coupling so the MCP layer never saw it; the research pipeline tool bypassed identity resolution; discovery tools relied on the host middleware only; catalog invoke composed the unfiltered registry. Result bounds, error codes and the call bound were also incomplete (R-P7-03, R-P7-04, R-P7-05).

## Decision

1. **One identity authority, fail-closed by default.** MCP resolves the caller through the existing Harness Auth principal resolution and nothing else (`resolve_mcp_run_agent_identity`, `resolve_mcp_caller_principal`). `McpIdentityMode.HARNESS_BOUND` is the default: no verified principal raises `MCP_PRINCIPAL_REQUIRED`; asserted tenant or user can only confirm the principal and conflict raises `MCP_TENANT_ASSERTION_CONFLICT`.
2. **Local development is an explicit code opt-in.** `LOCAL_DEV_EXPLICIT` is chosen only by a composition parameter, never by an environment variable or a configuration default. It is admitted only for in-process calls and loopback clients, never for catalog invoke, never downgrades configured auth, labels tasks `mcp_identity_assertion=unverified`, and logs a warning when composed.
3. **Request-time auth resolution.** Coupling installs a resolver, so identity wired on the host after MCP coupling (the order used by three factories) is honoured; no one-shot copy.
4. **One bounded-result choke point and one wire-error contract** (`mcp_wire_errors`, `mcp_result_bounds`): every result-producing path is size-checked incrementally; overflow is always `MCP_RESULT_BOUND_EXCEEDED` with `isError=true`; legacy lowercase strings map to declared `MCP_*` codes and stay in the detail.
5. **The MCP call bound is a transport trigger, not a deadline authority** (`OD-MCP-RC-02`): with a protected cancel context `max_call_seconds` requests the existing governed cancel and applies the bounded drain; without one (`LOCAL_DEV_EXPLICIT`) and for catalog invoke it cancels the awaited call. It never reports success, never creates a deadline authority and never changes the host's run budget (`ExecutionDeadlineAuthorityResolver` stays host-owned). The effective wall time is `max_call_seconds` plus the host drain.
6. **Every tool is behind the principal gate by default** (custom host tools included; explicit `public_custom_tool_names` opt-out) and runs under `sanitized_tool_errors`: raw exception text never reaches the wire.
7. **Correlation, not authority:** `mcp_invocation_id` correlates one MCP call with `task_id`, `run_id` and `execution_id`.

## Consequences

- **Breaking for unconfigured hosts** (accepted, versioned as wire contract v1.1; `mcp-server-definition.v1` is unchanged): hosts without Harness Auth now reject `run_agent`, discovery and catalog tools unless they wire auth or pass `identity_mode=LOCAL_DEV_EXPLICIT` in code. All eight facades take an additive `identity_mode` keyword.
- Clients matching legacy message prefixes must match the declared code; the legacy text remains in the detail.
- No new authenticator, deadline, execution or registry authority is introduced; the architecture gate test forbids authenticator or deadline-authority definitions in the engine.

## Rejected alternatives

- Environment-variable selection of the local-development mode (rejected: it is a switch that disables the fail-closed default).
- Feeding the canonical deadline resolver from MCP (rejected: the resolver is composed by the host; MCP cannot own it, see `OD-MCP-RC-02`).
- Suppressing the FastMCP-advertised empty capabilities (rejected by `OD-MCP-RC-01` D-6: documentation only).

## Open limits (accepted scoped, to be confirmed by the independent audit)

- A non-cooperative synchronous catalog tool thread cannot be terminated in-process; MCP guarantees a bounded wait and `MCP_DEADLINE_EXCEEDED` with an unknown outcome, never success.
- The concurrency cap bounds MCP handler occupancy; executions handed to the transport-detachment supervisor are bounded by that supervisor.
- A failing disconnect probe is indistinguishable from a connected client; cancellation through the `CancelledError` path still applies.
- `LOCAL_DEV_EXPLICIT` trusts `request.client.host`; behind a same-host proxy the loopback guard is defeated, so the mode must not be used behind a proxy (a development convenience, not an access control).
- `execute_mcp_agent_task` remains a public low-level entry that accepts caller-asserted tenant and user for hosts that already resolved identity; the `unverified` task label is evidence, not a gate.
- `apply_harness_auth_middleware` still attaches nothing when `require_api_key` is set without a key or provider; the MCP per-call gate fails closed regardless (owner: the harness host, not MCP).
