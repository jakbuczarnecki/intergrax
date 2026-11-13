# MCP-RC mutation report (head f3078b293ecee402583b3d6ad9ef5aaeff0b2d1d)

| mutant | invariant | status |
| --- | --- | --- |
| A1-local-dev-loopback-guard-removed | LOCAL_DEV refuses non-loopback clients | killed |
| A2-catalog-invoke-unfiltered-registry | catalog invoke composes the allowlist-filtered registry | killed |
| A3-cross-loop-payload-bound-removed | cross-loop catalog payload is bounded at the MCP boundary | survived |
| A3b-cross-loop-success-payload-bound-removed | cross-loop success payload is bounded | survived |
| A4-cancelled-outcome-returned-as-success | a non-completed outcome after the call bound is never success | killed |
| A5-list-agents-principal-gate-removed | list_agents requires a principal in the default mode | killed |
| A6-request-time-auth-resolver-disabled | identity wired after coupling still reaches MCP | killed |
| A7-custom-tool-exception-text-echoed | custom tool exception text is never echoed | killed |
| A8-unknown-tool-echo-uncapped | unknown-tool echo is capped | killed |
| A9-invocation-id-constant | mcp_invocation_id is unique per call | killed |
| A10-unverified-label-dropped | assertion-only identity is labelled unverified | killed |
| A11-local-dev-allowed-with-configured-auth | LOCAL_DEV never downgrades configured auth | survived |
