# MCP-RC mutation report (head e75748d3407bca71425a8fd34c6dacc59d18ad5b)

| mutant | invariant | status |
| --- | --- | --- |
| M1-tenant-admission-bypass | No verified principal means MCP_PRINCIPAL_REQUIRED in the default mode | killed |
| M2-principal-mismatch-ignored | An asserted tenant/user that differs from the verified principal is a conflict | killed |
| M3-result-bound-missing-on-custom-tools | Custom host tool results are bounded like every other result | killed |
| M4-deadline-propagation-dropped | max_call_seconds reaches the transport cancel context | killed |
| M5-deadline-does-not-request-governed-cancel | The call bound requests the governed cancel (cancellation reaches the canonical coordinator) | killed |
| M6-admission-slot-release-lost | An admission slot is released when the handler ends | killed |
| M7-error-code-mapping-swapped | Legacy identity_conflict maps to MCP_TENANT_ASSERTION_CONFLICT | killed |
| M8-unauthorized-execution-path | Every tool that starts an agent task resolves identity in the caller's mode (no forced local-dev path) | killed |
