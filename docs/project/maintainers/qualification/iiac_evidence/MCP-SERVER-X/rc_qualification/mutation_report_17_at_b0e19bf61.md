# MCP-RC mutation report (head b0e19bf61b346e6941b2e190b703c6922e708b34)

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
| M9-cross-loop-failure-payload-bound-removed | Cross-loop failure payloads are bounded and sanitized at the MCP boundary | killed |
| M10-cross-loop-success-payload-bound-removed | Cross-loop success payloads are bounded | killed |
| M11-local-dev-downgrades-configured-auth | LOCAL_DEV never downgrades configured auth for in-process callers | killed |
| M12-list-catalog-tools-principal-gate-removed | list_catalog_tools requires a principal in the default mode | killed |
| M13-describe-catalog-tool-principal-gate-removed | describe_catalog_tool requires a principal in the default mode | killed |
| M14-raw-exception-text-echoed-by-tools | No tool path echoes raw exception text | killed |
| M15-custom-tool-principal-gate-removed | Custom host tools are behind the principal gate by default | killed |
| M16-failed-host-after-bound-not-deadline-exceeded | A host that fails after the call bound is reported as MCP_DEADLINE_EXCEEDED | killed |
| M17-host-failure-during-drain-window-reraised | Observing the host inside the bounded drain never re-raises the host's own exception | killed |
