# Ms365 Graph (ms365_graph)

Category: `collaboration_suite`

## Legacy facade

- `create_ms365_graph_integration()` remains backward-compatible.

## Contract-based integration

- `Ms365GraphCollaborationSuiteIntegration` derives from the category-specific contract.
- Factory: `create_ms365_graph_collaboration_suite_integration()`.
- Disabled by default (`enabled=False`).
- No vendor SDK or network I/O in the contract adapter.
- Injectable `{prefix}Client` required when `enabled=True`.

## Registry

- `register.py` remains legacy-compatible.
- Registry v2 / contract registry wiring deferred.
