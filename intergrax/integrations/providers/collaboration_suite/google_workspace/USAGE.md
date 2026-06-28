# Google Workspace (google_workspace)

Category: `collaboration_suite`

## Legacy facade

- `create_google_workspace_collaboration_suite()` remains backward-compatible.

## Contract-based integration

- `GoogleWorkspaceCollaborationSuiteIntegration` derives from the category-specific contract.
- Factory: `create_google_workspace_collaboration_suite_integration()`.
- Disabled by default (`enabled=False`).
- No vendor SDK or network I/O in the contract adapter.
- Injectable `{prefix}Client` required when `enabled=True`.

## Registry

- `register.py` remains legacy-compatible.
- Registry v2 / contract registry wiring deferred.
