# Azure Devops (azure_devops)

Category: `issue_tracker`

## Legacy facade

- `create_azure_devops_issue_tracker()` remains backward-compatible.

## Contract-based integration

- `AzureDevopsIssueTrackerIntegration` derives from the category-specific contract.
- Factory: `create_azure_devops_issue_tracker_integration()`.
- Disabled by default (`enabled=False`).
- No vendor SDK or network I/O in the contract adapter.
- Injectable `{prefix}Client` required when `enabled=True`.

## Registry

- `register.py` remains legacy-compatible.
- Registry v2 / contract registry wiring deferred.
