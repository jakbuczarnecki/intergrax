# Jira (jira)

Category: `issue_tracker`

## Legacy facade

- `create_jira_integration()` remains backward-compatible.

## Contract-based integration

- `JiraIssueTrackerIntegration` derives from the category-specific contract.
- Factory: `create_jira_issue_tracker_integration()`.
- Disabled by default (`enabled=False`).
- No vendor SDK or network I/O in the contract adapter.
- Injectable `{prefix}Client` required when `enabled=True`.

## Registry

- `register.py` remains legacy-compatible.
- Registry v2 / contract registry wiring deferred.
