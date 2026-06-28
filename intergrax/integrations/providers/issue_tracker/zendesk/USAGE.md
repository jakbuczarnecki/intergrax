# Zendesk (zendesk)

Category: `issue_tracker`

## Legacy facade

- `create_zendesk_issue_tracker()` remains backward-compatible.

## Contract-based integration

- `ZendeskIssueTrackerIntegration` derives from the category-specific contract.
- Factory: `create_zendesk_issue_tracker_integration()`.
- Disabled by default (`enabled=False`).
- No vendor SDK or network I/O in the contract adapter.
- Injectable `{prefix}Client` required when `enabled=True`.

## Registry

- `register.py` remains legacy-compatible.
- Registry v2 / contract registry wiring deferred.
