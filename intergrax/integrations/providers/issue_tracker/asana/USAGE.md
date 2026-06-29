# Asana (asana)

Category: `issue_tracker`

## Single public entrypoint

- **`AsanaIssueTrackerIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `AsanaIssueTrackerIntegration`.
- Contract factory: `create_asana_issue_tracker_integration()`.
