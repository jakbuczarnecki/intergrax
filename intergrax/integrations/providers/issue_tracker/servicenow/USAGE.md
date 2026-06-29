# Servicenow (servicenow)

Category: `issue_tracker`

## Single public entrypoint

- **`ServicenowIssueTrackerIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `ServicenowIssueTrackerIntegration`.
- Contract factory: `create_servicenow_issue_tracker_integration()`.
