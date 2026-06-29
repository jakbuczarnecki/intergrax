# Linear (linear)

Category: `issue_tracker`

## Single public entrypoint

- **`LinearIssueTrackerIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `LinearIssueTrackerIntegration`.
- Contract factory: `create_linear_issue_tracker_integration()`.
