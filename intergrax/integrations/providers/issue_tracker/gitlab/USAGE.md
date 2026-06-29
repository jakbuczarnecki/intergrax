# Gitlab (gitlab)

Category: `issue_tracker`

## Single public entrypoint

- **`GitlabIssueTrackerIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `GitlabIssueTrackerIntegration`.
- Contract factory: `create_gitlab_issue_tracker_integration()`.
