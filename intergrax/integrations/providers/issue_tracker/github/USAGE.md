# Github (github)

Category: `issue_tracker`

## Single public entrypoint

- **`GithubIssueTrackerIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `GithubIssueTrackerIntegration`.
- Contract factory: `create_github_issue_tracker_integration()`.
