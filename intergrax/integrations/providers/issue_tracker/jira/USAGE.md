# Jira (jira)

Category: `issue_tracker`

## Single public entrypoint

- **`JiraIssueTrackerIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `JiraIssueTrackerIntegration`.
- Contract factory: `create_jira_issue_tracker_integration()`.
