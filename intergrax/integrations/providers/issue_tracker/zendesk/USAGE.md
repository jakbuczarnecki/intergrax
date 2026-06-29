# Zendesk (zendesk)

Category: `issue_tracker`

## Single public entrypoint

- **`ZendeskIssueTrackerIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `ZendeskIssueTrackerIntegration`.
- Contract factory: `create_zendesk_issue_tracker_integration()`.
