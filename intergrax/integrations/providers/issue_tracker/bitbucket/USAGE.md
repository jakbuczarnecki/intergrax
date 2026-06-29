# Bitbucket (bitbucket)

Category: `issue_tracker`

## Single public entrypoint

- **`BitbucketIssueTrackerIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `BitbucketIssueTrackerIntegration`.
- Contract factory: `create_bitbucket_issue_tracker_integration()`.
