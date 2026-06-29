# Azure Devops (azure_devops)

Category: `issue_tracker`

## Single public entrypoint

- **`AzureDevopsIssueTrackerIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `AzureDevopsIssueTrackerIntegration`.
- Contract factory: `create_azure_devops_issue_tracker_integration()`.
