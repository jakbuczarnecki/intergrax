# Google Workspace (google_workspace)

Category: `collaboration_suite`

## Single public entrypoint

- **`GoogleWorkspaceCollaborationSuiteIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `GoogleWorkspaceCollaborationSuiteIntegration`.
- Contract factory: `create_google_workspace_collaboration_suite_integration()`.
