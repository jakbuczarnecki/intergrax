# Ms365 Graph (ms365_graph)

Category: `collaboration_suite`

## Single public entrypoint

- **`Ms365GraphCollaborationSuiteIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `Ms365GraphCollaborationSuiteIntegration`.
- Contract factory: `create_ms365_graph_collaboration_suite_integration()`.
