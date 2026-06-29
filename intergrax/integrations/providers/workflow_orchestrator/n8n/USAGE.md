# N8N (n8n)

Category: `workflow_orchestrator`

## Single public entrypoint

- **`N8nWorkflowOrchestratorIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `N8nWorkflowOrchestratorIntegration`.
- Contract factory: `create_n8n_workflow_orchestrator_integration()`.
