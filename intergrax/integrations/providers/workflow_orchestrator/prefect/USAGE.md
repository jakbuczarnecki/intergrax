# Prefect (prefect)

Category: `workflow_orchestrator`

## Single public entrypoint

- **`PrefectWorkflowOrchestratorIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `PrefectWorkflowOrchestratorIntegration`.
- Contract factory: `create_prefect_workflow_orchestrator_integration()`.
