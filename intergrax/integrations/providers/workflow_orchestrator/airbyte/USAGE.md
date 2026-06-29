# Airbyte (airbyte)

Category: `workflow_orchestrator`

## Single public entrypoint

- **`AirbyteWorkflowOrchestratorIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `AirbyteWorkflowOrchestratorIntegration`.
- Contract factory: `create_airbyte_workflow_orchestrator_integration()`.
