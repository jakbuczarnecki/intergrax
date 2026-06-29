# Airflow (airflow)

Category: `workflow_orchestrator`

## Single public entrypoint

- **`AirflowWorkflowOrchestratorIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `AirflowWorkflowOrchestratorIntegration`.
- Contract factory: `create_airflow_workflow_orchestrator_integration()`.
