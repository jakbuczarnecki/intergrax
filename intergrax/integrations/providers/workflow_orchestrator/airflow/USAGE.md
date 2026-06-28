# Airflow (airflow)

Category: `workflow_orchestrator`

## Legacy facade

- `create_airflow_workflow_orchestrator()` remains backward-compatible.

## Contract-based integration

- `AirflowWorkflowOrchestratorIntegration` derives from the category-specific contract.
- Factory: `create_airflow_workflow_orchestrator_integration()`.
- Disabled by default (`enabled=False`).
- No vendor SDK or network I/O in the contract adapter.
- Injectable `{prefix}Client` required when `enabled=True`.

## Registry

- `register.py` remains legacy-compatible.
- Registry v2 / contract registry wiring deferred.
