# n8n (n8n)

Category: `workflow_orchestrator`

## Legacy facade

- `create_n8n_workflow_orchestrator()` remains backward-compatible.

## Contract-based integration

- `N8nWorkflowOrchestratorIntegration` derives from the category-specific contract.
- Factory: `create_n8n_workflow_orchestrator_integration()`.
- Disabled by default (`enabled=False`).
- No vendor SDK or network I/O in the contract adapter.
- Injectable `{prefix}Client` required when `enabled=True`.

## Registry

- `register.py` remains legacy-compatible.
- Registry v2 / contract registry wiring deferred.
