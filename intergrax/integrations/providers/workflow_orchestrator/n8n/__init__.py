# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "N8N_WORKFLOW_ORCHESTRATOR_PROVIDER_ID",
    "N8nWorkflowOrchestratorIntegration",
    "N8nWorkflowOrchestratorIntegrationConfig",
    "N8nWorkflowOrchestratorClient",
    "create_n8n_workflow_orchestrator",
    "create_n8n_workflow_orchestrator_integration",
    "register_n8n_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_n8n_workflow_orchestrator",
        "create_n8n_workflow_orchestrator_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "N8N_WORKFLOW_ORCHESTRATOR_PROVIDER_ID",
        "N8nWorkflowOrchestratorIntegration",
        "N8nWorkflowOrchestratorIntegrationConfig",
        "N8nWorkflowOrchestratorClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "N8N_WORKFLOW_ORCHESTRATOR_PROVIDER_ID",
        "N8nWorkflowOrchestratorIntegration",
        "N8nWorkflowOrchestratorIntegrationConfig",
        "N8nWorkflowOrchestratorClient",
    }
)

def __getattr__(name: str):
    if name == "register_n8n_integration":
        from intergrax.integrations.providers.workflow_orchestrator.n8n.register import register_n8n_integration

        return register_n8n_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.workflow_orchestrator.n8n import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.workflow_orchestrator.n8n import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.workflow_orchestrator.n8n import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
