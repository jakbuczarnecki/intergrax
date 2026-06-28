# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "PREFECT_WORKFLOW_ORCHESTRATOR_PROVIDER_ID",
    "PrefectWorkflowOrchestratorIntegration",
    "PrefectWorkflowOrchestratorIntegrationConfig",
    "PrefectWorkflowOrchestratorClient",
    "create_prefect_workflow_orchestrator",
    "create_prefect_workflow_orchestrator_integration",
    "register_prefect_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_prefect_workflow_orchestrator",
        "create_prefect_workflow_orchestrator_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "PREFECT_WORKFLOW_ORCHESTRATOR_PROVIDER_ID",
        "PrefectWorkflowOrchestratorIntegration",
        "PrefectWorkflowOrchestratorIntegrationConfig",
        "PrefectWorkflowOrchestratorClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "PREFECT_WORKFLOW_ORCHESTRATOR_PROVIDER_ID",
        "PrefectWorkflowOrchestratorIntegration",
        "PrefectWorkflowOrchestratorIntegrationConfig",
        "PrefectWorkflowOrchestratorClient",
    }
)

def __getattr__(name: str):
    if name == "register_prefect_integration":
        from intergrax.integrations.providers.workflow_orchestrator.prefect.register import register_prefect_integration

        return register_prefect_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.workflow_orchestrator.prefect import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.workflow_orchestrator.prefect import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.workflow_orchestrator.prefect import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
