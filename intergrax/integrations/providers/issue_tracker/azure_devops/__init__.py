# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_azure_devops_issue_tracker", "register_azure_devops_integration"]

def __getattr__(name: str):
    if name == "register_azure_devops_integration":
        from intergrax.integrations.providers.issue_tracker.azure_devops.register import register_azure_devops_integration
        return register_azure_devops_integration
    if name == "create_azure_devops_issue_tracker":
        from intergrax.integrations.providers.issue_tracker.azure_devops.bundle import create_azure_devops_issue_tracker
        return create_azure_devops_issue_tracker
    raise AttributeError(name)
