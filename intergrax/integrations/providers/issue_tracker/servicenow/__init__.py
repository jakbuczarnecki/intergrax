# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_servicenow_issue_tracker", "register_servicenow_integration"]

def __getattr__(name: str):
    if name == "register_servicenow_integration":
        from intergrax.integrations.providers.issue_tracker.servicenow.register import register_servicenow_integration
        return register_servicenow_integration
    if name == "create_servicenow_issue_tracker":
        from intergrax.integrations.providers.issue_tracker.servicenow.bundle import create_servicenow_issue_tracker
        return create_servicenow_issue_tracker
    raise AttributeError(name)
