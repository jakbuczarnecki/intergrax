# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_asana_issue_tracker", "register_asana_integration"]

def __getattr__(name: str):
    if name == "register_asana_integration":
        from intergrax.integrations.providers.issue_tracker.asana.register import register_asana_integration
        return register_asana_integration
    if name == "create_asana_issue_tracker":
        from intergrax.integrations.providers.issue_tracker.asana.bundle import create_asana_issue_tracker
        return create_asana_issue_tracker
    raise AttributeError(name)
