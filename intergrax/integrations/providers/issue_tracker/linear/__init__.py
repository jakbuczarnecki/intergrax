# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_linear_issue_tracker", "register_linear_integration"]

def __getattr__(name: str):
    if name == "register_linear_integration":
        from intergrax.integrations.providers.issue_tracker.linear.register import register_linear_integration
        return register_linear_integration
    if name == "create_linear_issue_tracker":
        from intergrax.integrations.providers.issue_tracker.linear.bundle import create_linear_issue_tracker
        return create_linear_issue_tracker
    raise AttributeError(name)
