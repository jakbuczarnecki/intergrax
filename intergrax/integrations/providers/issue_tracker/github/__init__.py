# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_github_issue_tracker", "register_github_integration"]

def __getattr__(name: str):
    if name == "register_github_integration":
        from intergrax.integrations.providers.issue_tracker.github.register import register_github_integration
        return register_github_integration
    if name == "create_github_issue_tracker":
        from intergrax.integrations.providers.issue_tracker.github.bundle import create_github_issue_tracker
        return create_github_issue_tracker
    raise AttributeError(name)
