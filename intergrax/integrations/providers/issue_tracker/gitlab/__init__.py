# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_gitlab_issue_tracker", "register_gitlab_integration"]

def __getattr__(name: str):
    if name == "register_gitlab_integration":
        from intergrax.integrations.providers.issue_tracker.gitlab.register import register_gitlab_integration
        return register_gitlab_integration
    if name == "create_gitlab_issue_tracker":
        from intergrax.integrations.providers.issue_tracker.gitlab.bundle import create_gitlab_issue_tracker
        return create_gitlab_issue_tracker
    raise AttributeError(name)
