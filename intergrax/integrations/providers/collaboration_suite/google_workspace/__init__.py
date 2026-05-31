# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_google_workspace_collaboration_suite", "register_google_workspace_integration"]

def __getattr__(name: str):
    if name == "register_google_workspace_integration":
        from intergrax.integrations.providers.collaboration_suite.google_workspace.register import register_google_workspace_integration
        return register_google_workspace_integration
    if name == "create_google_workspace_collaboration_suite":
        from intergrax.integrations.providers.collaboration_suite.google_workspace.bundle import create_google_workspace_collaboration_suite
        return create_google_workspace_collaboration_suite
    raise AttributeError(name)
