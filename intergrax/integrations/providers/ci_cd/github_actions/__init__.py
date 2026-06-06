# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_github_actions_ci_cd", "register_github_actions_integration"]

def __getattr__(name: str):
    if name == "register_github_actions_integration":
        from intergrax.integrations.providers.ci_cd.github_actions.register import register_github_actions_integration
        return register_github_actions_integration
    if name == "create_github_actions_ci_cd":
        from intergrax.integrations.providers.ci_cd.github_actions.bundle import create_github_actions_ci_cd
        return create_github_actions_ci_cd
    raise AttributeError(name)
