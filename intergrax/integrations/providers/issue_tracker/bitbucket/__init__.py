# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_bitbucket_issue_tracker", "register_bitbucket_integration"]

def __getattr__(name: str):
    if name == "register_bitbucket_integration":
        from intergrax.integrations.providers.issue_tracker.bitbucket.register import register_bitbucket_integration
        return register_bitbucket_integration
    if name == "create_bitbucket_issue_tracker":
        from intergrax.integrations.providers.issue_tracker.bitbucket.bundle import create_bitbucket_issue_tracker
        return create_bitbucket_issue_tracker
    raise AttributeError(name)
