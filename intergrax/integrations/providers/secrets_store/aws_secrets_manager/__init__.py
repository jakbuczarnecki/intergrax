# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_aws_secrets_manager_secrets_store", "register_aws_secrets_manager_integration"]

def __getattr__(name: str):
    if name == "register_aws_secrets_manager_integration":
        from intergrax.integrations.providers.secrets_store.aws_secrets_manager.register import register_aws_secrets_manager_integration
        return register_aws_secrets_manager_integration
    if name == "create_aws_secrets_manager_secrets_store":
        from intergrax.integrations.providers.secrets_store.aws_secrets_manager.bundle import create_aws_secrets_manager_secrets_store
        return create_aws_secrets_manager_secrets_store
    raise AttributeError(name)
