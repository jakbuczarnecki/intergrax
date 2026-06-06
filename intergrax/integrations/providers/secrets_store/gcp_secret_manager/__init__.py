# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_gcp_secret_manager_secrets_store", "register_gcp_secret_manager_integration"]

def __getattr__(name: str):
    if name == "register_gcp_secret_manager_integration":
        from intergrax.integrations.providers.secrets_store.gcp_secret_manager.register import register_gcp_secret_manager_integration
        return register_gcp_secret_manager_integration
    if name == "create_gcp_secret_manager_secrets_store":
        from intergrax.integrations.providers.secrets_store.gcp_secret_manager.bundle import create_gcp_secret_manager_secrets_store
        return create_gcp_secret_manager_secrets_store
    raise AttributeError(name)
