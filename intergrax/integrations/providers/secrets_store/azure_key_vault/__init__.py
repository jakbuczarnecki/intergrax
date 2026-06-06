# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_azure_key_vault_secrets_store", "register_azure_key_vault_integration"]

def __getattr__(name: str):
    if name == "register_azure_key_vault_integration":
        from intergrax.integrations.providers.secrets_store.azure_key_vault.register import register_azure_key_vault_integration
        return register_azure_key_vault_integration
    if name == "create_azure_key_vault_secrets_store":
        from intergrax.integrations.providers.secrets_store.azure_key_vault.bundle import create_azure_key_vault_secrets_store
        return create_azure_key_vault_secrets_store
    raise AttributeError(name)
