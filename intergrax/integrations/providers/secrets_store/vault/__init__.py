# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_vault_secrets_store", "register_vault_integration"]

def __getattr__(name: str):
    if name == "register_vault_integration":
        from intergrax.integrations.providers.secrets_store.vault.register import register_vault_integration
        return register_vault_integration
    if name == "create_vault_secrets_store":
        from intergrax.integrations.providers.secrets_store.vault.bundle import create_vault_secrets_store
        return create_vault_secrets_store
    raise AttributeError(name)
