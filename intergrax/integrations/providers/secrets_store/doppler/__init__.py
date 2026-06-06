# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_doppler_secrets_store", "register_doppler_integration"]

def __getattr__(name: str):
    if name == "register_doppler_integration":
        from intergrax.integrations.providers.secrets_store.doppler.register import register_doppler_integration
        return register_doppler_integration
    if name == "create_doppler_secrets_store":
        from intergrax.integrations.providers.secrets_store.doppler.bundle import create_doppler_secrets_store
        return create_doppler_secrets_store
    raise AttributeError(name)
