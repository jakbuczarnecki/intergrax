# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_cloudflare_r2_object_storage", "register_cloudflare_r2_integration"]

def __getattr__(name: str):
    if name == "register_cloudflare_r2_integration":
        from intergrax.integrations.providers.object_storage.cloudflare_r2.register import register_cloudflare_r2_integration
        return register_cloudflare_r2_integration
    if name == "create_cloudflare_r2_object_storage":
        from intergrax.integrations.providers.object_storage.cloudflare_r2.bundle import create_cloudflare_r2_object_storage
        return create_cloudflare_r2_object_storage
    raise AttributeError(name)
