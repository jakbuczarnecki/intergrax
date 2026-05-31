# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_brave_search_provider", "register_brave_integration"]

def __getattr__(name: str):
    if name == "register_brave_integration":
        from intergrax.integrations.providers.search_provider.brave.register import register_brave_integration
        return register_brave_integration
    if name == "create_brave_search_provider":
        from intergrax.integrations.providers.search_provider.brave.bundle import create_brave_search_provider
        return create_brave_search_provider
    raise AttributeError(name)
