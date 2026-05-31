# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_exa_search_provider", "register_exa_integration"]

def __getattr__(name: str):
    if name == "register_exa_integration":
        from intergrax.integrations.providers.search_provider.exa.register import register_exa_integration
        return register_exa_integration
    if name == "create_exa_search_provider":
        from intergrax.integrations.providers.search_provider.exa.bundle import create_exa_search_provider
        return create_exa_search_provider
    raise AttributeError(name)
