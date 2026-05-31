# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_serpapi_search_provider", "register_serpapi_integration"]

def __getattr__(name: str):
    if name == "register_serpapi_integration":
        from intergrax.integrations.providers.search_provider.serpapi.register import register_serpapi_integration
        return register_serpapi_integration
    if name == "create_serpapi_search_provider":
        from intergrax.integrations.providers.search_provider.serpapi.bundle import create_serpapi_search_provider
        return create_serpapi_search_provider
    raise AttributeError(name)
