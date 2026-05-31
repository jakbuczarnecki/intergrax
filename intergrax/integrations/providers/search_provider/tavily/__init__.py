# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_tavily_search_provider", "register_tavily_integration"]

def __getattr__(name: str):
    if name == "register_tavily_integration":
        from intergrax.integrations.providers.search_provider.tavily.register import register_tavily_integration
        return register_tavily_integration
    if name == "create_tavily_search_provider":
        from intergrax.integrations.providers.search_provider.tavily.bundle import create_tavily_search_provider
        return create_tavily_search_provider
    raise AttributeError(name)
