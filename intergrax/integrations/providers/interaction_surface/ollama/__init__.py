# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_ollama_interaction_surface", "register_ollama_integration"]

def __getattr__(name: str):
    if name == "register_ollama_integration":
        from intergrax.integrations.providers.interaction_surface.ollama.register import register_ollama_integration
        return register_ollama_integration
    if name == "create_ollama_interaction_surface":
        from intergrax.integrations.providers.interaction_surface.ollama.bundle import create_ollama_interaction_surface
        return create_ollama_interaction_surface
    raise AttributeError(name)
