# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_inmemory_vector_store", "register_inmemory_integration"]

def __getattr__(name: str):
    if name == "register_inmemory_integration":
        from intergrax.integrations.providers.vector_store.inmemory.register import register_inmemory_integration
        return register_inmemory_integration
    if name == "create_inmemory_vector_store":
        from intergrax.integrations.providers.vector_store.inmemory.bundle import create_inmemory_vector_store
        return create_inmemory_vector_store
    raise AttributeError(name)
