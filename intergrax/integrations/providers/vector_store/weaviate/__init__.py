# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_weaviate_vector_store", "register_weaviate_integration"]

def __getattr__(name: str):
    if name == "register_weaviate_integration":
        from intergrax.integrations.providers.vector_store.weaviate.register import register_weaviate_integration
        return register_weaviate_integration
    if name == "create_weaviate_vector_store":
        from intergrax.integrations.providers.vector_store.weaviate.bundle import create_weaviate_vector_store
        return create_weaviate_vector_store
    raise AttributeError(name)
