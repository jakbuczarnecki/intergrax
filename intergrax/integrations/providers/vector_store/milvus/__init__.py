# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_milvus_vector_store", "register_milvus_integration"]

def __getattr__(name: str):
    if name == "register_milvus_integration":
        from intergrax.integrations.providers.vector_store.milvus.register import register_milvus_integration
        return register_milvus_integration
    if name == "create_milvus_vector_store":
        from intergrax.integrations.providers.vector_store.milvus.bundle import create_milvus_vector_store
        return create_milvus_vector_store
    raise AttributeError(name)
