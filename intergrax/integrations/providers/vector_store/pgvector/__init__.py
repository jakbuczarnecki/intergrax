# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_pgvector_vector_store", "register_pgvector_integration"]

def __getattr__(name: str):
    if name == "register_pgvector_integration":
        from intergrax.integrations.providers.vector_store.pgvector.register import register_pgvector_integration
        return register_pgvector_integration
    if name == "create_pgvector_vector_store":
        from intergrax.integrations.providers.vector_store.pgvector.bundle import create_pgvector_vector_store
        return create_pgvector_vector_store
    raise AttributeError(name)
