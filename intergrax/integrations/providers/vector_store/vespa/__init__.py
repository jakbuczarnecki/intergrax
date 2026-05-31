# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_vespa_vector_store", "register_vespa_integration"]

def __getattr__(name: str):
    if name == "register_vespa_integration":
        from intergrax.integrations.providers.vector_store.vespa.register import register_vespa_integration
        return register_vespa_integration
    if name == "create_vespa_vector_store":
        from intergrax.integrations.providers.vector_store.vespa.bundle import create_vespa_vector_store
        return create_vespa_vector_store
    raise AttributeError(name)
