# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_opensearch_observability_backend", "register_opensearch_integration"]

def __getattr__(name: str):
    if name == "register_opensearch_integration":
        from intergrax.integrations.providers.observability_backend.opensearch.register import register_opensearch_integration
        return register_opensearch_integration
    if name == "create_opensearch_observability_backend":
        from intergrax.integrations.providers.observability_backend.opensearch.bundle import create_opensearch_observability_backend
        return create_opensearch_observability_backend
    raise AttributeError(name)
