# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_langsmith_observability_backend", "register_langsmith_integration"]

def __getattr__(name: str):
    if name == "register_langsmith_integration":
        from intergrax.integrations.providers.observability_backend.langsmith.register import register_langsmith_integration
        return register_langsmith_integration
    if name == "create_langsmith_observability_backend":
        from intergrax.integrations.providers.observability_backend.langsmith.bundle import create_langsmith_observability_backend
        return create_langsmith_observability_backend
    raise AttributeError(name)
