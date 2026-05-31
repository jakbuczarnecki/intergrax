# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_langfuse_observability_backend", "register_langfuse_integration"]

def __getattr__(name: str):
    if name == "register_langfuse_integration":
        from intergrax.integrations.providers.observability_backend.langfuse.register import register_langfuse_integration
        return register_langfuse_integration
    if name == "create_langfuse_observability_backend":
        from intergrax.integrations.providers.observability_backend.langfuse.bundle import create_langfuse_observability_backend
        return create_langfuse_observability_backend
    raise AttributeError(name)
