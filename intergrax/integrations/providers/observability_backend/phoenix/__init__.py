# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_phoenix_observability_backend", "register_phoenix_integration"]

def __getattr__(name: str):
    if name == "register_phoenix_integration":
        from intergrax.integrations.providers.observability_backend.phoenix.register import register_phoenix_integration
        return register_phoenix_integration
    if name == "create_phoenix_observability_backend":
        from intergrax.integrations.providers.observability_backend.phoenix.bundle import create_phoenix_observability_backend
        return create_phoenix_observability_backend
    raise AttributeError(name)
