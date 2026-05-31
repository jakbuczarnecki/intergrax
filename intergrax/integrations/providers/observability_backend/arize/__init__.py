# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_arize_observability_backend", "register_arize_integration"]

def __getattr__(name: str):
    if name == "register_arize_integration":
        from intergrax.integrations.providers.observability_backend.arize.register import register_arize_integration
        return register_arize_integration
    if name == "create_arize_observability_backend":
        from intergrax.integrations.providers.observability_backend.arize.bundle import create_arize_observability_backend
        return create_arize_observability_backend
    raise AttributeError(name)
