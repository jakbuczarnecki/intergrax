# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_honeycomb_observability_backend", "register_honeycomb_integration"]

def __getattr__(name: str):
    if name == "register_honeycomb_integration":
        from intergrax.integrations.providers.observability_backend.honeycomb.register import register_honeycomb_integration
        return register_honeycomb_integration
    if name == "create_honeycomb_observability_backend":
        from intergrax.integrations.providers.observability_backend.honeycomb.bundle import create_honeycomb_observability_backend
        return create_honeycomb_observability_backend
    raise AttributeError(name)
