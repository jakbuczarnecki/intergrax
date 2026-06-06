# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_loki_observability_backend", "register_loki_integration"]

def __getattr__(name: str):
    if name == "register_loki_integration":
        from intergrax.integrations.providers.observability_backend.loki.register import register_loki_integration
        return register_loki_integration
    if name == "create_loki_observability_backend":
        from intergrax.integrations.providers.observability_backend.loki.bundle import create_loki_observability_backend
        return create_loki_observability_backend
    raise AttributeError(name)
