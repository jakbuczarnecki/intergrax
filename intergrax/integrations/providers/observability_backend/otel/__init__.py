# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_otel_observability_backend", "register_otel_integration"]

def __getattr__(name: str):
    if name == "register_otel_integration":
        from intergrax.integrations.providers.observability_backend.otel.register import register_otel_integration
        return register_otel_integration
    if name == "create_otel_observability_backend":
        from intergrax.integrations.providers.observability_backend.otel.bundle import create_otel_observability_backend
        return create_otel_observability_backend
    raise AttributeError(name)
