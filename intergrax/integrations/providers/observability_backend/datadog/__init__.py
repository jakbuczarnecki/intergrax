# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_datadog_observability_backend", "register_datadog_integration"]

def __getattr__(name: str):
    if name == "register_datadog_integration":
        from intergrax.integrations.providers.observability_backend.datadog.register import register_datadog_integration
        return register_datadog_integration
    if name == "create_datadog_observability_backend":
        from intergrax.integrations.providers.observability_backend.datadog.bundle import create_datadog_observability_backend
        return create_datadog_observability_backend
    raise AttributeError(name)
