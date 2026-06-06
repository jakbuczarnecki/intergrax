# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_grafana_observability_backend", "register_grafana_integration"]

def __getattr__(name: str):
    if name == "register_grafana_integration":
        from intergrax.integrations.providers.observability_backend.grafana.register import register_grafana_integration
        return register_grafana_integration
    if name == "create_grafana_observability_backend":
        from intergrax.integrations.providers.observability_backend.grafana.bundle import create_grafana_observability_backend
        return create_grafana_observability_backend
    raise AttributeError(name)
