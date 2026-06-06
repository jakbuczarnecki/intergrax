# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_influxdb_observability_backend", "register_influxdb_integration"]

def __getattr__(name: str):
    if name == "register_influxdb_integration":
        from intergrax.integrations.providers.observability_backend.influxdb.register import register_influxdb_integration
        return register_influxdb_integration
    if name == "create_influxdb_observability_backend":
        from intergrax.integrations.providers.observability_backend.influxdb.bundle import create_influxdb_observability_backend
        return create_influxdb_observability_backend
    raise AttributeError(name)
