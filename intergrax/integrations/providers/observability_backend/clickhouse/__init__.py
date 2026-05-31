# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_clickhouse_observability_backend", "register_clickhouse_integration"]

def __getattr__(name: str):
    if name == "register_clickhouse_integration":
        from intergrax.integrations.providers.observability_backend.clickhouse.register import register_clickhouse_integration
        return register_clickhouse_integration
    if name == "create_clickhouse_observability_backend":
        from intergrax.integrations.providers.observability_backend.clickhouse.bundle import create_clickhouse_observability_backend
        return create_clickhouse_observability_backend
    raise AttributeError(name)
