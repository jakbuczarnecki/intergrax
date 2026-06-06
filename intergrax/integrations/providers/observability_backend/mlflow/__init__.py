# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_mlflow_observability_backend", "register_mlflow_integration"]

def __getattr__(name: str):
    if name == "register_mlflow_integration":
        from intergrax.integrations.providers.observability_backend.mlflow.register import register_mlflow_integration
        return register_mlflow_integration
    if name == "create_mlflow_observability_backend":
        from intergrax.integrations.providers.observability_backend.mlflow.bundle import create_mlflow_observability_backend
        return create_mlflow_observability_backend
    raise AttributeError(name)
