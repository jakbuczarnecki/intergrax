# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_braintrust_observability_backend", "register_braintrust_integration"]

def __getattr__(name: str):
    if name == "register_braintrust_integration":
        from intergrax.integrations.providers.observability_backend.braintrust.register import register_braintrust_integration
        return register_braintrust_integration
    if name == "create_braintrust_observability_backend":
        from intergrax.integrations.providers.observability_backend.braintrust.bundle import create_braintrust_observability_backend
        return create_braintrust_observability_backend
    raise AttributeError(name)
