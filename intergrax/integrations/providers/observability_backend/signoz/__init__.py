# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_signoz_observability_backend", "register_signoz_integration"]

def __getattr__(name: str):
    if name == "register_signoz_integration":
        from intergrax.integrations.providers.observability_backend.signoz.register import register_signoz_integration
        return register_signoz_integration
    if name == "create_signoz_observability_backend":
        from intergrax.integrations.providers.observability_backend.signoz.bundle import create_signoz_observability_backend
        return create_signoz_observability_backend
    raise AttributeError(name)
