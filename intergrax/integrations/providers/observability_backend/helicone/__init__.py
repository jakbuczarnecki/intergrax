# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_helicone_observability_backend", "register_helicone_integration"]

def __getattr__(name: str):
    if name == "register_helicone_integration":
        from intergrax.integrations.providers.observability_backend.helicone.register import register_helicone_integration
        return register_helicone_integration
    if name == "create_helicone_observability_backend":
        from intergrax.integrations.providers.observability_backend.helicone.bundle import create_helicone_observability_backend
        return create_helicone_observability_backend
    raise AttributeError(name)
