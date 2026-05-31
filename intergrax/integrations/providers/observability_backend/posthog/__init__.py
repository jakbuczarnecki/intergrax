# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_posthog_observability_backend", "register_posthog_integration"]

def __getattr__(name: str):
    if name == "register_posthog_integration":
        from intergrax.integrations.providers.observability_backend.posthog.register import register_posthog_integration
        return register_posthog_integration
    if name == "create_posthog_observability_backend":
        from intergrax.integrations.providers.observability_backend.posthog.bundle import create_posthog_observability_backend
        return create_posthog_observability_backend
    raise AttributeError(name)
