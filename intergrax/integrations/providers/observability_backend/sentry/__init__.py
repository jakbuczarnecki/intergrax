# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_sentry_observability_backend", "register_sentry_integration"]

def __getattr__(name: str):
    if name == "register_sentry_integration":
        from intergrax.integrations.providers.observability_backend.sentry.register import register_sentry_integration
        return register_sentry_integration
    if name == "create_sentry_observability_backend":
        from intergrax.integrations.providers.observability_backend.sentry.bundle import create_sentry_observability_backend
        return create_sentry_observability_backend
    raise AttributeError(name)
