# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_tempo_observability_backend", "register_tempo_integration"]

def __getattr__(name: str):
    if name == "register_tempo_integration":
        from intergrax.integrations.providers.observability_backend.tempo.register import register_tempo_integration
        return register_tempo_integration
    if name == "create_tempo_observability_backend":
        from intergrax.integrations.providers.observability_backend.tempo.bundle import create_tempo_observability_backend
        return create_tempo_observability_backend
    raise AttributeError(name)
