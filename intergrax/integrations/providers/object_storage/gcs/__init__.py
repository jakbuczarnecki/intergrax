# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_gcs_object_storage", "register_gcs_integration"]

def __getattr__(name: str):
    if name == "register_gcs_integration":
        from intergrax.integrations.providers.object_storage.gcs.register import register_gcs_integration
        return register_gcs_integration
    if name == "create_gcs_object_storage":
        from intergrax.integrations.providers.object_storage.gcs.bundle import create_gcs_object_storage
        return create_gcs_object_storage
    raise AttributeError(name)
