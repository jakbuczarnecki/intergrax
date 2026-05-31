# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_minio_object_storage", "register_minio_integration"]

def __getattr__(name: str):
    if name == "register_minio_integration":
        from intergrax.integrations.providers.object_storage.minio.register import register_minio_integration
        return register_minio_integration
    if name == "create_minio_object_storage":
        from intergrax.integrations.providers.object_storage.minio.bundle import create_minio_object_storage
        return create_minio_object_storage
    raise AttributeError(name)
