# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_azure_blob_object_storage", "register_azure_blob_integration"]


def __getattr__(name: str):
    if name == "register_azure_blob_integration":
        from intergrax.integrations.providers.object_storage.azure_blob.register import register_azure_blob_integration

        return register_azure_blob_integration
    if name == "create_azure_blob_object_storage":
        from intergrax.integrations.providers.object_storage.azure_blob.bundle import create_azure_blob_object_storage

        return create_azure_blob_object_storage
    raise AttributeError(name)
