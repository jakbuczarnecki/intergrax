# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_filesystem_object_storage", "register_filesystem_integration"]

def __getattr__(name: str):
    if name == "register_filesystem_integration":
        from intergrax.integrations.providers.object_storage.filesystem.register import register_filesystem_integration
        return register_filesystem_integration
    if name == "create_filesystem_object_storage":
        from intergrax.integrations.providers.object_storage.filesystem.bundle import create_filesystem_object_storage
        return create_filesystem_object_storage
    raise AttributeError(name)
