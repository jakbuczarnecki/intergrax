# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_huggingface_hub_object_storage", "register_huggingface_hub_integration"]

def __getattr__(name: str):
    if name == "register_huggingface_hub_integration":
        from intergrax.integrations.providers.object_storage.huggingface_hub.register import register_huggingface_hub_integration
        return register_huggingface_hub_integration
    if name == "create_huggingface_hub_object_storage":
        from intergrax.integrations.providers.object_storage.huggingface_hub.bundle import create_huggingface_hub_object_storage
        return create_huggingface_hub_object_storage
    raise AttributeError(name)
