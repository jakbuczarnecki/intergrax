# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations.providers.object_storage.backblaze_b2.bundle import create_backblaze_b2_object_storage
from intergrax.integrations.providers.object_storage.backblaze_b2.register import register_backblaze_b2_integration

__all__ = ["create_backblaze_b2_object_storage", "register_backblaze_b2_integration"]
