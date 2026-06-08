# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations.providers.object_storage.google_drive.bundle import create_google_drive_object_storage
from intergrax.integrations.providers.object_storage.google_drive.register import register_google_drive_integration

__all__ = ["create_google_drive_object_storage", "register_google_drive_integration"]
