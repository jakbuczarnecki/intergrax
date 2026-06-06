# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations.providers.notification_channel.mailpit.bundle import create_mailpit_notification_channel
from intergrax.integrations.providers.notification_channel.mailpit.register import register_mailpit_integration

__all__ = ["create_mailpit_notification_channel", "register_mailpit_integration"]
