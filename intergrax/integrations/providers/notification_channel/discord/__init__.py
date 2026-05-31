# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_discord_notification_channel", "register_discord_integration"]

def __getattr__(name: str):
    if name == "register_discord_integration":
        from intergrax.integrations.providers.notification_channel.discord.register import register_discord_integration
        return register_discord_integration
    if name == "create_discord_notification_channel":
        from intergrax.integrations.providers.notification_channel.discord.bundle import create_discord_notification_channel
        return create_discord_notification_channel
    raise AttributeError(name)
