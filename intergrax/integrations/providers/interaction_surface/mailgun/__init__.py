# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_mailgun_interaction_surface", "register_mailgun_integration"]

def __getattr__(name: str):
    if name == "register_mailgun_integration":
        from intergrax.integrations.providers.interaction_surface.mailgun.register import register_mailgun_integration
        return register_mailgun_integration
    if name == "create_mailgun_interaction_surface":
        from intergrax.integrations.providers.interaction_surface.mailgun.bundle import create_mailgun_interaction_surface
        return create_mailgun_interaction_surface
    raise AttributeError(name)
