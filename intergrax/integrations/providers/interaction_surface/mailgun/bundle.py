# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p5.factories import create_mailgun_interaction_surface as _legacy_create_mailgun_interaction_surface

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.interaction_surface.mailgun.integration import (
    MAILGUN_INTERACTION_SURFACE_PROVIDER_ID,
    MailgunInteractionSurfaceIntegration,
    MailgunInteractionSurfaceIntegrationConfig,
    MailgunInteractionSurfaceClient,
)

__all__ = [
    "create_mailgun_interaction_surface",
    "create_mailgun_interaction_surface_integration",
]


def create_mailgun_interaction_surface_integration(
    *,
    client: MailgunInteractionSurfaceClient | None = None,
    enabled: bool = False,
) -> MailgunInteractionSurfaceIntegration:
    """
    Build a contract-based Mailgun interaction surface integration.

    The legacy facade (create_mailgun_interaction_surface) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Mailgun interaction surface integration requires an injected client when enabled=True",
        )
    if client is not None:
        return MailgunInteractionSurfaceIntegration.from_client(client, enabled=enabled)
    return MailgunInteractionSurfaceIntegration.for_provider(
        provider_id=MAILGUN_INTERACTION_SURFACE_PROVIDER_ID,
        display_name="Mailgun",
        config=MailgunInteractionSurfaceIntegrationConfig(enabled=enabled),
    )


def create_mailgun_interaction_surface(**kwargs: object) -> MailgunInteractionSurfaceIntegration:
    """Compatibility shim — constructs MailgunInteractionSurfaceIntegration from legacy runtime."""
    runtime = _legacy_create_mailgun_interaction_surface(**kwargs)
    if isinstance(runtime, MailgunInteractionSurfaceIntegration):
        return runtime
    return MailgunInteractionSurfaceIntegration.from_client(runtime)
