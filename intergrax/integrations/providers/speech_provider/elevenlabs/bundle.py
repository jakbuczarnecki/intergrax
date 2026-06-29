# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p7.factories import create_elevenlabs_speech_provider as _legacy_create_elevenlabs_speech_provider

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.speech_provider.elevenlabs.integration import (
    ELEVENLABS_SPEECH_PROVIDER_PROVIDER_ID,
    ElevenlabsSpeechProviderIntegration,
    ElevenlabsSpeechProviderIntegrationConfig,
    ElevenlabsSpeechProviderClient,
)

__all__ = [
    "create_elevenlabs_speech_provider",
    "create_elevenlabs_speech_provider_integration",
]


def create_elevenlabs_speech_provider_integration(
    *,
    client: ElevenlabsSpeechProviderClient | None = None,
    enabled: bool = False,
) -> ElevenlabsSpeechProviderIntegration:
    """
    Build a contract-based Elevenlabs speech provider integration.

    The legacy facade (create_elevenlabs_speech_provider) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Elevenlabs speech provider integration requires an injected client when enabled=True",
        )
    if client is not None:
        return ElevenlabsSpeechProviderIntegration.from_client(client, enabled=enabled)
    return ElevenlabsSpeechProviderIntegration.for_provider(
        provider_id=ELEVENLABS_SPEECH_PROVIDER_PROVIDER_ID,
        display_name="Elevenlabs",
        config=ElevenlabsSpeechProviderIntegrationConfig(enabled=enabled),
    )


def create_elevenlabs_speech_provider(**kwargs: object) -> ElevenlabsSpeechProviderIntegration:
    """Compatibility shim — constructs ElevenlabsSpeechProviderIntegration from legacy runtime."""
    runtime = _legacy_create_elevenlabs_speech_provider(**kwargs)
    if isinstance(runtime, ElevenlabsSpeechProviderIntegration):
        return runtime
    return ElevenlabsSpeechProviderIntegration.from_client(runtime)
