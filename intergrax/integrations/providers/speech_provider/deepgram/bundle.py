# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p7.factories import create_deepgram_speech_provider

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.speech_provider.deepgram.integration import (
    DEEPGRAM_SPEECH_PROVIDER_PROVIDER_ID,
    DeepgramSpeechProviderIntegration,
    DeepgramSpeechProviderIntegrationConfig,
    DeepgramSpeechProviderClient,
)

__all__ = [
    "create_deepgram_speech_provider",
    "create_deepgram_speech_provider_integration",
]


def create_deepgram_speech_provider_integration(
    *,
    client: DeepgramSpeechProviderClient | None = None,
    enabled: bool = False,
) -> DeepgramSpeechProviderIntegration:
    """
    Build a contract-based Deepgram speech provider integration.

    The legacy facade (create_deepgram_speech_provider) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Deepgram speech provider integration requires an injected client when enabled=True",
        )
    if client is not None:
        return DeepgramSpeechProviderIntegration.from_client(client, enabled=enabled)
    return DeepgramSpeechProviderIntegration.for_provider(
        provider_id=DEEPGRAM_SPEECH_PROVIDER_PROVIDER_ID,
        display_name="Deepgram",
        config=DeepgramSpeechProviderIntegrationConfig(enabled=enabled),
    )
