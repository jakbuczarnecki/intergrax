# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for Elevenlabs speech provider."""

from __future__ import annotations

from intergrax.integrations.providers.speech_provider.elevenlabs.bundle import (
    create_elevenlabs_speech_provider_integration,
)
from intergrax.integrations.providers.speech_provider.elevenlabs.integration import (
    ELEVENLABS_SPEECH_PROVIDER_PROVIDER_ID,
    ElevenlabsSpeechProviderIntegration,
    ElevenlabsSpeechProviderIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.ai import SpeechProviderIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="speech_provider",
    provider_id=ELEVENLABS_SPEECH_PROVIDER_PROVIDER_ID,
    integration_class=ElevenlabsSpeechProviderIntegration,
    contract_class=SpeechProviderIntegrationContract,
    contract_factory=create_elevenlabs_speech_provider_integration,
    display_name="Elevenlabs",
    config_class=ElevenlabsSpeechProviderIntegrationConfig,
    capabilities=(
        PlatformIntegrationCapability.CONNECT,
        PlatformIntegrationCapability.READ,
        PlatformIntegrationCapability.HEALTH_CHECK,
    ),
    security_posture=PlatformIntegrationSecurityPosture(),
    supports_runtime_binding=True,
    supports_health_check=True,
    metadata={"source": "explicit_provider_declaration", "speech_features": ('tts', 'stt')},
)

CONTRACT_SPECS = (CONTRACT_SPEC,)

__all__ = ["CONTRACT_SPEC", "CONTRACT_SPECS"]
