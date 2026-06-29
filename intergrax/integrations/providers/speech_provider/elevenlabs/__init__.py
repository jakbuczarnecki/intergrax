# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "ELEVENLABS_SPEECH_PROVIDER_PROVIDER_ID",
    "ElevenlabsSpeechProviderIntegration",
    "ElevenlabsSpeechProviderIntegrationConfig",
    "ElevenlabsSpeechProviderClient",
    "create_elevenlabs_speech_provider",
    "create_elevenlabs_speech_provider_integration",
    "register_elevenlabs_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_elevenlabs_speech_provider",
        "create_elevenlabs_speech_provider_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "ELEVENLABS_SPEECH_PROVIDER_PROVIDER_ID",
        "ElevenlabsSpeechProviderIntegration",
        "ElevenlabsSpeechProviderIntegrationConfig",
        "ElevenlabsSpeechProviderClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "ELEVENLABS_SPEECH_PROVIDER_PROVIDER_ID",
        "ElevenlabsSpeechProviderIntegration",
        "ElevenlabsSpeechProviderIntegrationConfig",
        "ElevenlabsSpeechProviderClient",
    }
)

def __getattr__(name: str):
    if name == "register_elevenlabs_integration":
        from intergrax.integrations.providers.speech_provider.elevenlabs.register import register_elevenlabs_integration

        return register_elevenlabs_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.speech_provider.elevenlabs import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.speech_provider.elevenlabs import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.speech_provider.elevenlabs import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
