# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "DEEPGRAM_SPEECH_PROVIDER_PROVIDER_ID",
    "DeepgramSpeechProviderIntegration",
    "DeepgramSpeechProviderIntegrationConfig",
    "DeepgramSpeechProviderClient",
    "create_deepgram_speech_provider",
    "create_deepgram_speech_provider_integration",
    "register_deepgram_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_deepgram_speech_provider",
        "create_deepgram_speech_provider_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "DEEPGRAM_SPEECH_PROVIDER_PROVIDER_ID",
        "DeepgramSpeechProviderIntegration",
        "DeepgramSpeechProviderIntegrationConfig",
        "DeepgramSpeechProviderClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "DEEPGRAM_SPEECH_PROVIDER_PROVIDER_ID",
        "DeepgramSpeechProviderIntegration",
        "DeepgramSpeechProviderIntegrationConfig",
        "DeepgramSpeechProviderClient",
    }
)

def __getattr__(name: str):
    if name == "register_deepgram_integration":
        from intergrax.integrations.providers.speech_provider.deepgram.register import register_deepgram_integration

        return register_deepgram_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.speech_provider.deepgram import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.speech_provider.deepgram import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.speech_provider.deepgram import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
