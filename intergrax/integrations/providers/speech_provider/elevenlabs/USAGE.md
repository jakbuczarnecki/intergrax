# Elevenlabs (elevenlabs)

Category: `speech_provider`

## Single public entrypoint

- **`ElevenlabsSpeechProviderIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `ElevenlabsSpeechProviderIntegration`.
- Contract factory: `create_elevenlabs_speech_provider_integration()`.
