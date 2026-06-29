# Deepgram (deepgram)

Category: `speech_provider`

## Single public entrypoint

- **`DeepgramSpeechProviderIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `DeepgramSpeechProviderIntegration`.
- Contract factory: `create_deepgram_speech_provider_integration()`.
