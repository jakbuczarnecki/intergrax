# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations.providers.speech_provider.elevenlabs.bundle import create_elevenlabs_speech_provider
from intergrax.integrations.providers.speech_provider.elevenlabs.register import register_elevenlabs_integration

__all__ = ["create_elevenlabs_speech_provider", "register_elevenlabs_integration"]
