# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations.providers.speech_provider.deepgram.bundle import create_deepgram_speech_provider
from intergrax.integrations.providers.speech_provider.deepgram.register import register_deepgram_integration

__all__ = ["create_deepgram_speech_provider", "register_deepgram_integration"]
