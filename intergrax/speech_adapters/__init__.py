# © Artur Czarnecki. All rights reserved.

"""Dedicated speech inference plane (Phase W-ML.2) — mirrors ``llm_adapters`` layout."""

from intergrax.speech_adapters.contracts.speech_adapter import SpeechAdapter
from intergrax.speech_adapters.contracts.speech_provider import SpeechProvider
from intergrax.speech_adapters.registry.profile import SpeechProfile, speech_profile_from_env

__all__ = [
    "SpeechAdapter",
    "SpeechProfile",
    "SpeechProvider",
    "speech_profile_from_env",
]
