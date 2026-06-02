# © Artur Czarnecki. All rights reserved.

"""Speech provider slugs (mirrors ``LLMProvider``)."""

from __future__ import annotations

from enum import Enum


class SpeechProvider(str, Enum):
    """Harness-registered speech synthesis/transcription backends."""

    STUB = "stub"
    ELEVENLABS = "elevenlabs"
