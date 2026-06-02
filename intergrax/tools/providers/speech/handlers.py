# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.speech.contracts import (
    SpeechSynthesizeInput,
    SpeechSynthesizeOutput,
    SpeechTranscribeInput,
    SpeechTranscribeOutput,
)
from intergrax.tools.providers.speech.service import speech_synthesize, speech_transcribe


class SpeechSynthesizeHandler(ServiceToolHandler[SpeechSynthesizeInput, SpeechSynthesizeOutput]):
    _service = speech_synthesize


class SpeechTranscribeHandler(ServiceToolHandler[SpeechTranscribeInput, SpeechTranscribeOutput]):
    _service = speech_transcribe
