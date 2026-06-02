# © Artur Czarnecki. All rights reserved.

"""Universal speech adapter interface (mirrors ``LLMAdapter``)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from intergrax.speech_adapters.contracts.io import (
    SpeechSynthesizeInput,
    SpeechSynthesizeOutput,
    SpeechTranscribeInput,
    SpeechTranscribeOutput,
)
from intergrax.speech_adapters.contracts.speech_provider import SpeechProvider


class SpeechAdapter(ABC):
    """Plane C speech provider contract — explicit subclassing required."""

    provider: SpeechProvider

    @abstractmethod
    def synthesize(self, payload: SpeechSynthesizeInput) -> SpeechSynthesizeOutput:
        raise NotImplementedError

    @abstractmethod
    def transcribe(self, payload: SpeechTranscribeInput) -> SpeechTranscribeOutput:
        raise NotImplementedError

    def validate(self) -> None:
        if not isinstance(self.provider, SpeechProvider):
            raise ValueError(f"{type(self).__name__}.provider must be SpeechProvider")
