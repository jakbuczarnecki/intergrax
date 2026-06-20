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


class SpeechAdapter(ABC):
    """Plane C speech provider contract — explicit subclassing required."""

    @property
    @abstractmethod
    def provider_slug(self) -> str:
        """Integration catalog slug for this adapter."""

    @abstractmethod
    def synthesize(self, payload: SpeechSynthesizeInput) -> SpeechSynthesizeOutput:
        raise NotImplementedError

    @abstractmethod
    def transcribe(self, payload: SpeechTranscribeInput) -> SpeechTranscribeOutput:
        raise NotImplementedError

    def validate(self) -> None:
        slug = self.provider_slug
        if not isinstance(slug, str) or not slug.strip():
            raise ValueError(f"{type(self).__name__}.provider_slug must be a non-empty string slug")
