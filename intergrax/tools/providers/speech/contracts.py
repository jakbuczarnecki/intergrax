# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pydantic import BaseModel, Field


class SpeechSynthesizeInput(BaseModel):
    text: str
    voice_id: str = "default"
    language: str = "en"


class SpeechSynthesizeOutput(BaseModel):
    audio_uri: str
    character_count: int


class SpeechTranscribeInput(BaseModel):
    audio_uri: str
    language: str = "en"


class SpeechTranscribeOutput(BaseModel):
    transcript: str
    duration_ms: int = Field(ge=0)
