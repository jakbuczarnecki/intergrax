# © Artur Czarnecki. All rights reserved.

"""Speech tool I/O contracts (shared by tools and adapters)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SpeechSynthesizeInput(BaseModel):
    text: str
    voice_id: str = "default"


class SpeechSynthesizeOutput(BaseModel):
    audio_uri: str
    character_count: int = Field(ge=0)


class SpeechTranscribeInput(BaseModel):
    audio_uri: str


class SpeechTranscribeOutput(BaseModel):
    transcript: str
    duration_ms: int = Field(ge=0)
