# © Artur Czarnecki. All rights reserved.

"""Declarative Tier-3 speech provider selection (mirrors ``LLMProfile``)."""

from __future__ import annotations

import os
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.speech_adapters.contracts.speech_adapter import SpeechAdapter
from intergrax.speech_adapters.contracts.speech_provider import SpeechProvider
from intergrax.speech_adapters.registry.secrets import merge_secrets_into_options
from intergrax.speech_adapters.registry.speech_adapter_registry import SpeechAdapterRegistry

SPEECH_PROFILE_EXTRA_KEY = "speech_profile"


class SpeechProfile(BaseModel):
    """
    Typed speech provider + voice defaults + constructor options.

    Example::

        profile = SpeechProfile(provider=SpeechProvider.ELEVENLABS, voice_id="my-voice")
        adapter = profile.create_adapter(secrets={"api_key": "..."})
    """

    model_config = ConfigDict(extra="forbid", use_enum_values=False)

    provider: SpeechProvider
    voice_id: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("provider", mode="before")
    @classmethod
    def _coerce_provider(cls, value: str | SpeechProvider) -> SpeechProvider:
        if isinstance(value, SpeechProvider):
            return value
        if isinstance(value, str) and value.strip():
            return SpeechProvider(value.strip().lower())
        raise ValueError("provider must be a non-empty SpeechProvider or string slug")

    def create_adapter(
        self,
        *,
        secrets: Mapping[str, str] | None = None,
        **overrides: Any,
    ) -> SpeechAdapter:
        kwargs = merge_secrets_into_options(self.provider, {**self.options, **overrides}, secrets)
        if self.voice_id:
            kwargs.setdefault("default_voice_id", self.voice_id)
        if self.provider == SpeechProvider.ELEVENLABS and not kwargs.get("api_key"):
            raise ValueError(
                "ElevenLabs speech requires api_key in options, secrets=, or ELEVENLABS_API_KEY"
            )
        return SpeechAdapterRegistry.create(self.provider, **kwargs)

    def with_secrets(self, secrets: Mapping[str, str]) -> SpeechProfile:
        return self.model_copy(
            update={"options": merge_secrets_into_options(self.provider, dict(self.options), secrets)}
        )

    @classmethod
    def lab(cls) -> SpeechProfile:
        """Laboratory default — stub TTS/STT unless API key is configured."""
        return cls(provider=SpeechProvider.STUB)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> SpeechProfile:
        return cls.model_validate(dict(data))


def speech_profile_from_env(*, prefix: str = "INTERGRAX_SPEECH") -> SpeechProfile:
    """
    Build profile from environment:

    - ``{PREFIX}_PROVIDER`` (``stub`` | ``elevenlabs``; default ``stub`` unless ``ELEVENLABS_API_KEY`` set)
    - ``{PREFIX}_VOICE_ID`` (optional default voice)
    """
    provider_raw = os.getenv(f"{prefix}_PROVIDER")
    voice_id = os.getenv(f"{prefix}_VOICE_ID")
    if provider_raw is None:
        if os.getenv("ELEVENLABS_API_KEY", "").strip():
            provider_raw = SpeechProvider.ELEVENLABS.value
        else:
            provider_raw = SpeechProvider.STUB.value
    return SpeechProfile(
        provider=SpeechProvider(provider_raw.strip().lower()),
        voice_id=voice_id or None,
    )
