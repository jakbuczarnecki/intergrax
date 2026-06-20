# © Artur Czarnecki. All rights reserved.

"""Declarative Tier-3 speech provider selection (catalog slug or binding)."""

from __future__ import annotations

import os
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.contracts.speech_provider import SpeechProviderBackend
from intergrax.integrations.core.binding import IntegrationBinding
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.speech_adapters.contracts.speech_adapter import SpeechAdapter
from intergrax.speech_adapters.registry.resolver import resolve_speech_provider_backend
from intergrax.speech_adapters.registry.secrets import merge_secrets_into_options
from intergrax.speech_adapters.registry.speech_adapter_registry import SpeechAdapterRegistry

SPEECH_PROFILE_EXTRA_KEY = "speech_profile"


class SpeechProfile(BaseModel):
    """
    Typed speech provider slug + voice defaults, or integration binding / live backend.

    Example::

        profile = SpeechProfile(provider_slug="elevenlabs", voice_id="my-voice")
        adapter = profile.create_adapter(secrets={"api_key": "..."})
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    provider_slug: str = "stub"
    voice_id: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)
    binding: IntegrationBinding | None = None
    backend: SpeechProviderBackend | None = Field(default=None, repr=False)

    @field_validator("provider_slug", mode="before")
    @classmethod
    def _coerce_provider_slug(cls, value: str) -> str:
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
        raise ValueError("provider_slug must be a non-empty string")

    @model_validator(mode="after")
    def _align_slug_with_binding(self) -> SpeechProfile:
        if self.binding is not None:
            binding_slug = self.binding.resolved_slug()
            if binding_slug is not None and binding_slug != self.provider_slug:
                return self.model_copy(update={"provider_slug": binding_slug})
        return self

    def resolved_provider_slug(self) -> str:
        if self.binding is not None:
            binding_slug = self.binding.resolved_slug()
            if binding_slug is not None:
                return binding_slug
        return self.provider_slug

    def create_adapter(
        self,
        *,
        secrets: Mapping[str, str] | None = None,
        **overrides: Any,
    ) -> SpeechAdapter:
        from intergrax.integrations._shared.speech_integration_bridge import IntegrationSpeechAdapter

        slug = self.resolved_provider_slug()
        if self.backend is not None:
            return IntegrationSpeechAdapter(self.backend, provider_slug=slug)

        if self.binding is not None:
            backend = IntegrationProfile(
                speech_provider=self.binding,
                options=self._profile_options(secrets=secrets, overrides=overrides),
            ).resolve(IntegrationCategory.SPEECH_PROVIDER)
            return IntegrationSpeechAdapter(backend, provider_slug=slug)

        kwargs = merge_secrets_into_options(slug, {**self.options, **overrides}, secrets)
        if self.voice_id:
            kwargs.setdefault("default_voice_id", self.voice_id)
        if slug == "elevenlabs" and not kwargs.get("api_key"):
            raise ValueError(
                "ElevenLabs speech requires api_key in options, secrets=, or ELEVENLABS_API_KEY"
            )

        if SpeechAdapterRegistry.is_registered(slug):
            return SpeechAdapterRegistry.create(slug, **kwargs)

        backend = resolve_speech_provider_backend(slug, options=kwargs)
        return IntegrationSpeechAdapter(backend, provider_slug=slug)

    def with_secrets(self, secrets: Mapping[str, str]) -> SpeechProfile:
        slug = self.resolved_provider_slug()
        return self.model_copy(
            update={"options": merge_secrets_into_options(slug, dict(self.options), secrets)}
        )

    @classmethod
    def from_binding(
        cls,
        binding: IntegrationBinding,
        *,
        voice_id: str | None = None,
        options: Mapping[str, Any] | None = None,
        provider_slug: str | None = None,
    ) -> SpeechProfile:
        slug = provider_slug or binding.resolved_slug() or "stub"
        return cls(
            provider_slug=slug,
            voice_id=voice_id,
            binding=binding,
            options=dict(options or {}),
        )

    @classmethod
    def from_backend(
        cls,
        backend: SpeechProviderBackend,
        *,
        provider_slug: str,
        voice_id: str | None = None,
    ) -> SpeechProfile:
        return cls(
            provider_slug=provider_slug,
            voice_id=voice_id,
            backend=backend,
        )

    @classmethod
    def lab(cls) -> SpeechProfile:
        """Laboratory default — stub TTS/STT unless API key is configured."""
        return cls(provider_slug="stub")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> SpeechProfile:
        return cls.model_validate(dict(data))

    def _profile_options(
        self,
        *,
        secrets: Mapping[str, str] | None,
        overrides: Mapping[str, Any],
    ) -> dict[str, dict[str, Any]]:
        slug = self.resolved_provider_slug()
        merged = merge_secrets_into_options(slug, {**self.options, **dict(overrides)}, secrets)
        if self.voice_id:
            merged.setdefault("default_voice_id", self.voice_id)
        return {slug: merged}


def speech_profile_from_env(*, prefix: str = "INTERGRAX_SPEECH") -> SpeechProfile:
    """
    Build profile from environment:

    - ``{PREFIX}_PROVIDER`` — integration catalog slug (default ``stub`` unless ``ELEVENLABS_API_KEY`` set)
    - ``{PREFIX}_VOICE_ID`` (optional default voice)
    """
    provider_raw = os.getenv(f"{prefix}_PROVIDER")
    voice_id = os.getenv(f"{prefix}_VOICE_ID")
    if provider_raw is None:
        if os.getenv("ELEVENLABS_API_KEY", "").strip():
            provider_raw = "elevenlabs"
        else:
            provider_raw = "stub"
    return SpeechProfile(
        provider_slug=provider_raw.strip().lower(),
        voice_id=voice_id or None,
    )
