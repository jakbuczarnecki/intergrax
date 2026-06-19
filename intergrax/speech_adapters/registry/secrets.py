# © Artur Czarnecki. All rights reserved.

"""Secret resolution for speech adapters (mirrors ``llm_adapters.registry.secrets``)."""

from __future__ import annotations

import os
from typing import Mapping

from intergrax.speech_adapters.providers.elevenlabs_speech import ElevenLabsSpeechAdapter


def resolve_api_key(
    provider_slug: str,
    secrets: Mapping[str, str] | None,
) -> str | None:
    if secrets and secrets.get("api_key"):
        return secrets["api_key"]
    if provider_slug.strip().lower() == "elevenlabs":
        env_key = os.getenv(ElevenLabsSpeechAdapter.ENV_API_KEY, "").strip()
        return env_key or None
    return None


def merge_secrets_into_options(
    provider_slug: str,
    options: dict[str, object],
    secrets: Mapping[str, str] | None,
) -> dict[str, object]:
    merged = dict(options)
    api_key = resolve_api_key(provider_slug, secrets)
    if api_key:
        merged.setdefault("api_key", api_key)
    return merged
