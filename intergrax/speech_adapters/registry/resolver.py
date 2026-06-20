# © Artur Czarnecki. All rights reserved.

"""Resolve ``SpeechProviderBackend`` instances from integration catalog slugs."""

from __future__ import annotations

from typing import Any, Mapping

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.contracts.speech_provider import SpeechProviderBackend
from intergrax.integrations.core.binding import IntegrationBinding
from intergrax.integrations.registry.profile import IntegrationProfile


def resolve_speech_provider_backend(
    binding: IntegrationBinding | str,
    *,
    options: Mapping[str, Any] | None = None,
) -> SpeechProviderBackend:
    """Instantiate a catalog ``speech_provider`` backend from slug or binding."""
    profile_options: dict[str, dict[str, Any]] = {}
    if options:
        if isinstance(binding, str):
            profile_options[binding.strip().lower()] = dict(options)
        else:
            slug = binding.resolved_slug()
            if slug is not None:
                profile_options[slug] = dict(options)
    profile = IntegrationProfile(speech_provider=binding, options=profile_options)
    return profile.resolve(IntegrationCategory.SPEECH_PROVIDER)
