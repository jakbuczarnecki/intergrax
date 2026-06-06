# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register deepgram in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.speech_provider.deepgram.bundle import create_deepgram_speech_provider
from intergrax.integrations.providers.speech_provider.deepgram.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_deepgram_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_deepgram_speech_provider, override=override)
