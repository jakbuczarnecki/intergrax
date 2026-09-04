# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register elevenlabs in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.speech_provider.elevenlabs.bundle import create_elevenlabs_speech_provider
from intergrax.integrations.providers.speech_provider.elevenlabs.contract_spec import CONTRACT_SPECS
from intergrax.integrations.providers.speech_provider.elevenlabs.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_elevenlabs_integration(*, override: bool = False) -> None:
    register_from_manifest(
        MANIFEST,
        create_elevenlabs_speech_provider,
        override=override,
        contract_specs=CONTRACT_SPECS,
    )
