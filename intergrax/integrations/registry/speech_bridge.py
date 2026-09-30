# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Sanctioned speech integration bridge composition (EBH-3-R2)."""

from __future__ import annotations

from intergrax.integrations._shared.speech_integration_bridge import (
    IntegrationSpeechAdapter,
    infer_speech_provider_slug,
)

__all__ = [
    "IntegrationSpeechAdapter",
    "infer_speech_provider_slug",
]
