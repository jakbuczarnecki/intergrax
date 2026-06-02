# © Artur Czarnecki. All rights reserved.

"""Generative multimodal capability flags for Plane A LLM adapters (Phase W-ML.1)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ModalityCapableAdapter(Protocol):
    """Optional capability surface for multimodal LLM adapters."""

    def supports_vision(self) -> bool:
        ...

    def supports_audio_input(self) -> bool:
        ...

    def supports_audio_output(self) -> bool:
        ...
