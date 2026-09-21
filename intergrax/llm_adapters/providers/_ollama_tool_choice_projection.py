# © Artur Czarnecki. All rights reserved.

"""Ollama-native wire projection for canonical tool-choice semantics."""

from __future__ import annotations

from intergrax.llm_adapters.contracts.native_tool_choice import (
    NativeForcedFunctionChoice,
    NativeToolChoice,
    NativeToolChoiceLiteral,
)


def project_ollama_native_tool_choice(
    choice: NativeToolChoice | None,
) -> NativeToolChoiceLiteral | None:
    """Map canonical tool-choice intent to Ollama ``tool_choice`` wire values."""
    if choice is None:
        return None
    if isinstance(choice, NativeForcedFunctionChoice):
        return "required"
    return choice
