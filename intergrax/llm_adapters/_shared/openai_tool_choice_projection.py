# © Artur Czarnecki. All rights reserved.

"""OpenAI-compatible wire projection for canonical native tool-choice semantics."""

from __future__ import annotations

from intergrax.llm_adapters.contracts.native_tool_choice import (
    NativeForcedFunctionChoice,
    NativeToolChoice,
)


def project_openai_compatible_tool_choice(
    choice: NativeToolChoice | None,
) -> str | dict[str, str] | None:
    """Map canonical tool-choice intent to OpenAI-compatible request ``tool_choice``."""
    if choice is None:
        return None
    if isinstance(choice, NativeForcedFunctionChoice):
        return {"type": "function", "name": choice.function_name}
    return choice
