# © Artur Czarnecki. All rights reserved.

"""Provider-neutral native tool-choice semantics and adapter projection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Union

NativeToolChoiceLiteral = Literal["auto", "required", "none"]


@dataclass(frozen=True, slots=True)
class NativeForcedFunctionChoice:
    """Semantic intent: model must emit exactly one named native function."""

    function_name: str


NativeToolChoice = Union[NativeToolChoiceLiteral, NativeForcedFunctionChoice]


def project_native_tool_choice_for_provider(
    choice: NativeToolChoice | None,
    *,
    provider: str,
) -> str | dict[str, str] | None:
    """Translate canonical Nexus tool-choice intent to provider wire format."""
    if choice is None:
        return None
    if isinstance(choice, NativeForcedFunctionChoice):
        provider_slug = provider.strip().lower()
        if provider_slug in {"ollama", "native_ollama"}:
            return "required"
        return {"type": "function", "name": choice.function_name}
    return choice
