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


def native_tool_choice_function_name(choice: NativeToolChoice | None) -> str | None:
    """Return the forced function name when tool choice pins a single tool."""
    if isinstance(choice, NativeForcedFunctionChoice):
        return choice.function_name
    return None
