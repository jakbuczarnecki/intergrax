# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Neutral execution request contract (UE-2A)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Generic, TypeVar

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class ExecutionCapability(str, Enum):
    """
    Semantic execution requirements consumed by strategy resolution (UE-3A).

    Describes WHAT capabilities the work requires — not HOW the platform
    selects inference, agentic, or orchestration executors.
    """

    TOOLS = "tools"
    ORCHESTRATION = "orchestration"
    STREAMING = "streaming"


@dataclass(frozen=True, slots=True)
class ExecutionRequest(Generic[InputT, OutputT]):
    """
    Canonical neutral execution request describing requested work (UE-2A).

    Expresses developer intent and explicit requirements without encoding
    executor selection, orchestration engine choice, or tool-invocation strategy.
    """

    input: InputT
    output_type: type[OutputT] | None = None
    capabilities: frozenset[ExecutionCapability] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if not isinstance(self.capabilities, frozenset):
            object.__setattr__(self, "capabilities", frozenset(self.capabilities))
