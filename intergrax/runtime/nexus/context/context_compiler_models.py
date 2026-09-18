# © Artur Czarnecki. All rights reserved.

"""Context Compiler domain models (Phase MEM-DEPTH-1.1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List

from intergrax.context.budget.contracts import DegradationStepKind
from intergrax.llm.messages import ChatMessage

__all__ = (
    "ContextCandidate",
    "ContextCandidateSource",
    "ContextCompileResult",
    "ContextPreflightResult",
    "DegradationStepKind",
)


class ContextCandidateSource(str, Enum):
    """Origin of a context fragment in the compile pass."""

    SYSTEM_INSTRUCTIONS = "system_instructions"
    SESSION_HISTORY = "session_history"
    LONGTERM_MEMORY = "longterm_memory"
    RAG = "rag"
    WEBSEARCH = "websearch"
    ATTACHMENTS = "attachments"
    TOOLS = "tools"
    USER_TURN = "user_turn"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class ContextCandidate:
    """Single message classified for budget allocation."""

    source: ContextCandidateSource
    message_index: int
    score: float
    token_estimate: int
    mandatory: bool


@dataclass(frozen=True, slots=True)
class ContextCompileResult:
    """Output of a ContextCompiler pass."""

    messages: List[ChatMessage]
    total_tokens: int
    budget_tokens: int
    degradation_steps: tuple[str, ...] = ()
    trimmed: bool = False
    bytes_removed: int = 0


@dataclass(frozen=True, slots=True)
class ContextPreflightResult:
    """Pre-LLM invariant check."""

    ok: bool
    assembled_tokens: int
    max_output_tokens: int
    context_window: int
    margin_tokens: int
    message: str = ""
