# © Artur Czarnecki. All rights reserved.

"""Context Compiler domain models (Phase MEM-DEPTH-1.1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List

from intergrax.llm.messages import ChatMessage


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


class DegradationStepKind(str, Enum):
    """Ordered degradation ladder steps (MEMORY canon §8.2)."""

    FULL = "full"
    DROP_OPTIONAL_INJECTIONS = "drop_optional_injections"
    REDUCE_INJECTION_BLOCKS = "reduce_injection_blocks"
    TRUNCATE_OLDEST_HISTORY = "truncate_oldest_history"
    DROP_LOWEST_SCORED = "drop_lowest_scored"
    TOKENIZER_HARD_TRIM = "tokenizer_hard_trim"


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
