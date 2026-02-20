# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class EvalResult:
    """
    Immutable result of a single evaluation case execution.

    This structure is runtime-agnostic and contains only
    deterministic execution metrics collected after run completion.
    """

    case_id: str
    success: bool
    final_answer: str
    total_tokens: int
    total_cost: float
    tool_calls_count: int
    error: Optional[str] = None