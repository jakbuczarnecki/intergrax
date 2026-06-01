# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from intergrax.runtime.nexus.planning.stepplan_models import (
    OutputFormat,
    WebSearchStrategy,
)


@dataclass(frozen=True)
class StepPlannerConfig:
    """
    Rule-based step planner configuration.
    Keep it deterministic; no LLM prompting here.
    """

    # Output style
    final_answer_style: str = "concise_technical"
    final_format: OutputFormat = OutputFormat.MARKDOWN

    # Default per-step budgets
    step_max_chars: int = 2000

    web_top_k: int = 5
    web_max_results: int = 5
    web_recency_days: int = 30
    web_strategy: WebSearchStrategy = WebSearchStrategy.HYBRID

    # Plan-level budgets
    max_total_steps: int = 6
    max_total_tool_calls: int = 3
    max_total_web_queries: int = 5
    max_total_chars_context: int = 12000
    max_total_tokens_output: Optional[int] = None


