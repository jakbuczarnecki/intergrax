# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol

from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.responses.response_schema import (
    RuntimeRequest,
    HistoryCompressionStrategy,
)


@dataclass
class HistorySummaryPromptBundle:
    """
    Container for prompt elements related to history optimization
    (mainly summarization of older conversation turns).

    For now it only carries a single system_prompt string, but this
    structure allows us to extend it later (e.g. additional guardrails,
    style hints, etc.) without changing the interface.
    """

    system_prompt: str


class HistorySummaryPromptBuilder(Protocol):
    """
    Strategy interface for building the history-summary-related part
    of the prompt.

    You can provide a custom implementation and pass it to
    RuntimeEngine to fully control:

    - the exact system prompt text used when summarizing older history,
    - how the request / strategy / message splits influence that prompt.
    """

    def build_history_summary_prompt(
        self,
        *,
        request: RuntimeRequest,
        strategy: HistoryCompressionStrategy,
        older_messages: List[ChatMessage],
        tail_messages: List[ChatMessage],
    ) -> HistorySummaryPromptBundle:
        ...


class DefaultHistorySummaryPromptBuilder(HistorySummaryPromptBuilder):
    """
    Default prompt builder for history summarization in nexus Mode.

    Responsibilities:
    - Provide a safe, generic system prompt for summarizing older
      conversation turns into an information-dense summary.
    - Ignore request / strategy / messages for now (but the signature
      allows future, more advanced implementations).
    """

    def __init__(self, config: RuntimeConfig) -> None:
        self._config = config

    def build_history_summary_prompt(
        self,
        *,
        request: RuntimeRequest,
        strategy: HistoryCompressionStrategy,
        older_messages: List[ChatMessage],
        tail_messages: List[ChatMessage],
    ) -> HistorySummaryPromptBundle:
        # For now we return a static, default prompt. Later we can use
        # fields from `request` or `config` (e.g. domain, language,
        # user preferences) to customize the text.
        system_prompt = (
            "You are a summarization assistant.\n"
            "Summarize the following conversation history into a short, "
            "factual bullet list that preserves key decisions, key facts, "
            "and open questions.\n"
            "Do not invent new facts. Do not change the meaning.\n"
            "Keep the summary compact and information-dense."
        )

        return HistorySummaryPromptBundle(system_prompt=system_prompt)
