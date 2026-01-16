# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, TYPE_CHECKING

from intergrax.llm.messages import ChatMessage
from intergrax.prompts.registry.yaml_registry import YamlPromptRegistry

if TYPE_CHECKING:
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
    - Provide system prompt loaded from Prompt Registry.
    - Respect localization and pinning handled by registry.
    - Keep interface stable regardless of prompt source.
    """

    def __init__(
        self,
        config: RuntimeConfig,
        prompt_registry: YamlPromptRegistry,
    ) -> None:
        self._config = config
        self._prompt_registry = prompt_registry

    def build_history_summary_prompt(
        self,
        *,
        request: RuntimeRequest,
        strategy: HistoryCompressionStrategy,
        older_messages: List[ChatMessage],
        tail_messages: List[ChatMessage],
    ) -> HistorySummaryPromptBundle:
        """
        Build prompt bundle using prompt registry.

        Current implementation ignores request/strategy/messages,
        but keeps signature for future advanced customization.
        """

        localized = self._prompt_registry.resolve_localized(
            prompt_id="history_summary",
        )

        # We use only system part for now, but content may contain
        # developer/user_template in the future.
        return HistorySummaryPromptBundle(
            system_prompt=localized.system
        )
